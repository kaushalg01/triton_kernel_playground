import torch
import triton
import triton.language as t1
DEVICE = torch.device("cuda")
@triton.jit

    
def indexer_kernel(
    q_index_fp8_ptr,
    k_index_cache_fp8_ptr,
    weights_ptr,
    seq_lens_ptr,
    block_table_ptr,
    #this stores the seq length cumulative sum to calculate batch_offset
    seq_offsets_ptr,
    #this stores the accumulated result per token
    acc_ptr,

    batch_size,
    num_index_heads,
    index_head_dim,
    num_pages,
    page_size,
    kv_cache_num_heads,
    head_dim_with_scale,
    max_num_pages,

    #defines the number of dimensions processed at once
    #number of index_head_dimension processed at once
    BLOCK_SIZE: t1.constexpr,
    #defines the number of K tokens present in a K-tile
    BLOCK_TOKENS: t1.constexpr,
    #number of heads processed in one program of query
    BLOCK_HEADS: t1.constexpr
):
    #load a tile of q in a program having multiple heads that stream over k tiles and accumulate their sum per token
    #divide the kv cache into tiles, load it, and then dequantise + upscale it
    #for this, load the pagess for the given sequence
    #start going through the batch of q sequences, go to the first sequence, fetch its kv cache pages
    #in each parallel program, we go over each q sequence of the batch, select a program having multiple heads and fetch its k pages, divide it into tile-streaming
    # and perform computations
    #note: each program computes each sequence's dot product with its k values along one query index-head group, per page of K 
    #total such programs needed are: batch_size x num_index_heads / block_heads
    #launch a grid of (B * H/BH)
    #NOTE: In this implementation we launch programs across different query head blocks while streaming through K tiles one-by-one.
    # Reason is, once K tile is brought into L2 cache, it gets reused by all the heads running in parallel.
    # Had we run multiple parallel programs across different K tiles, we would not have the provision of reusing K tiles.
    #visual layout of a batch of Q
    #          head_id
    #        0   1   2
    #batch_id
    #0       h0  h1  h2
    #1       h3  h4  h5
    #2       h6  h7  h8
    #3       h9  h10 h11

    pid = t1.program_id(0)
    #BLOCK_HEADS defines the number of heads executing in one program in parallel
    HEAD_BLOCK = BLOCK_HEADS
    batch_id = pid // (num_index_heads // HEAD_BLOCK)
    head_block_id = pid % (num_index_heads // HEAD_BLOCK)

    #define the offset of each head present in the HEAD_BLOCK program
    offs_h = head_block_id * HEAD_BLOCK + t1.arange(0, HEAD_BLOCK)
    mask_h = offs_h < num_index_heads

    #number of tokens per sequence present in KV cache as pages
    seq_len = t1.load(seq_lens_ptr + batch_id)
    num_pages_per_seq = (seq_len + page_size - 1) // page_size
    #loading weight_ptr to be used later per K tile per HEAD_BLOCK head dimensions
    weight_ptr = weights_ptr + batch_id * num_index_heads + offs_h
    head_weight = t1.load(weight_ptr, mask=mask_h)

    #cumulative sum of seq_lens to calculate global_token_ids
    seq_start = t1.load(seq_offsets_ptr + batch_id)
    #stores the final score of QKt

    for page_id in range(0, num_pages_per_seq):
        #even though we have varying page length per sequence, we pad it with 0 to keep the block table uniform
        page_index = t1.load(
            block_table_ptr + batch_id * max_num_pages + page_id
        )
        #kv cache stores on a page basis, having different tokens, per head, given head dimension
        k_page_ptr = k_index_cache_fp8_ptr + (page_index * page_size * head_dim_with_scale * kv_cache_num_heads) 
        #bringing in a tile of selected K page
        for token_start in range(0, page_size, BLOCK_TOKENS):
            scores = t1.zeros([BLOCK_HEADS, BLOCK_TOKENS], t1.float32)
            #defines the per token index in a page, shape: BLOCK_TOKENS,1 
            offset_token = token_start + t1.arange(0, BLOCK_TOKENS)
            #every iteration requires scale pointer to upscale the K values
            scale_ptrs = (
                k_page_ptr
                + offset_token * head_dim_with_scale
                + index_head_dim        # last element = scale
            )
            token_mask = offset_token < seq_len
            scale_vals = t1.load(scale_ptrs,
                    mask=token_mask, 
                    other=0.0
            )
            acc_scores = t1.zeros([BLOCK_HEADS, BLOCK_TOKENS], t1.float32)
            # go to the start of each head in a tensor and make each row BLOCK_SIZE head dimension 
            for d_start in range(0, index_head_dim, BLOCK_SIZE):
            #computes the offset HEAD dimension for a given BLOCK_HEAD
                offs_d = d_start + t1.arange(0, BLOCK_SIZE) 
                # compute a tensor their Q-pointers
                q_ptrs = (
                    q_index_fp8_ptr
                    + batch_id * num_index_heads * index_head_dim
                    + offs_h[:, None] * index_head_dim
                    + offs_d[None, :]
                )

                q_tile = t1.load(
                    q_ptrs,
                    mask=mask_h[:, None] & (offs_d[None, :] < index_head_dim),
                    other=0.0
                )
                #shape = [BLOCK_HEADS, BLOCK_SIZE]
                #Loading K tiles from the intended pages
                #defines the head dimensions to be brought-in per token in different iterations, shape: 1, BLOCK_SIZE
                off_dim = d_start + t1.arange(0, BLOCK_SIZE)
                #separating fp8 and scale_values for the last iteration of index_head dimension
                #define the starting pointer and broadcast it to convert each token to BLOCK_SIZE wide row
                #this finds the starting k pointer of a given token and creates the 2-d K tile
                k_ptrs = (
                    k_page_ptr
                    + offset_token[:, None] * head_dim_with_scale
                    + off_dim[None, :]
                )
                #load tile such that any row outside seq_len is masked and any column outside head_dim is masked
                k_tile = t1.load(
                    k_ptrs,
                    mask=(offset_token[:, None] < seq_len) &
                    #scale pointer is never brought in
                    (off_dim[None, :] < index_head_dim),
                    other=0.0
                )
                #dequantising them
                fp8_vals = k_tile.to(t1.float8e4nv)
                k_vals = fp8_vals.to(t1.float16) * scale_vals[:, None]
                #scores is heads x seq_length being summed across head dimension iteration
                scores += t1.dot(q_tile, k_vals.T)
            scores = t1.maximum(scores, 0.0)
            scores *= head_weight[:, None]
            # add the sum across all heads per token 
            token_scores = t1.sum(scores, axis=0)
            #global_token_id in the K cache
            global_token_ids = seq_start + offset_token

            t1.atomic_add(
                acc_ptr + global_token_ids,
                token_scores
            )
            #we store per token sum in an accumulator sum and perform atomic adds to it
                    
def topk_kernel(
    acc_ptr,
    seq_offsets_ptr,
    seq_lens_ptr,
    topk_scores_ptr,
    topk_indices_ptr,
    #runtime parameter for actual topK selection
    K,
    #number of tiles to be brought-in from K page
    BLOCK_TOKENS: t1.constexpr,
    #use MAX_K for register allocation during static runtime
    MAX_K: t1.constexpr,
):

    batch_id = t1.program_id(0)

    seq_start = t1.load(seq_offsets_ptr + batch_id)
    seq_len   = t1.load(seq_lens_ptr + batch_id)

    # running topK buffers
    top_scores = t1.full([MAX_K], -1e9, t1.float32)
    top_indices = t1.zeros([MAX_K], t1.int32)

    for token_start in range(0, seq_len, BLOCK_TOKENS):

        offs = token_start + t1.arange(0, BLOCK_TOKENS)
        mask = offs < seq_len

        scores = t1.load(
            acc_ptr + seq_start + offs,
            mask=mask,
            other=-1e9,
        )

        global_ids = seq_start + offs

        # merge old + new candidates
        merged_scores = t1.cat(top_scores, scores)
        merged_indices = t1.cat(top_indices, global_ids)

        # select topK
        order = t1.argsort(merged_scores, descending=True)

        top_scores = t1.gather(merged_scores, order[:K])
        top_indices = t1.gather(merged_indices, order[:K])

    # store results
    t1.store(
        topk_scores_ptr + batch_id * K + t1.arange(0, K),
        top_scores[:K],
    )

    t1.store(
        topk_indices_ptr + batch_id * K + t1.arange(0, K),
        top_indices[:K],
    )

def run_indexer_and_topk(
    q_index_fp8,           # [batch_size, num_index_heads, index_head_dim] fp8
    k_index_cache_fp8,     # [num_pages * page_size * head_dim_with_scale] fp8
    weights,               # [batch_size, num_index_heads] float32
    seq_lens,              # [batch_size] int32
    block_table,           # [batch_size, max_num_pages] int32
    seq_offsets,           # [batch_size] int32, cumulative sum of seq_lens
    batch_size,
    num_index_heads,
    index_head_dim,
    num_pages,
    page_size,
    kv_cache_num_heads,
    head_dim_with_scale,
    max_num_pages,
    topk,                  # number of top-k tokens to select
    BLOCK_SIZE=32,
    BLOCK_TOKENS=32,
    BLOCK_HEADS=4,
    device='cuda',
):
    # Step 0: allocate accumulator for per-token scores
    max_seq_len = seq_lens.max()
    acc = torch.zeros((batch_size, max_seq_len), device=device, dtype=torch.float32)

    # Step 1: Launch the indexer kernel per sequence per BLOCK_HEADS
    grid = (batch_size * (num_index_heads // BLOCK_HEADS),)
    indexer_kernel[grid](
        q_index_fp8_ptr=q_index_fp8,
        k_index_cache_fp8_ptr=k_index_cache_fp8,
        weights_ptr=weights,
        seq_lens_ptr=seq_lens,
        block_table_ptr=block_table,
        seq_offsets_ptr=seq_offsets,
        acc_ptr=acc,
        batch_size=batch_size,
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        num_pages=num_pages,
        page_size=page_size,
        kv_cache_num_heads=kv_cache_num_heads,
        head_dim_with_scale=head_dim_with_scale,
        max_num_pages=max_num_pages,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_TOKENS=BLOCK_TOKENS,
        BLOCK_HEADS=BLOCK_HEADS,
    )

    # Step 2: Allocate output tensor for top-k indices per batch_size
    topk_indices = torch.zeros((batch_size, topk), device=device, dtype=torch.int32)

    # Step 3: Launch topk kernel per batch
    # Triton prefers 1D grid, we launch batch_size programs, each processing one sequence
    topk_grid = (batch_size,)
    topk_kernel[topk_grid](
        acc_ptr=acc,
        topk_indices_ptr=topk_indices,
        seq_lens_ptr=seq_lens,
        topk=topk,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_TOKENS=BLOCK_TOKENS,
    )

    return topk_indices