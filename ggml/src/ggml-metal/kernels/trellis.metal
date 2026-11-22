// trellis.metal — Metal kernels for EXL3 (exllamav3 v1.x) packed weights.
//
// Paso 2b of the Trellis-for-Metal fork. Ports of the signed MLX kernels:
//   kernel_trellis_gemv    <- Engine/injerto35/kernel_gemv_mcg.py (GEMV opt4)
//   kernel_trellis_decode  <- Engine/injerto35/kernel_lote_mcg.py (decode-a-scratch)
// The H128 transforms are NOT here: they reuse the stock kernel_fwht
// (kernels/misc.metal, same normalized H128, scale 1/sqrt(N)) through the
// GGML_OP_TRELLIS_FWHT op. The suh/svh scales are graph-level MUL ops.
//
// Format (per linear, W is (k, n), y = x @ W):
//   trellis : int16 flat [tk*tn*16*K], tk = k/16, tn = n/16, one 16x16 tile
//             per (i, j); tile bitstream = uint32 words (LE pairs of int16),
//             bits MSB-first; element t decodes from the 16-bit window that
//             ENDS at bit (t+1)*K (circular in the tile); element t lands at
//             tile position perm[t] (tensor-core order).
//   K  : trellis depth per element (3 for the 35B, 6 for its lm_head)
//   cb : codebook, 0 = 3INST, 1 = MCG, 2 = MUL1 (same as reference.py)

#include "common.h"

// fp16 value for one 16-bit trellis state. Mirrors reference.py codebook_lut
// and the 1b C++ decode: the arithmetic is done in float (Metal has no
// double) and rounded once to fp16, exactly like the Python float64 path
// up to the last rounding (the MUL1 add can round once in fp32 first;
// that path is not judged in 2b — the 35B is mcg).
inline float trellis_value(uint state, uint cb) {
    if (cb == 2u) { // MUL1
        uint x = state * 0x83DCD12Du;
        uint bsum = (x & 0xFFu) + ((x >> 8) & 0xFFu)
                  + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
        half inv  = as_type<half>((ushort) 0x1EEE);
        half bias = as_type<half>((ushort) 0xC931);
        return float((half) ((1024.0f + (float) bsum) * (float) inv + (float) bias));
    }
    uint x;
    if (cb == 0u) x = state * 89226354u + 64248484u; // 3INST
    else          x = state * 0xCBAC1FEDu;           // MCG
    x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
    half lo = as_type<half>((ushort) (x & 0xFFFFu));
    half hi = as_type<half>((ushort) (x >> 16));
    return float((half) ((float) lo + (float) hi));
}

// GEMV in the Hadamard domain: out[n] = xh @ W_inner, one row.
//   trellis : int16 flat [tk*tn*16*K]
//   xh      : float32 [k], already H128-transformed (see TRELLIS_FWHT)
//   out     : float32 [n]
// Grid: (tn) threadgroups of 256 threads; threadgroup nj owns output
// columns [nj*16, nj*16+16). Literal port of the MLX opt4 mcg kernel.
kernel void kernel_trellis_gemv(
        constant ggml_metal_kargs_trellis_gemv & args,
        device const uint16_t * trellis,
        device const float * xh,
        device float * out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tptg  [[thread_position_in_threadgroup]]) {
    const uint tid = tptg.x;
    const uint nj = tgpig.x;
    const uint tk = (uint) args.tk;
    const uint tn = (uint) args.tn;
    const uint K  = (uint) args.K;
    const uint cb = (uint) args.cb;

    device const uint * words32 = (device const uint *) trellis;
    threadgroup uint  wsh[48]; // 8*K words; K<=6 (40 overflowed on lm_head)
    threadgroup float xsh[16];
    threadgroup float red[256];

    const uint r = tid >> 4;
    const uint c = tid & 15u;
    // inverse of the tensor-core permutation: (r, c) -> stream position t
    const uint c0  = c & 7u;
    const uint hic = c >> 3;
    const uint r0  = (r & 7u) & 6u;
    const uint rr  = r - r0;
    const uint j   = hic * 4u + (rr < 8u ? rr : rr - 6u);
    const uint t   = (c0 * 4u + (r0 >> 1)) * 8u + j;
    const uint nbits = 256u * K;
    const uint b0 = (t * K + K - 16u + nbits) % nbits;
    const uint nw = 8u * K;
    const uint q  = b0 >> 5;
    const uint s  = b0 & 31u;

    float acc = 0.0f;
    for (uint ti = 0; ti < tk; ++ti) {
        const uint wbase = (ti * tn + nj) * 8u * K;
        if (tid < 8u * K) wsh[tid] = words32[wbase + tid];
        if (tid < 16u)   xsh[tid] = xh[ti * 16u + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint A = wsh[q];
        const uint B = wsh[q + 1u == nw ? 0u : q + 1u];
        const uint X = (A << s) | (s == 0u ? 0u : (B >> (32u - s)));
        const uint state = X >> 16;
        const float v = trellis_value(state, cb);
        red[tid] = v * xsh[r];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (r == 0u) {
            float sm = 0.0f;
            for (uint i = 0; i < 16u; ++i) sm += red[i * 16u + c];
            acc += sm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r == 0u) out[nj * 16u + c] = acc;
}

// Decode W_inner to fp16 scratch [k, n] (prefill path; the batch matmul
// then runs through the stock mul_mm). One thread per element.
// Grid: (tk*tn) threadgroups of 256 threads.
kernel void kernel_trellis_decode(
        constant ggml_metal_kargs_trellis_decode & args,
        device const uint16_t * trellis,
        device half * out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tptg  [[thread_position_in_threadgroup]]) {
    const uint tid = tptg.x;
    const uint tk = (uint) args.tk;
    const uint tn = (uint) args.tn;
    const uint K  = (uint) args.K;
    const uint cb = (uint) args.cb;

    const uint idx  = tgpig.x * 256u + tid;
    const uint tile = idx >> 8;
    const uint t    = idx & 255u;
    const uint nbits = 256u * K;
    const uint base  = tile * 16u * K;

    uint words[48]; // 8*K; K<=6
    for (uint j = 0; j < 8u * K; ++j) {
        words[j] = (uint) trellis[base + 2u * j]
                 | ((uint) trellis[base + 2u * j + 1u] << 16);
    }
    const uint b0 = (t * K + K - 16u + nbits) % nbits;
    uint state = 0;
    for (uint i = 0; i < 16u; ++i) {
        uint g = b0 + i;
        if (g >= nbits) g -= nbits;
        const uint bit = (words[g >> 5] >> (31u - (g & 31u))) & 1u;
        state = (state << 1) | bit;
    }
    const half v = (half) trellis_value(state, cb);
    // forward tensor-core permutation: stream position t -> (r, c)
    const uint grp = t >> 3;
    const uint j   = t & 7u;
    const uint r0  = (grp & 3u) * 2u;
    const uint c0  = grp >> 2;
    const uint r   = r0 + ((j == 1u || j == 5u) ? 1u :
                           (j == 2u || j == 6u) ? 8u :
                           (j == 3u || j == 7u) ? 9u : 0u);
    const uint c   = c0 + (j >= 4u ? 8u : 0u);
    const uint ti  = tile / tn;
    const uint nj  = tile % tn;
    out[(ti * 16u + r) * (tn * 16u) + nj * 16u + c] = v;
}

// kernel_trellis_fused — Paso 2e, v2 (replaces v1).
//
// v1 FAILED on speed (measured, not assumed): the standalone judge
// (gate_proj) timed the fused graph at 1.208 ms vs 0.439 ms for the
// six-node graph, and the model smoke confirmed it — generation 4.7 tok/s
// vs 13.8 tok/s with the six-node path. Cause: v1 kept the opt4 shared
// tile (wsh) + per-tile reduction (red), i.e. 3 threadgroup barriers per
// (row-tile, sub-tile) — ~3200 sequential barriers per threadgroup —
// while owning only n/128 threadgroups (4 for n=512). The barrier chain,
// not the dispatch count, was the cost.
//
// v2 removes ALL barriers from the GEMV loop:
//  - No shared tile words: each thread loads the 2 uint32 words it needs
//    straight from global memory. The tile window is 8*K words (<= 192 B,
//    1-2 cache lines), so the 256 threads' loads are effectively
//    broadcasts out of L1.
//  - No per-tile reduction: each thread accumulates its own partial sums
//    in registers (acc[8], one per 16-column sub-tile) across ALL tiles;
//    the sum over the 16 rows happens ONCE at the end, through shared
//    memory (part[8*256]) + one barrier.
//  - The only barriers left: Phase A (input FWHT, 9 per 128-block),
//    one per block between Phase B and the next Phase A (xh_sh reuse),
//    and Phase C/D (~9). Total ~170 for k=2048 (v1: ~3200).
// Summation order changes vs v1/opt4 (per-thread accumulation, then one
// 16-way sum); fp32 rounding differences are ~1e-7, inside the 1e-4 judge.
//
// y = H128( H128(x * suh) @ W_inner ) * svh. One threadgroup (256 threads)
// owns 128 output columns, so both H128 transforms stay in threadgroup
// memory. Inputs: trellis I16 flat [tk*tn*16*K], x F32 [k], suh F32 [k],
// svh F32 [n]. Output: y F32 [n]. k and n are multiples of 128;
// K in 2..6 (K=6 is the lm_head). cb: 0=3INST, 1=MCG, 2=MUL1.
kernel void kernel_trellis_fused(
        constant ggml_metal_kargs_trellis_fused & args,
        device const uint16_t * trellis,
        device const float * x,
        device const float * suh,
        device const float * svh,
        device float * out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tptg  [[thread_position_in_threadgroup]]) {
    const uint tid = tptg.x;
    const uint nb  = tgpig.x; // output block: columns [nb*128, nb*128+128)
    const uint tk = (uint) args.tk; // k/16
    const uint tn = (uint) args.tn; // n/16
    const uint K  = (uint) args.K;
    const uint cb = (uint) args.cb;

    device const uint * words32 = (device const uint *) trellis;
    threadgroup float xh_sh[128];  // one 128-block of H128(x*suh)
    threadgroup float part[8 * 256]; // per-thread partials for the final sum
    threadgroup float out_sh[128]; // GEMV outputs before the output H128

    const uint r = tid >> 4;
    const uint c = tid & 15u;
    // inverse of the tensor-core permutation: (r, c) -> stream position t
    // (identical to kernel_trellis_gemv / MLX opt4)
    const uint c0  = c & 7u;
    const uint hic = c >> 3;
    const uint r0  = (r & 7u) & 6u;
    const uint rr  = r - r0;
    const uint j   = hic * 4u + (rr < 8u ? rr : rr - 6u);
    const uint t   = (c0 * 4u + (r0 >> 1)) * 8u + j;
    const uint nbits = 256u * K;
    const uint b0 = (t * K + K - 16u + nbits) % nbits;
    const uint nw = 8u * K;
    const uint q  = b0 >> 5;
    const uint s  = b0 & 31u;

    // per-thread partial sums: one per 16-column sub-tile, accumulated
    // across ALL row-tiles (no reduction until the end)
    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    const uint nbk = tk >> 3; // k/128 input blocks (k % 128 == 0)
    for (uint b = 0; b < nbk; ++b) {
        // Phase A: xh_sh = H128(x[b*128:(b+1)*128] * suh), butterfly
        // in threadgroup memory, scale 1/sqrt(128) at the end (same as
        // the trellis_fwht128 CPU forward and the stock kernel_fwht).
        if (tid < 128u) xh_sh[tid] = x[b * 128u + tid] * suh[b * 128u + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint len = 64u; len >= 1u; len >>= 1) {
            if (tid < 64u) {
                const uint i = (tid / len) * (2u * len) + (tid % len);
                const float u = xh_sh[i];
                const float v = xh_sh[i + len];
                xh_sh[i]       = u + v;
                xh_sh[i + len] = u - v;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid < 128u) xh_sh[tid] *= (1.0f / sqrt(128.0f));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase B: GEMV with on-the-fly trellis decode, no barriers.
        // Each thread reads the 2 words its (r, c) element needs
        // straight from global memory (tile window <= 192 B, L1-cached).
        for (uint t8 = 0; t8 < 8u; ++t8) {
            const uint ti = b * 8u + t8;
            const float xr = xh_sh[t8 * 16u + r];
            for (uint sj = 0; sj < 8u; ++sj) {
                const uint nj = nb * 8u + sj;
                const uint wbase = (ti * tn + nj) * 8u * K;
                const uint A = words32[wbase + q];
                const uint B = words32[wbase + (q + 1u == nw ? 0u : q + 1u)];
                const uint X = (A << s) | (s == 0u ? 0u : (B >> (32u - s)));
                const uint state = X >> 16;
                acc[sj] += trellis_value(state, cb) * xr;
            }
        }
        // xh_sh is rewritten by the next block's Phase A: every thread
        // must have finished reading it first.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Phase C: partials to threadgroup memory, one barrier, then the
    // r==0 threads sum the 16 rows per column into out_sh.
    for (uint sj = 0; sj < 8u; ++sj) part[sj * 256u + tid] = acc[sj];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r == 0u) {
        for (uint sj = 0; sj < 8u; ++sj) {
            float sm = 0.0f;
            for (uint i = 0; i < 16u; ++i) sm += part[sj * 256u + i * 16u + c];
            out_sh[sj * 16u + c] = sm;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase D: H128 on the 128 outputs, same butterfly as Phase A
    for (uint len = 64u; len >= 1u; len >>= 1) {
        if (tid < 64u) {
            const uint i = (tid / len) * (2u * len) + (tid % len);
            const float u = out_sh[i];
            const float v = out_sh[i + len];
            out_sh[i]       = u + v;
            out_sh[i + len] = u - v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 128u) out_sh[tid] *= (1.0f / sqrt(128.0f));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase E: y = out_sh * svh
    if (tid < 128u) out[nb * 128u + tid] = out_sh[tid] * svh[nb * 128u + tid];
}

// kernel_trellis_gemv_nb + kernel_trellis_fwht_svh — Paso 2e, v3.
//
// v3 replaces the v1/v2 single-kernel fusion (TRELLIS_MUL_MAT_FUSED,
// kept defined but unwired: correct, 4.917e-07/8.345e-07, but SLOW —
// smoke generation 4.7 tok/s (v1) / 7.9 tok/s (v2) vs 13.8 tok/s for
// the six-node path). Measured cause (per-node timing test,
// /tmp/test_trellis_timing, N=50 amortized): the six-node chain costs
// 0.152 ms serialized, its GEMV 0.029 ms overlapped, the fused v2
// 0.038 ms overlapped — but in the model the fused kernel is ~5x slower
// per linear than the chain. The difference is SERIALIZED latency: the
// fused kernel owns 128 output columns per threadgroup, so each thread
// does 1024 sequential iterations (16 blocks x 8 tiles x 8 sub-tiles)
// with dependent global loads, and only n/128 = 4 threadgroups exist
// for n=512 — far too little parallelism to hide load latency. Removing
// the barriers (v2) helped 1.7x but not enough: the sequential work per
// thread is the cost, not the barriers.
//
// v3 restores the MLX opt4 parallelism (16 output columns per
// threadgroup, tn threadgroups, 128 iterations per thread) and splits
// the fusion into two nodes:
//   kernel A (GEMV_NB): the GEMV with on-the-fly trellis decode, but
//     with NO barriers in the k-loop: each thread loads the 2 uint32
//     words it needs straight from global memory (tile window <= 192 B,
//     L1-cached) and accumulates into ONE scalar register; the sum over
//     the 16 rows happens ONCE at the end (part[256] + one barrier).
//     This is the v2 loop structure with 16-column tiles instead of 128.
//   kernel B (FWHT_SVH): H128 on the 128 outputs + *svh, one threadgroup
//     per 128-block (fully parallel, ~10 barriers).
// The graph becomes MUL(suh) -> FWHT -> GEMV_NB -> FWHT_SVH: 4 nodes
// per dense linear instead of 6, with the GEMV at opt4 parallelism.
// k and n are multiples of 16 (GEMV_NB) / 128 (FWHT_SVH).

// kernel_trellis_gemv_nb — barrier-free trellis GEMV, 16-col tiles.
// c = xh @ W_inner. Inputs: trellis I16 flat [tk*tn*16*K], xh F32 [k].
// Output: c F32 [n]. K in 2..6, cb: 0=3INST, 1=MCG, 2=MUL1.
kernel void kernel_trellis_gemv_nb(
        constant ggml_metal_kargs_trellis_gemv_nb & args,
        device const uint16_t * trellis,
        device const float * xh,
        device float * out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tptg  [[thread_position_in_threadgroup]]) {
    const uint tid = tptg.x;
    const uint nj  = tgpig.x; // output tile: columns [nj*16, nj*16+16)
    const uint tk = (uint) args.tk; // k/16
    const uint tn = (uint) args.tn; // n/16
    const uint K  = (uint) args.K;
    const uint cb = (uint) args.cb;

    device const uint * words32 = (device const uint *) trellis;
    threadgroup float part[256]; // per-thread partials for the final sum

    const uint r = tid >> 4;
    const uint c = tid & 15u;
    // inverse of the tensor-core permutation: (r, c) -> stream position t
    // (identical to kernel_trellis_gemv / MLX opt4)
    const uint c0  = c & 7u;
    const uint hic = c >> 3;
    const uint r0  = (r & 7u) & 6u;
    const uint rr  = r - r0;
    const uint j   = hic * 4u + (rr < 8u ? rr : rr - 6u);
    const uint t   = (c0 * 4u + (r0 >> 1)) * 8u + j;
    const uint nbits = 256u * K;
    const uint b0 = (t * K + K - 16u + nbits) % nbits;
    const uint nw = 8u * K;
    const uint q  = b0 >> 5;
    const uint s  = b0 & 31u;

    // One scalar accumulator per thread: the sum over ALL row-tiles for
    // this thread's (r, c) element. No shared tile, no per-tile reduce,
    // no barriers in the loop — the 2 global word loads per iteration
    // are independent across iterations and hide behind the 32
    // threadgroups' warps.
    float acc = 0.0f;
    for (uint ti = 0; ti < tk; ++ti) {
        const uint wbase = (ti * tn + nj) * 8u * K;
        const uint A = words32[wbase + q];
        const uint B = words32[wbase + (q + 1u == nw ? 0u : q + 1u)];
        const uint X = (A << s) | (s == 0u ? 0u : (B >> (32u - s)));
        const uint state = X >> 16;
        acc += trellis_value(state, cb) * xh[ti * 16u + r];
    }

    // Single final reduction over the 16 rows, then write this tile's
    // 16 outputs.
    part[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r == 0u) {
        float sm = 0.0f;
        for (uint i = 0; i < 16u; ++i) sm += part[i * 16u + c];
        out[nj * 16u + c] = sm;
    }
}

// kernel_trellis_fwht_svh — H128 on the GEMV outputs + *svh.
// y = H128(c) * svh. Inputs: c F32 [n], svh F32 [n]. Output: y F32 [n].
// One threadgroup per 128-block; butterfly in threadgroup memory,
// scale 1/sqrt(128) at the end (same as trellis_fwht128 / stock fwht).
kernel void kernel_trellis_fwht_svh(
        device const float * c,
        device const float * svh,
        device float * out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tptg  [[thread_position_in_threadgroup]]) {
    const uint tid = tptg.x;
    const uint nb  = tgpig.x; // output block: columns [nb*128, nb*128+128)
    threadgroup float sh[128];

    if (tid < 128u) sh[tid] = c[nb * 128u + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint len = 64u; len >= 1u; len >>= 1) {
        if (tid < 64u) {
            const uint i = (tid / len) * (2u * len) + (tid % len);
            const float u = sh[i];
            const float v = sh[i + len];
            sh[i]       = u + v;
            sh[i + len] = u - v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 128u) sh[tid] *= (1.0f / sqrt(128.0f));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128u) out[nb * 128u + tid] = sh[tid] * svh[nb * 128u + tid];
}


// kernel_trellis_xh_id + kernel_trellis_gemv_nb_id + kernel_trellis_fwht_svh_id
// — Paso 3: MoE experts via trellis.
//
// The expert linear is the dense v3 chain (MUL suh -> FWHT -> GEMV_NB ->
// FWHT_SVH) with per-expert suh/svh/trellis selected by ids[slot, token].
// Three nodes because ggml-metal ops have no scratch mechanism: the
// intermediates (xh, c) are graph tensors, exactly like the dense path
// (proven at 22.2 tok/s in 2e v3).
//
// Layouts (ggml column-major):
//   suh/svh/trellis: flat stacked [n_exp * per_expert_size], expert e
//     slice offset = e * per_expert_size.
//   x:  F32 [k, tokens]          (i, token)        at token*k + i
//   ids: I32 [topk, tokens]      (slot, token)     at slot + token*topk
//   xh/c/y: F32 [dim, topk, tokens] (i, slot, token) at (token*topk+slot)*dim + i
// k and n are multiples of 128 (XH/FWHT_SVH) / 16 (GEMV_NB).

// kernel_trellis_xh_id — xh = H128(x * suh[e]) per (token, slot).
// Grid: (tokens * topk * (k/128)) threadgroups of 256 threads (128 used).
kernel void kernel_trellis_xh_id(
        constant ggml_metal_kargs_trellis_xh_id & args,
        device const float * suh,
        device const float * x,
        device const int * ids,
        device float * out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tptg  [[thread_position_in_threadgroup]]) {
    const uint tid   = tptg.x;
    const uint k     = (uint) args.k;
    const uint topk  = (uint) args.topk;
    const uint nb    = k / 128u;
    const uint blk   = tgpig.x % nb;
    const uint ts    = tgpig.x / nb; // token*topk + slot
    const uint slot  = ts % topk;
    const uint token = ts / topk;
    const uint e     = (uint) ids[slot + token * topk];
    threadgroup float sh[128];

    const uint x_off = (args.per_slot ? ts : token) * k + blk * 128u;
    if (tid < 128u) sh[tid] = x[x_off + tid] * suh[e * k + blk * 128u + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint len = 64u; len >= 1u; len >>= 1) {
        if (tid < 64u) {
            const uint i = (tid / len) * (2u * len) + (tid % len);
            const float u = sh[i];
            const float v = sh[i + len];
            sh[i]       = u + v;
            sh[i + len] = u - v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 128u) out[ts * k + blk * 128u + tid] = sh[tid] * (1.0f / sqrt(128.0f));
}

// kernel_trellis_gemv_nb_id — c = xh @ W_inner[e] per (token, slot).
// Barrier-free GEMV, 16-col tiles (same loop as kernel_trellis_gemv_nb).
// Grid: (tokens * topk * tn) threadgroups of 256 threads.
kernel void kernel_trellis_gemv_nb_id(
        constant ggml_metal_kargs_trellis_gemv_nb_id & args,
        device const uint16_t * trellis,
        device const float * xh,
        device const int * ids,
        device float * out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tptg  [[thread_position_in_threadgroup]]) {
    const uint tid   = tptg.x;
    const uint tk    = (uint) args.tk; // k/16
    const uint tn    = (uint) args.tn; // n/16
    const uint K     = (uint) args.K;
    const uint cb    = (uint) args.cb;
    const uint topk  = (uint) args.topk;
    const uint nj    = tgpig.x % tn; // output tile: columns [nj*16, nj*16+16)
    const uint ts    = tgpig.x / tn; // token*topk + slot
    const uint slot  = ts % topk;
    const uint token = ts / topk;
    const uint e     = (uint) ids[slot + token * topk];
    const uint k     = tk * 16u;
    const uint n     = tn * 16u;

    device const uint * words32 = (device const uint *) trellis + e * tk * tn * 8u * K;
    device const float * xv = xh + ts * k;
    threadgroup float part[256]; // per-thread partials for the final sum

    const uint r = tid >> 4;
    const uint c = tid & 15u;
    // inverse of the tensor-core permutation: (r, c) -> stream position t
    // (identical to kernel_trellis_gemv_nb / MLX opt4)
    const uint c0  = c & 7u;
    const uint hic = c >> 3;
    const uint r0  = (r & 7u) & 6u;
    const uint rr  = r - r0;
    const uint j   = hic * 4u + (rr < 8u ? rr : rr - 6u);
    const uint t   = (c0 * 4u + (r0 >> 1)) * 8u + j;
    const uint nbits = 256u * K;
    const uint b0 = (t * K + K - 16u + nbits) % nbits;
    const uint nw = 8u * K;
    const uint q  = b0 >> 5;
    const uint s  = b0 & 31u;

    float acc = 0.0f;
    for (uint ti = 0; ti < tk; ++ti) {
        const uint wbase = (ti * tn + nj) * 8u * K;
        const uint A = words32[wbase + q];
        const uint B = words32[wbase + (q + 1u == nw ? 0u : q + 1u)];
        const uint X = (A << s) | (s == 0u ? 0u : (B >> (32u - s)));
        const uint state = X >> 16;
        acc += trellis_value(state, cb) * xv[ti * 16u + r];
    }

    part[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r == 0u) {
        float sm = 0.0f;
        for (uint i = 0; i < 16u; ++i) sm += part[i * 16u + c];
        out[ts * n + nj * 16u + c] = sm;
    }
}

// kernel_trellis_fwht_svh_id — y = H128(c) * svh[e] per (token, slot).
// Grid: (tokens * topk * (n/128)) threadgroups of 256 threads (128 used).
kernel void kernel_trellis_fwht_svh_id(
        constant ggml_metal_kargs_trellis_fwht_svh_id & args,
        device const float * c,
        device const float * svh,
        device const int * ids,
        device float * out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tptg  [[thread_position_in_threadgroup]]) {
    const uint tid   = tptg.x;
    const uint n     = (uint) args.n;
    const uint topk  = (uint) args.topk;
    const uint nb    = n / 128u;
    const uint blk   = tgpig.x % nb;
    const uint ts    = tgpig.x / nb; // token*topk + slot
    const uint slot  = ts % topk;
    const uint token = ts / topk;
    const uint e     = (uint) ids[slot + token * topk];
    threadgroup float sh[128];

    if (tid < 128u) sh[tid] = c[ts * n + blk * 128u + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint len = 64u; len >= 1u; len >>= 1) {
        if (tid < 64u) {
            const uint i = (tid / len) * (2u * len) + (tid % len);
            const float u = sh[i];
            const float v = sh[i + len];
            sh[i]       = u + v;
            sh[i + len] = u - v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 128u) sh[tid] *= (1.0f / sqrt(128.0f));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128u) out[ts * n + blk * 128u + tid] = sh[tid] * svh[e * n + blk * 128u + tid];
}


// kernel_trellis_gemm_nb + kernel_trellis_decode_id — Prefill fusion
// (DISENO §12, 19-nov-2026).
//
// The 2d bench showed the dense prefill TTFT is nearly flat with prompt
// length (3.36/3.40/3.60 s at 39/34/393 tokens): the fixed cost of the
// S>1 path (TRELLIS_DECODE of the whole matrix to scratch + transpose
// + cont + stock mul_mat, per linear, per prefill) dominates. The
// expert prefill re-decodes each expert ~12x (once per token that
// selected it).
//
// kernel_trellis_gemm_nb: multi-row GEMV_NB. c[n, S] = xh[k, S] @
// W_inner without materializing W. Grid (tn, ceil(S/16)) threadgroups
// of 256 threads; a threadgroup owns 16 output columns x 16 rows.
// Each thread decodes its (r, c) element once per tile (2 word loads,
// funnel shift, codebook — the GEMV_NB decode) and accumulates 16 rows
// in registers. Decode cost is amortized over 16 rows; the FMA part is
// scalar. Tail rows (beyond S) are clamped on load and discarded on
// store, keeping the inner loop uniform.
//
// kernel_trellis_decode_id: decode the stacked expert trellis to fp16
// scratch [k, n, n_exp] in ggml layout (element (r, c) of expert e at
// e*k*n + c*k + r), one thread per element with the 2-load decode
// (the 2b DECODE kernel loads 8*K words per thread; this one loads 2).
// Feeds the stock mul_mat_id for grouped expert prefill.
//
// Layouts (ggml column-major):
//   xh:  F32 [k, S]      (i, s) at s*k + i
//   out: F32 [n, S]      (j, s) at s*n + j
//   decode_id out: F16 [k, n, n_exp], (r, c, e) at e*k*n + c*k + r
// k and n are multiples of 16.

// kernel_trellis_gemm_nb — c = xh @ W_inner for S rows.
// Grid: (tn, ceil(S/16)) threadgroups of 256 threads.
kernel void kernel_trellis_gemm_nb(
        constant ggml_metal_kargs_trellis_gemm_nb & args,
        device const uint16_t * trellis,
        device const float * xh,
        device float * out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tptg  [[thread_position_in_threadgroup]]) {
    const uint tid = tptg.x;
    const uint nj  = tgpig.x; // output tile: columns [nj*16, nj*16+16)
    const uint rb  = tgpig.y; // row block: rows [rb*16, rb*16+16)
    const uint tk = (uint) args.tk; // k/16
    const uint tn = (uint) args.tn; // n/16
    const uint K  = (uint) args.K;
    const uint cb = (uint) args.cb;
    const uint S  = (uint) args.nrows; // total rows
    const uint k  = tk * 16u;
    const uint n  = tn * 16u;

    device const uint * words32 = (device const uint *) trellis;
    threadgroup float part[256]; // per-thread partials for the final sum

    const uint r = tid >> 4;
    const uint c = tid & 15u;
    // inverse of the tensor-core permutation: (r, c) -> stream position t
    // (identical to kernel_trellis_gemv_nb / MLX opt4)
    const uint c0  = c & 7u;
    const uint hic = c >> 3;
    const uint r0  = (r & 7u) & 6u;
    const uint rr  = r - r0;
    const uint j   = hic * 4u + (rr < 8u ? rr : rr - 6u);
    const uint t   = (c0 * 4u + (r0 >> 1)) * 8u + j;
    const uint nbits = 256u * K;
    const uint b0 = (t * K + K - 16u + nbits) % nbits;
    const uint nw = 8u * K;
    const uint q  = b0 >> 5;
    const uint s  = b0 & 31u;

    // row offsets into xh for this thread's r (tail rows clamped to
    // S-1; their results are discarded on store)
    uint roff[16];
    for (uint i = 0; i < 16u; ++i) {
        const uint row = rb * 16u + i;
        roff[i] = (row < S ? row : S - 1u) * k + r;
    }

    // Decode each (r, c) element once per tile, FMA against 16 rows.
    // No barriers in the loop (the 2e lesson: serialized latency rules).
    float acc[16] = {};
    for (uint ti = 0; ti < tk; ++ti) {
        const uint wbase = (ti * tn + nj) * 8u * K;
        const uint A = words32[wbase + q];
        const uint B = words32[wbase + (q + 1u == nw ? 0u : q + 1u)];
        const uint X = (A << s) | (s == 0u ? 0u : (B >> (32u - s)));
        const uint state = X >> 16;
        const float v = trellis_value(state, cb);
        const uint base = ti * 16u;
        for (uint i = 0; i < 16u; ++i) {
            acc[i] += v * xh[roff[i] + base];
        }
    }

    // Final reduction over the 16 rows, one output row at a time.
    for (uint i = 0; i < 16u; ++i) {
        part[tid] = acc[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint row = rb * 16u + i;
        if (r == 0u && row < S) {
            float sm = 0.0f;
            for (uint ii = 0; ii < 16u; ++ii) sm += part[ii * 16u + c];
            out[row * n + nj * 16u + c] = sm;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// kernel_trellis_decode_id — decode stacked expert trellis to fp16
// [k, n, n_exp] (ggml layout, ready for mul_mat_id). One thread per
// element, 2-load decode. Grid: (tk*tn*n_exp) threadgroups of 256.
kernel void kernel_trellis_decode_id(
        constant ggml_metal_kargs_trellis_decode_id & args,
        device const uint16_t * trellis,
        device half * out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tptg  [[thread_position_in_threadgroup]]) {
    const uint tid = tptg.x;
    const uint tk = (uint) args.tk; // k/16
    const uint tn = (uint) args.tn; // n/16
    const uint K  = (uint) args.K;
    const uint cb = (uint) args.cb;
    const uint k  = tk * 16u;
    const uint n  = tn * 16u;

    const uint tiles_per_exp = tk * tn;
    const uint e    = tgpig.x / tiles_per_exp;
    const uint tile = tgpig.x % tiles_per_exp;
    const uint ti   = tile / tn;
    const uint nj   = tile % tn;

    device const uint * words32 = (device const uint *) trellis + e * tiles_per_exp * 8u * K;

    const uint r = tid >> 4;
    const uint c = tid & 15u;
    // inverse of the tensor-core permutation: (r, c) -> stream position t
    const uint c0  = c & 7u;
    const uint hic = c >> 3;
    const uint r0  = (r & 7u) & 6u;
    const uint rr  = r - r0;
    const uint j   = hic * 4u + (rr < 8u ? rr : rr - 6u);
    const uint t   = (c0 * 4u + (r0 >> 1)) * 8u + j;
    const uint nbits = 256u * K;
    const uint b0 = (t * K + K - 16u + nbits) % nbits;
    const uint nw = 8u * K;
    const uint q  = b0 >> 5;
    const uint s  = b0 & 31u;

    const uint wbase = (ti * tn + nj) * 8u * K;
    const uint A = words32[wbase + q];
    const uint B = words32[wbase + (q + 1u == nw ? 0u : q + 1u)];
    const uint X = (A << s) | (s == 0u ? 0u : (B >> (32u - s)));
    const uint state = X >> 16;
    const half v = (half) trellis_value(state, cb);

    // ggml [k, n] layout: element (r, c) of W_inner at c*k + r
    out[e * k * n + (nj * 16u + c) * k + (ti * 16u + r)] = v;
}
