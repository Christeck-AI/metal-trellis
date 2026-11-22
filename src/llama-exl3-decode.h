// llama-exl3-decode.h — CPU reference decode of EXL3 (exllamav3 v1.x) linears,
// ported from trellis_core/reference.py. Used by the 1b loader (decode at load
// + re-quantize) and by the Paso 2 kernel judges. This is the slow, obviously
// correct path: bitstream -> states -> codebook LUT -> tile perm -> FWHT.
#pragma once

#include <cstdint>
#include <string>

// Codebook ids (same numbering as reference.py).
enum exl3_codebook {
    EXL3_CB_3INST = 0,
    EXL3_CB_MCG   = 1,
    EXL3_CB_MUL1  = 2,
};

// Decode one linear to its original-basis fp32 weight, row-major [k, n].
//   trellis: int16 [k/16, n/16, 16*K]  (one 16x16 tile per (i, j))
//   suh: fp16 bits [k], svh: fp16 bits [n]
//   K: trellis depth per element (3 for the 35B, 6 for its lm_head)
//   cb: EXL3_CB_*
// Returns false + err on bad input (shapes not multiples of 16/128, bad K/cb).
bool exl3_weight_orig(const int16_t * trellis, const uint16_t * suh, const uint16_t * svh,
                      int64_t k, int64_t n, int K, int cb,
                      float * out, std::string & err);

// The 256-entry tensor-core permutation (reference.py tensor_core_perm()).
// Element t of a tile lands at position perm[t] in the 16x16 tile.
const int * exl3_tensor_core_perm();
