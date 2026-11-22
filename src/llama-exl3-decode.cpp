// llama-exl3-decode.cpp — CPU reference decode of EXL3 linears.
// Ported from trellis_core/reference.py (bit-exact); the FWHT replaces the
// reference's dense Hadamard matmuls (same transform, different summation
// order — compare against the reference with a fp32 tolerance, ~1e-4 rel).
#include "llama-exl3-decode.h"

#include <cmath>
#include <cstring>
#include <vector>

// ---------- fp16 helpers (exact conversions, round-to-nearest-even) ----------

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1;
    const uint32_t exp  = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else { // subnormal: normalize
            uint32_t m = mant;
            int e = -14;
            while ((m & 0x400) == 0) { m <<= 1; e--; }
            m &= 0x3FF;
            f = (sign << 31) | ((uint32_t) (e + 127) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

static uint16_t f32_to_f16(float x) {
    uint32_t f;
    std::memcpy(&f, &x, 4);
    const uint32_t sign = (f >> 16) & 0x8000;
    const uint32_t exp  = (f >> 23) & 0xFF;
    const uint32_t mant = f & 0x7FFFFF;
    if (exp == 255) return (uint16_t) (sign | 0x7C00 | (mant ? 0x200 : 0)); // nan/inf
    if (exp > 142) return (uint16_t) (sign | 0x7C00);                       // overflow -> inf
    if (exp < 113) {                                                         // subnormal / zero
        if (exp < 102) return (uint16_t) sign;                              // underflows to zero
        const uint32_t m = mant | 0x800000;
        const int shift = 113 - (int) exp;
        uint32_t v = m >> shift;
        // round to nearest even on the dropped bits
        const uint32_t dropped = m & ((1u << shift) - 1);
        const uint32_t half = 1u << (shift - 1);
        if (dropped > half || (dropped == half && (v & 1))) v++;
        return (uint16_t) (sign | v);
    }
    // normal: round the 13 dropped mantissa bits to nearest even
    uint32_t v = mant >> 13;
    const uint32_t dropped = mant & 0x1FFF;
    if (dropped > 0x1000 || (dropped == 0x1000 && (v & 1))) {
        v++;
        if (v == 0x400) return (uint16_t) (sign | ((exp - 111) << 10)); // carried into exponent
    }
    return (uint16_t) (sign | ((exp - 112) << 10) | v);
}

// ---------- codebook LUT: fp16 value for every 16-bit state ----------

static std::vector<float> codebook_lut(int cb) {
    std::vector<float> lut(65536);
    for (uint32_t s = 0; s < 65536; s++) {
        if (cb == EXL3_CB_MUL1) {
            const uint32_t x = (uint32_t) (s * 0x83DCD12Du);
            const uint32_t bsum = (x & 0xFF) + ((x >> 8) & 0xFF)
                                + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF);
            const double h = 1024.0 + (double) bsum;
            const double inv  = (double) f16_to_f32(0x1EEE);
            const double bias = (double) f16_to_f32(0xC931);
            lut[s] = f16_to_f32(f32_to_f16((float) (h * inv + bias)));
        } else {
            uint32_t x;
            if (cb == EXL3_CB_3INST) x = (uint32_t) (s * 89226354u + 64248484u);
            else                     x = (uint32_t) (s * 0xCBAC1FEDu); // MCG
            x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
            const float lo = f16_to_f32((uint16_t) (x & 0xFFFF));
            const float hi = f16_to_f32((uint16_t) (x >> 16));
            lut[s] = f16_to_f32(f32_to_f16(lo + hi)); // __hadd: single rounding
        }
    }
    return lut;
}

// ---------- tensor-core permutation ----------

const int * exl3_tensor_core_perm() {
    static int perm[256];
    static bool init = false;
    if (!init) {
        for (int t = 0; t < 32; t++) {
            const int r0 = (t % 4) * 2;
            const int r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
            const int c0 = t / 4, c1 = c0 + 8;
            perm[t * 8 + 0] = r0 * 16 + c0;
            perm[t * 8 + 1] = r1 * 16 + c0;
            perm[t * 8 + 2] = r2 * 16 + c0;
            perm[t * 8 + 3] = r3 * 16 + c0;
            perm[t * 8 + 4] = r0 * 16 + c1;
            perm[t * 8 + 5] = r1 * 16 + c1;
            perm[t * 8 + 6] = r2 * 16 + c1;
            perm[t * 8 + 7] = r3 * 16 + c1;
        }
        init = true;
    }
    return perm;
}

// ---------- FWHT on 128-blocks (normalized Sylvester Hadamard) ----------

static void fwht128(float * x, int64_t stride) {
    for (int64_t len = 1; len < 128; len <<= 1) {
        for (int64_t i = 0; i < 128; i += 2 * len) {
            for (int64_t j = 0; j < len; j++) {
                const float u = x[(i + j) * stride];
                const float v = x[(i + j + len) * stride];
                x[(i + j) * stride]       = u + v;
                x[(i + j + len) * stride] = u - v;
            }
        }
    }
    const float scale = 1.0f / std::sqrt(128.0f);
    for (int64_t i = 0; i < 128; i++) x[i * stride] *= scale;
}

// ---------- the decode ----------

bool exl3_weight_orig(const int16_t * trellis, const uint16_t * suh, const uint16_t * svh,
                      int64_t k, int64_t n, int K, int cb,
                      float * out, std::string & err) {
    if (k % 128 != 0 || n % 128 != 0) { err = "k/n must be multiples of 128"; return false; }
    if (K < 2 || K > 6) { err = "K out of range"; return false; }
    if (cb < 0 || cb > 2) { err = "bad codebook"; return false; }
    if (!trellis || !suh || !svh || !out) { err = "null pointer"; return false; }

    const std::vector<float> lut = codebook_lut(cb);
    const int * perm = exl3_tensor_core_perm();
    const int64_t tk = k / 16, tn = n / 16;
    const int nbits = 256 * K;
    const int nwords = 8 * K; // uint32 words per tile

    // reconstruct_inner: trellis [tk, tn, 16K] -> W_inner fp32 [k, n]
    for (int64_t i = 0; i < tk; i++) {
        for (int64_t j = 0; j < tn; j++) {
            const int16_t * tile = trellis + (i * tn + j) * (16 * K);
            uint32_t words[48]; // max 8*K for K<=6
            for (int w = 0; w < nwords; w++) {
                const uint16_t lo = (uint16_t) tile[2 * w];
                const uint16_t hi = (uint16_t) tile[2 * w + 1];
                words[w] = (uint32_t) lo | ((uint32_t) hi << 16);
            }
            for (int t = 0; t < 256; t++) {
                const int b0 = (t * K + K - 16 + nbits) % nbits;
                const int wi = b0 / 32, off = b0 % 32;
                uint32_t state;
                if (off <= 16) {
                    state = (words[wi] >> (32 - off - 16)) & 0xFFFFu;
                } else {
                    const uint32_t w0 = words[wi];
                    const uint32_t w1 = words[(wi + 1) % nwords];
                    state = ((w0 & ((1u << (32 - off)) - 1u)) << (off - 16))
                          | (w1 >> (32 - (off - 16)));
                }
                const float v = lut[state];
                const int p = perm[t];
                out[(i * 16 + p / 16) * n + (j * 16 + p % 16)] = v;
            }
        }
    }

    // H on rows (128-blocks down each column), then x suh
    for (int64_t rb = 0; rb < k / 128; rb++) {
        for (int64_t c = 0; c < n; c++) {
            fwht128(out + rb * 128 * n + c, n);
        }
    }
    for (int64_t r = 0; r < k; r++) {
        const float s = f16_to_f32(suh[r]);
        for (int64_t c = 0; c < n; c++) out[r * n + c] *= s;
    }
    // H on cols (128-blocks across each row), then x svh
    for (int64_t r = 0; r < k; r++) {
        for (int64_t cbk = 0; cbk < n / 128; cbk++) {
            fwht128(out + r * n + cbk * 128, 1);
        }
    }
    for (int64_t c = 0; c < n; c++) {
        const float s = f16_to_f32(svh[c]);
        for (int64_t r = 0; r < k; r++) out[r * n + c] *= s;
    }
    return true;
}
