// q36 pack: the model file laid out for this engine.
//  * every quantized matrix is stored "row-SoA": one contiguous record per row, planes inside the row
//      Q8_0 : [qs n][d half n/32]
//      Q4_K : [qs n/2][scales 12/blk][d,dmin half pairs 4/blk]
//      Q5_K : [qs n/2][qh n/8][scales 12/blk][d,dmin 4/blk]
//      Q6_K : [ql n/2][qh n/4][scales 16/blk][d half 2/blk]
//    rows padded to 16 bytes. The quantized values are bit-identical to the GGUF (lossless re-layout).
//  * one expert = one contiguous record: 512 x [gate row | up row], then 2048 x [down row]
//  * same-input dense matrices are fused (qkv+gate, q+k+v), small float tensors live in one params blob per layer
#pragma once
#include "cpu.hpp"
#include <sstream>

namespace q36 {

static inline uint32_t soa_row_bytes(uint32_t type, int n) {
    uint32_t b = 0;
    switch (type) {
        case T_Q8_0: b = n + n / 32 * 2; break;
        case T_Q4_K: b = n / 256 * 144; break;
        case T_Q5_K: b = n / 256 * 176; break;
        case T_Q6_K: b = n / 256 * 210; break;
        case T_F32:  b = n * 4; break;
        default: throw std::runtime_error("soa: unsupported type");
    }
    return (b + 15) / 16 * 16;
}

// convert one GGUF row (AoS blocks) to the row-SoA record
static inline void soa_convert_row(uint32_t type, int n, const uint8_t * a, uint8_t * o) {
    const int nb = n / 256;
    switch (type) {
        case T_Q8_0:
            for (int b = 0; b < n / 32; b++) { memcpy(o + 32 * b, a + 34 * b + 2, 32); memcpy(o + n + 2 * b, a + 34 * b, 2); }
            break;
        case T_Q4_K:
            for (int b = 0; b < nb; b++) { const uint8_t * s = a + 144 * b; memcpy(o + 128 * b, s + 16, 128); memcpy(o + nb * 128 + 12 * b, s + 4, 12); memcpy(o + nb * 140 + 4 * b, s, 4); }
            break;
        case T_Q5_K:
            for (int b = 0; b < nb; b++) { const uint8_t * s = a + 176 * b; memcpy(o + 128 * b, s + 48, 128); memcpy(o + nb * 128 + 32 * b, s + 16, 32); memcpy(o + nb * 160 + 12 * b, s + 4, 12); memcpy(o + nb * 172 + 4 * b, s, 4); }
            break;
        case T_Q6_K:
            for (int b = 0; b < nb; b++) { const uint8_t * s = a + 210 * b; memcpy(o + 128 * b, s, 128); memcpy(o + nb * 128 + 64 * b, s + 128, 64); memcpy(o + nb * 192 + 16 * b, s + 192, 16); memcpy(o + nb * 208 + 2 * b, s + 208, 2); }
            break;
        case T_F32: memcpy(o, a, (size_t) n * 4); break;
        case T_BF16: for (int i = 0; i < n; i++) { uint16_t h; memcpy(&h, a + 2 * i, 2); float f = bf2f(h); memcpy(o + 4 * i, &f, 4); } break;
        default: throw std::runtime_error("soa: unsupported type");
    }
}

// ---------------------------------------------------------------- CPU kernels on row-SoA records (exact float math)
static inline void k4_scales_soa(const uint8_t * q, const uint8_t * dm, float * sc, float * mn) {
    uint16_t dh, mh; memcpy(&dh, dm, 2); memcpy(&mh, dm + 2, 2); k4_scales(q, h2f(dh), h2f(mh), sc, mn);
}
static inline float srow_q8(const uint8_t * w, int n, const float * x) {
    __m256 acc = _mm256_setzero_ps(); const uint8_t * dp = w + n;
    for (int b = 0; b < n / 32; b++, w += 32, x += 32) {
        uint16_t dh; memcpy(&dh, dp + 2 * b, 2);
        __m128i q0 = _mm_loadu_si128((const __m128i *) w), q1 = _mm_loadu_si128((const __m128i *) (w + 16));
        __m256 s = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0)), _mm256_loadu_ps(x));
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8))), _mm256_loadu_ps(x + 8), s);
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1)), _mm256_loadu_ps(x + 16), s);
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8))), _mm256_loadu_ps(x + 24), s);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(h2f(dh)), s, acc);
    }
    return hsum(acc);
}
static inline float srow_q4k(const uint8_t * w, int n, const float * x, const float * xs) {
    const __m256i m4 = _mm256_set1_epi8(0xF); const int nb = n / 256;
    __m256 acc = _mm256_setzero_ps(); float msum = 0;
    for (int b = 0; b < nb; b++, x += 256, xs += 16) {
        float sc[8], mn[8]; k4_scales_soa(w + nb * 128 + 12 * b, w + nb * 140 + 4 * b, sc, mn);
        const uint8_t * qs = w + 128 * b;
        for (int g = 0; g < 4; g++) {
            __m256i v = _mm256_loadu_si256((const __m256i *) (qs + 32 * g));
            __m256i lo = _mm256_and_si256(v, m4), hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sc[2 * g]),     dot32_u8(lo, x + 64 * g),      acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sc[2 * g + 1]), dot32_u8(hi, x + 64 * g + 32), acc);
            msum += mn[2 * g] * (xs[4 * g] + xs[4 * g + 1]) + mn[2 * g + 1] * (xs[4 * g + 2] + xs[4 * g + 3]);
        }
    }
    return hsum(acc) - msum;
}
static inline float srow_q5k(const uint8_t * w, int n, const float * x, const float * xs) {
    const __m256i m4 = _mm256_set1_epi8(0xF), one = _mm256_set1_epi8(1); const int nb = n / 256;
    __m256 acc = _mm256_setzero_ps(); float msum = 0;
    for (int b = 0; b < nb; b++, x += 256, xs += 16) {
        float sc[8], mn[8]; k4_scales_soa(w + nb * 160 + 12 * b, w + nb * 172 + 4 * b, sc, mn);
        const __m256i qh = _mm256_loadu_si256((const __m256i *) (w + nb * 128 + 32 * b));
        const uint8_t * qs = w + 128 * b;
        for (int g = 0; g < 4; g++) {
            __m256i v = _mm256_loadu_si256((const __m256i *) (qs + 32 * g));
            __m256i h1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * g), one), 4);
            __m256i h2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * g + 1), one), 4);
            __m256i lo = _mm256_or_si256(_mm256_and_si256(v, m4), h1);
            __m256i hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v, 4), m4), h2);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sc[2 * g]),     dot32_u8(lo, x + 64 * g),      acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sc[2 * g + 1]), dot32_u8(hi, x + 64 * g + 32), acc);
            msum += mn[2 * g] * (xs[4 * g] + xs[4 * g + 1]) + mn[2 * g + 1] * (xs[4 * g + 2] + xs[4 * g + 3]);
        }
    }
    return hsum(acc) - msum;
}
static inline float srow_q6k(const uint8_t * w, int n, const float * x, const float * xs) {
    const __m256i m4 = _mm256_set1_epi8(0xF), m2 = _mm256_set1_epi8(3); const int nb = n / 256;
    __m256 acc = _mm256_setzero_ps(); float msum = 0;
    for (int b = 0; b < nb; b++, x += 256, xs += 16) {
        const uint8_t * ql = w + 128 * b, * qhp = w + nb * 128 + 64 * b; const int8_t * scl = (const int8_t *) (w + nb * 192 + 16 * b);
        uint16_t dh; memcpy(&dh, w + nb * 208 + 2 * b, 2); const float d = h2f(dh);
        for (int hf = 0; hf < 2; hf++) {
            __m256i A = _mm256_loadu_si256((const __m256i *) (ql + 64 * hf)), B = _mm256_loadu_si256((const __m256i *) (ql + 64 * hf + 32));
            __m256i H = _mm256_loadu_si256((const __m256i *) (qhp + 32 * hf));
            __m256i q[4];
            q[0] = _mm256_or_si256(_mm256_and_si256(A, m4), _mm256_slli_epi16(_mm256_and_si256(H, m2), 4));
            q[1] = _mm256_or_si256(_mm256_and_si256(B, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 2), m2), 4));
            q[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(A, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 4), m2), 4));
            q[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(B, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 6), m2), 4));
            const float * xx = x + 128 * hf; const float * xss = xs + 8 * hf; const int8_t * s8 = scl + 8 * hf;
            for (int j = 0; j < 4; j++) {
                const float sa = d * s8[2 * j], sb = d * s8[2 * j + 1];
                acc = _mm256_fmadd_ps(_mm256_set1_ps(sa), dot16_u8(_mm256_castsi256_si128(q[j]), xx + 32 * j, _mm256_setzero_ps()), acc);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(sb), dot16_u8(_mm256_extracti128_si256(q[j], 1), xx + 32 * j + 16, _mm256_setzero_ps()), acc);
                msum += 32.0f * (sa * xss[2 * j] + sb * xss[2 * j + 1]);
            }
        }
    }
    return hsum(acc) - msum;
}
static inline float srow(uint32_t type, const uint8_t * w, int n, const float * x, const float * xs) {
    switch (type) {
        case T_Q8_0: return srow_q8(w, n, x);
        case T_Q4_K: return srow_q4k(w, n, x, xs);
        case T_Q5_K: return srow_q5k(w, n, x, xs);
        case T_Q6_K: return srow_q6k(w, n, x, xs);
        case T_F32:  return dot_f32((const float *) w, x, n);
    }
    return 0;
}

// ---------------------------------------------------------------- CPU kernels with int8 activations (same quantization as the GPU kernels)
// act: q[n] int8, d[n/32] scale per 32, s[n/16] dequantized sums per 16
static inline void quant_act(const float * x, int n, int8_t * q, float * d, float * s) {
    const __m256 sign = _mm256_set1_ps(-0.0f); const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int b = 0; b < n / 32; b++, x += 32) {
        __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8), v2 = _mm256_loadu_ps(x + 16), v3 = _mm256_loadu_ps(x + 24);
        __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)), _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1)); m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4)); m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
        const float amax = _mm_cvtss_f32(m4), dd = amax / 127.0f, id = dd > 0.0f ? 1.0f / dd : 0.0f; const __m256 vid = _mm256_set1_ps(id);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vid)), i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vid)), i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vid)), i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vid));
        __m256i sa = _mm256_add_epi32(i0, i1), sb = _mm256_add_epi32(i2, i3);
        __m128i ha = _mm_add_epi32(_mm256_castsi256_si128(sa), _mm256_extracti128_si256(sa, 1)), hb = _mm_add_epi32(_mm256_castsi256_si128(sb), _mm256_extracti128_si256(sb, 1));
        ha = _mm_hadd_epi32(ha, hb); ha = _mm_hadd_epi32(ha, ha);
        d[b] = dd; s[2 * b] = dd * (float) _mm_extract_epi32(ha, 0); s[2 * b + 1] = dd * (float) _mm_extract_epi32(ha, 1);
        i0 = _mm256_packs_epi32(i0, i1); i2 = _mm256_packs_epi32(i2, i3); i0 = _mm256_packs_epi16(i0, i2);
        _mm256_storeu_si256((__m256i *) (q + 32 * b), _mm256_permutevar8x32_epi32(i0, perm));
    }
}
static inline __m256 dot32_i8(__m256i w_u8, const int8_t * x) {
    __m256i p = _mm256_maddubs_epi16(w_u8, _mm256_loadu_si256((const __m256i *) x));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}
// vectorized get_scale_min_k4 for one block: 12 packed bytes -> 8 scales * d, 8 mins * dmin
static inline void k4_decode(const uint8_t * q12, const uint8_t * dm, __m256 & scf, __m256 & mnf) {
    uint16_t dh, mh; memcpy(&dh, dm, 2); memcpy(&mh, dm + 2, 2);
    const __m128i q = _mm_loadu_si128((const __m128i *) q12);
    const __m128i lo6 = _mm_and_si128(q, _mm_set1_epi8(0x3F)), top2 = _mm_and_si128(_mm_srli_epi16(q, 6), _mm_set1_epi8(0x03));
    const __m128i q8 = _mm_srli_si128(q, 8), m4 = _mm_set1_epi8(0x0F);
    const __m128i sc_hi = _mm_or_si128(_mm_and_si128(q8, m4), _mm_slli_epi16(top2, 4));
    const __m128i mn_hi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q8, 4), m4), _mm_slli_epi16(_mm_srli_si128(top2, 4), 4));
    const __m128i sc8 = _mm_unpacklo_epi32(lo6, sc_hi), mn8 = _mm_unpacklo_epi32(_mm_srli_si128(lo6, 4), mn_hi);
    scf = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(sc8)), _mm256_set1_ps(h2f(dh)));
    mnf = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(mn8)), _mm256_set1_ps(h2f(mh)));
}
// xs32 = sums of the dequantized activation over 32
static inline float qrow_q4k(const uint8_t * w, int n, const int8_t * xq, const float * xd, const float * xs32) {
    const __m256i m4 = _mm256_set1_epi8(0xF); const int nb = n / 256;
    __m256 acc = _mm256_setzero_ps(), macc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++, xq += 256, xd += 8, xs32 += 8) {
        __m256 scf, mnf; k4_decode(w + nb * 128 + 12 * b, w + nb * 140 + 4 * b, scf, mnf);
        float sd[8]; _mm256_storeu_ps(sd, _mm256_mul_ps(scf, _mm256_loadu_ps(xd)));
        macc = _mm256_fmadd_ps(mnf, _mm256_loadu_ps(xs32), macc);
        const uint8_t * qs = w + 128 * b;
        for (int g = 0; g < 4; g++) {
            __m256i v = _mm256_loadu_si256((const __m256i *) (qs + 32 * g));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sd[2 * g]),     dot32_i8(_mm256_and_si256(v, m4), xq + 64 * g), acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sd[2 * g + 1]), dot32_i8(_mm256_and_si256(_mm256_srli_epi16(v, 4), m4), xq + 64 * g + 32), acc);
        }
    }
    return hsum(_mm256_sub_ps(acc, macc));
}
static inline float qrow_q5k(const uint8_t * w, int n, const int8_t * xq, const float * xd, const float * xs32) {
    const __m256i m4 = _mm256_set1_epi8(0xF), one = _mm256_set1_epi8(1); const int nb = n / 256;
    __m256 acc = _mm256_setzero_ps(), macc = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++, xq += 256, xd += 8, xs32 += 8) {
        __m256 scf, mnf; k4_decode(w + nb * 160 + 12 * b, w + nb * 172 + 4 * b, scf, mnf);
        float sd[8]; _mm256_storeu_ps(sd, _mm256_mul_ps(scf, _mm256_loadu_ps(xd)));
        macc = _mm256_fmadd_ps(mnf, _mm256_loadu_ps(xs32), macc);
        const __m256i qh = _mm256_loadu_si256((const __m256i *) (w + nb * 128 + 32 * b));
        const uint8_t * qs = w + 128 * b;
        for (int g = 0; g < 4; g++) {
            __m256i v = _mm256_loadu_si256((const __m256i *) (qs + 32 * g));
            __m256i lo = _mm256_or_si256(_mm256_and_si256(v, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * g), one), 4));
            __m256i hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(v, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * g + 1), one), 4));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sd[2 * g]),     dot32_i8(lo, xq + 64 * g), acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(sd[2 * g + 1]), dot32_i8(hi, xq + 64 * g + 32), acc);
        }
    }
    return hsum(_mm256_sub_ps(acc, macc));
}
static inline float qrow_q6k(const uint8_t * w, int n, const int8_t * xq, const float * xd, const float * xs) {
    const __m256i m4 = _mm256_set1_epi8(0xF), m2 = _mm256_set1_epi8(3); const int nb = n / 256;
    __m256 acc = _mm256_setzero_ps(); float msum = 0;
    for (int b = 0; b < nb; b++, xq += 256, xd += 8, xs += 16) {
        const uint8_t * ql = w + 128 * b, * qhp = w + nb * 128 + 64 * b; const int8_t * scl = (const int8_t *) (w + nb * 192 + 16 * b);
        uint16_t dh; memcpy(&dh, w + nb * 208 + 2 * b, 2); const float d = h2f(dh);
        for (int hf = 0; hf < 2; hf++) {
            __m256i A = _mm256_loadu_si256((const __m256i *) (ql + 64 * hf)), B = _mm256_loadu_si256((const __m256i *) (ql + 64 * hf + 32));
            __m256i H = _mm256_loadu_si256((const __m256i *) (qhp + 32 * hf));
            __m256i q[4];
            q[0] = _mm256_or_si256(_mm256_and_si256(A, m4), _mm256_slli_epi16(_mm256_and_si256(H, m2), 4));
            q[1] = _mm256_or_si256(_mm256_and_si256(B, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 2), m2), 4));
            q[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(A, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 4), m2), 4));
            q[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(B, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 6), m2), 4));
            const int8_t * xx = xq + 128 * hf; const float * xss = xs + 8 * hf; const int8_t * s8 = scl + 8 * hf; const float * xdd = xd + 4 * hf;
            for (int j = 0; j < 4; j++) { // 32 weights: two sub-blocks of 16 with their own scale, one activation scale
                const float sa = d * s8[2 * j], sb = d * s8[2 * j + 1];
                __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(q[j], _mm256_loadu_si256((const __m256i *) (xx + 32 * j))), _mm256_set1_epi16(1));
                const __m256 scv = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_set1_ps(sa * xdd[j])), _mm_set1_ps(sb * xdd[j]), 1);
                acc = _mm256_fmadd_ps(scv, _mm256_cvtepi32_ps(p), acc);
                msum += 32.0f * (sa * xss[2 * j] + sb * xss[2 * j + 1]);
            }
        }
    }
    return hsum(acc) - msum;
}
static inline float qrow(uint32_t type, const uint8_t * w, int n, const int8_t * xq, const float * xd, const float * xs, const float * xs32) {
    switch (type) {
        case T_Q4_K: return qrow_q4k(w, n, xq, xd, xs32);
        case T_Q5_K: return qrow_q5k(w, n, xq, xd, xs32);
        case T_Q6_K: return qrow_q6k(w, n, xq, xd, xs);
    }
    return 0;
}

// ---------------------------------------------------------------- pack reader
struct PEntry { std::string name; uint32_t type = 0; uint64_t rows = 0; uint32_t n_in = 0, row_bytes = 0; uint64_t off = 0, size = 0; const uint8_t * data = nullptr; };

struct Pack {
    MMap mm; std::map<std::string, PEntry> ent; std::map<std::string, double> hp;
    void open(const std::string & path) {
        mm.open(path);
        if (mm.n < 4096 || memcmp(mm.p, "Q36PACK1", 8) != 0) throw std::runtime_error("not a q36 pack: " + path);
        uint64_t toc_off, toc_size; memcpy(&toc_off, mm.p + 8, 8); memcpy(&toc_size, mm.p + 16, 8);
        std::istringstream is(std::string((const char *) mm.p + toc_off, toc_size)); std::string kind;
        while (is >> kind) {
            if (kind == "hp") { std::string k; double v; is >> k >> v; hp[k] = v; }
            else if (kind == "t") { PEntry e; is >> e.name >> e.type >> e.rows >> e.n_in >> e.row_bytes >> e.off >> e.size; e.data = mm.p + e.off; ent[e.name] = e; }
        }
    }
    const PEntry & get(const std::string & n) const { auto it = ent.find(n); if (it == ent.end()) throw std::runtime_error("pack: missing " + n); return it->second; }
    bool has(const std::string & n) const { return ent.count(n) > 0; }
    int hpi(const std::string & k) const { auto it = hp.find(k); if (it == hp.end()) throw std::runtime_error("pack: missing hp " + k); return (int) it->second; }
};

// offsets (in floats) inside the per-layer params blob
struct POff {
    static constexpr int attn_norm = 0, post_norm = 2048;
    static constexpr int conv = 4096;                    // GDN: 4 taps x 8192 channels, tap-major (tap 3 = current input)
    static constexpr int dt_bias = conv + 4 * 8192, ssm_a = dt_bias + 32, ssm_norm = ssm_a + 32, gdn_end = ssm_norm + 128;
    static constexpr int q_norm = 4096, k_norm = q_norm + 256, att_end = k_norm + 256; // attention
    static constexpr int enorm = att_end, hnorm = enorm + 2048, head_norm = hnorm + 2048, mtp_end = head_norm + 2048; // MTP block
};

} // namespace q36
