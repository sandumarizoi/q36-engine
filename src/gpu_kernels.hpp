// q36 GPU kernels (SYCL ESIMD, Xe2 / B580). All pointers are USM. One ESIMD thread = one row / head / column.
// Activations that feed quantized mat-vecs are int8 with one scale per 32 values ("act": q[n], d[n/32], s[n/16] where
// s is the sum of the dequantized values over 16, needed by the K-quant min terms).
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdint>

namespace q36g {
using namespace sycl::ext::intel::esimd;
namespace es = sycl::ext::intel::esimd;

constexpr int NCMAX = 8;      // max tokens per step (decode = 1, MTP verify up to NCMAX)
constexpr int RING = 16;      // GDN per-position ring (conv inputs, q/k/v, update records)
constexpr int RMASK = RING - 1;
constexpr int ATT_CH_MAX = 1024; // attention chunk (keys per ESIMD thread, a multiple of 16), chosen from the context depth
constexpr int ATT_NCH_MAX = 1024;

struct Act { int8_t * q; float * d; float * s; };

// quantize 32 floats -> 32 int8 + scale + two 16-sums (dequantized)
static ESIMD_INLINE void quant32(simd<float, 16> a0, simd<float, 16> a1, int8_t * qp, float * dp, float * sp) {
    const float amax = hmax<float>((es::max)(es::abs(a0), es::abs(a1)));
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    simd<int, 16> i0 = rnde<int, 16>(a0 * id), i1 = rnde<int, 16>(a1 * id);
    simd<int8_t, 16> b0 = convert<int8_t>(i0), b1 = convert<int8_t>(i1);
    block_store<int8_t, 16>(qp, b0); block_store<int8_t, 16>(qp + 16, b1);
    dp[0] = d; sp[0] = d * (float) reduce<int>(i0, std::plus<>{}); sp[1] = d * (float) reduce<int>(i1, std::plus<>{});
}

static ESIMD_INLINE simd<float, 16> silu16(simd<float, 16> x) { return x / (es::exp(-x) + 1.0f); }
static ESIMD_INLINE simd<float, 16> sigmoid16(simd<float, 16> x) { return simd<float, 16>(1.0f) / (es::exp(-x) + 1.0f); }

// ---------------------------------------------------------------- residual update + RMS norm + quantize
// one work-group of 64 threads per column (thread = block of 32 values); the mean square goes through SLM.
// x = (init ? 0 : x) + add1 + add2 ; xn = x * rsqrt(mean(x^2) + eps) * w ; xn -> device float, optional host float, act
// win_x / win_a: X / add1 are window buffers, row = prm[5] + c. win_o >= 0: xn_dev is a window buffer, row = prm[5] + c + win_o.
static inline sycl::event k_norm_quant(sycl::queue & q, int nc, float * X, const float * add1, const float * add2, bool init,
                                       const float * w, float eps, float * xn_dev, float * xn_host, Act a,
                                       const int * prm = nullptr, bool win_x = false, bool win_a = false, int win_o = -1) {
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nc * 64), sycl::range<1>(64)), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
        slm_init<64 * 4>();
        const int c = (int) it.get_group(0), b = (int) it.get_local_id(0); const int i = 32 * b;
        const int woff = (win_x || win_o >= 0) ? prm[5] : 0; const size_t xr = (size_t) (win_x ? woff : 0) + c; float * x = X + xr * 2048 + i;
        simd<float, 16> v0 = 0.0f, v1 = 0.0f;
        if (!init) { v0 = block_load<float, 16>(x); v1 = block_load<float, 16>(x + 16); }
        if (add1) { const size_t ar = (size_t) (win_a ? woff : 0) + c; v0 += block_load<float, 16>(add1 + ar * 2048 + i); v1 += block_load<float, 16>(add1 + ar * 2048 + i + 16); }
        if (add2) { v0 += block_load<float, 16>(add2 + c * 2048 + i); v1 += block_load<float, 16>(add2 + c * 2048 + i + 16); }
        block_store<float, 16>(x, v0); block_store<float, 16>(x + 16, v1);
        slm_scalar_store<float>(b * 4, reduce<float>(v0 * v0 + v1 * v1, std::plus<>{}));
        barrier();
        simd<float, 16> t = slm_block_load<float, 16>(0) + slm_block_load<float, 16>(64) + slm_block_load<float, 16>(128) + slm_block_load<float, 16>(192);
        const float sc = 1.0f / es::sqrt_ieee<float>(reduce<float>(t, std::plus<>{}) / 2048.0f + eps);
        simd<float, 16> a0 = v0 * block_load<float, 16>(w + i) * sc, a1 = v1 * block_load<float, 16>(w + i + 16) * sc;
        const size_t orow = win_o >= 0 ? (size_t) woff + c + win_o : (size_t) c;
        block_store<float, 16>(xn_dev + orow * 2048 + i, a0); block_store<float, 16>(xn_dev + orow * 2048 + i + 16, a1);
        if (xn_host) { block_store<float, 16>(xn_host + c * 2048 + i, a0); block_store<float, 16>(xn_host + c * 2048 + i + 16, a1); }
        quant32(a0, a1, a.q + c * 2048 + i, a.d + c * 64 + b, a.s + c * 128 + 2 * b);
    });
}

// ---------------------------------------------------------------- MTP input: [rms(emb) * enorm ; rms(h) * hnorm] -> act(4096)
// work-group of 64 threads per (column, half); emb comes from host USM, h from the device
static inline sycl::event k_mtp_in(sycl::queue & q, int nc, const float * emb, const float * hin, const float * enorm, const float * hnorm, float eps, Act a, const int * prm = nullptr, bool win = false) {
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nc * 2 * 64), sycl::range<1>(64)), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
        slm_init<64 * 4>();
        const int c = (int) it.get_group(0) / 2, hf = (int) it.get_group(0) % 2, b = (int) it.get_local_id(0); const int i = 32 * b;
        const size_t row = (size_t) (win ? prm[5] : 0) + c; const float * src = (hf ? hin : emb) + row * 2048 + i; const float * w = (hf ? hnorm : enorm) + i;
        simd<float, 16> v0 = block_load<float, 16>(src), v1 = block_load<float, 16>(src + 16);
        slm_scalar_store<float>(b * 4, reduce<float>(v0 * v0 + v1 * v1, std::plus<>{}));
        barrier();
        simd<float, 16> t = slm_block_load<float, 16>(0) + slm_block_load<float, 16>(64) + slm_block_load<float, 16>(128) + slm_block_load<float, 16>(192);
        const float sc = 1.0f / es::sqrt_ieee<float>(reduce<float>(t, std::plus<>{}) / 2048.0f + eps);
        quant32(v0 * block_load<float, 16>(w) * sc, v1 * block_load<float, 16>(w + 16) * sc,
                a.q + c * 4096 + hf * 2048 + i, a.d + c * 128 + hf * 64 + b, a.s + c * 256 + hf * 128 + 2 * b);
    });
}

// ---------------------------------------------------------------- layer-major prefill: X[prm[5] + c] += Y2[c] (the MoE output of the sub-batch)
static inline sycl::event k_resid_add(sycl::queue & q, int nc, float * X, const float * Y2, const int * prm) {
    return q.parallel_for(sycl::range<1>((size_t) nc * 128), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int c = (int) id[0] / 128, i = ((int) id[0] % 128) * 16; float * x = X + (size_t) (prm[5] + c) * 2048 + i;
        block_store<float, 16>(x, block_load<float, 16>(x) + block_load<float, 16>(Y2 + c * 2048 + i));
    });
}

// ---------------------------------------------------------------- Q8_0 row-SoA mat-vec, NC columns (compile time)
// y[col(c)][r] = W[r] . act[c]; output column c goes to y + ((prm[0] * ring + c) & pmask) * ystride (ring or plain buffer)
template <int NC>
static inline sycl::event k_gemv_q8(sycl::queue & q, const uint8_t * W, int rows, int n, Act a, float * y, const int * prm, int ring, int pmask, int ystride) {
    const int rb = n + n / 16; const int nb = n / 32;
    return q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int r = (int) id[0]; const uint8_t * w = W + (size_t) r * rb; const sycl::half * wd = (const sycl::half *) (w + n);
        simd<float, 16> acc[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) acc[c] = 0.0f;
        for (int c0 = 0; c0 < n; c0 += 256) {
            simd<int8_t, 256> w8 = block_load<int8_t, 256>((const int8_t *) w + c0);
            simd<int, 64> w32 = w8.template bit_cast_view<int>();
            simd<float, 8> wsc = convert<float>(block_load<sycl::half, 8>(wd + c0 / 32));
#pragma unroll
            for (int c = 0; c < NC; c++) {
                simd<int8_t, 256> x8 = block_load<int8_t, 256>(a.q + c * n + c0);
                simd<int, 64> x32 = x8.template bit_cast_view<int>();
                simd<float, 8> sc = wsc * block_load<float, 8>(a.d + c * nb + c0 / 32);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    simd<int, 16> wa = w32.template select<16, 1>(16 * j), xa = x32.template select<16, 1>(16 * j);
                    simd<float, 16> p = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), wa, xa));
                    simd<float, 16> s; const float s0 = sc[2 * j], s1 = sc[2 * j + 1];
                    s.template select<8, 1>(0) = s0; s.template select<8, 1>(8) = s1;
                    acc[c] += p * s;
                }
            }
        }
        const int p0 = ring ? prm[0] : 0;
#pragma unroll
        for (int c = 0; c < NC; c++) y[(size_t) ((p0 + c) & pmask) * ystride + r] = reduce<float>(acc[c], std::plus<>{});
    });
}

// ---------------------------------------------------------------- f32 mat-vec (router, alpha/beta), float input
// one work-group of 16 threads per row (thread = 128 columns), SLM reduce
static inline sycl::event k_gemv_f32(sycl::queue & q, const float * W, int rows, int n, int nc, const float * x, float * y, int ystride) {
    const int per = n / 16;
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) rows * 16), sycl::range<1>(16)), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
        slm_init<16 * NCMAX * 4>();
        const int r = (int) it.get_group(0), t = (int) it.get_local_id(0); const float * w = W + (size_t) r * n + t * per;
        simd<float, 16> acc[NCMAX];
        for (int c = 0; c < nc; c++) acc[c] = 0.0f;
        for (int i = 0; i < per; i += 16) {
            simd<float, 16> wv = block_load<float, 16>(w + i);
            for (int c = 0; c < nc; c++) acc[c] += wv * block_load<float, 16>(x + c * n + t * per + i);
        }
        for (int c = 0; c < nc; c++) slm_scalar_store<float>((t * NCMAX + c) * 4, reduce<float>(acc[c], std::plus<>{}));
        barrier();
        if (t < nc) { // thread t sums column t
            float s = 0.0f;
#pragma unroll
            for (int k = 0; k < 16; k++) s += slm_scalar_load<float>((k * NCMAX + t) * 4);
            y[t * ystride + r] = s;
        }
    });
}

// f32 mat-vec, NC columns: work-group of 8 threads per 4 rows, thread = 1/8 of the columns, 64-wide loads, SLM reduce
// G != nullptr (alpha/beta of a delta-net layer, rows = 64): instead of y, write the gates of the ring slot of every column:
// G[slot][h] = exp(softplus(alpha_h + dt_h) * a_h), G[slot][32 + h] = sigmoid(beta_h)   (what k_gdn_gates computes)
template <int NC>
static inline sycl::event k_gemv_f32n(sycl::queue & q, const float * W, int rows, int n, const float * x, float * y, int ystride,
                                      const int * prm = nullptr, const float * par_dt = nullptr, const float * par_a = nullptr, float * G = nullptr) {
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) ((rows + 3) / 4) * 8), sycl::range<1>(8)), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
        slm_init<8 * 4 * NC * 4>();
        const int r0 = (int) it.get_group(0) * 4, t = (int) it.get_local_id(0); const int per = n / 8;
        simd<float, 16> acc[4][NC];
#pragma unroll
        for (int k = 0; k < 4; k++)
#pragma unroll
            for (int c = 0; c < NC; c++) acc[k][c] = 0.0f;
        const float * w[4];
#pragma unroll
        for (int k = 0; k < 4; k++) w[k] = W + (size_t) std::min(r0 + k, rows - 1) * n + t * per;
        for (int i = 0; i < per; i += 64) {
            simd<float, 64> wv[4];
#pragma unroll
            for (int k = 0; k < 4; k++) wv[k] = block_load<float, 64>(w[k] + i);
#pragma unroll
            for (int c = 0; c < NC; c++) {
                simd<float, 64> xv = block_load<float, 64>(x + (size_t) c * n + t * per + i);
#pragma unroll
                for (int k = 0; k < 4; k++) { simd<float, 64> p = wv[k] * xv; acc[k][c] += p.select<16, 1>(0) + p.select<16, 1>(16) + p.select<16, 1>(32) + p.select<16, 1>(48); }
            }
        }
#pragma unroll
        for (int k = 0; k < 4; k++)
#pragma unroll
            for (int c = 0; c < NC; c++) slm_scalar_store<float>(((t * 4 + k) * NC + c) * 4, reduce<float>(acc[k][c], std::plus<>{}));
        barrier();
        if (t < 4 && r0 + t < rows) {
#pragma unroll
            for (int c = 0; c < NC; c++) { float s = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; j++) s += slm_scalar_load<float>(((j * 4 + t) * NC + c) * 4);
                if (G) {
                    const int r = r0 + t; float * g = G + (size_t) ((prm[0] + c) & RMASK) * 64;
                    if (r < 32) { simd<float, 1> al = s + par_dt[r]; simd<float, 1> sp = es::log(es::exp(al) + 1.0f); sp.merge(al, al > 20.0f);
                                  simd<float, 1> gv = es::exp(sp * par_a[r]); g[r] = gv[0]; }
                    else { simd<float, 1> bv = simd<float, 1>(1.0f) / (es::exp(-simd<float, 1>(s)) + 1.0f); g[r] = bv[0]; }
                } else y[(size_t) c * ystride + r0 + t] = s; }
        }
    });
}

// ---------------------------------------------------------------- GDN: gates. AB[c][64] (alpha, beta raw) -> G[slot][64] = (g = exp(softplus(alpha + dt) * a), sigmoid(beta))
static inline sycl::event k_gdn_gates(sycl::queue & q, int nc, const int * prm, const float * AB, const float * par_dt, const float * par_a, float * G) {
    return q.parallel_for(sycl::range<1>(nc * 2), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int P = prm[0]; const int c = (int) id[0] / 2, hf = (int) id[0] % 2; float * g = G + (size_t) ((P + c) & RMASK) * 64;
        simd<float, 16> al = block_load<float, 16>(AB + c * 64 + hf * 16) + block_load<float, 16>(par_dt + hf * 16);
        simd<float, 16> sp = es::log(es::exp(al) + 1.0f);
        sp.merge(al, al > 20.0f);
        block_store<float, 16>(g + hf * 16, es::exp(sp * block_load<float, 16>(par_a + hf * 16)));
        block_store<float, 16>(g + 32 + hf * 16, sigmoid16(block_load<float, 16>(AB + c * 64 + 32 + hf * 16)));
    });
}

// ---------------------------------------------------------------- GDN: causal conv (4 taps from the ring) + SiLU, L2 norm for q/k heads
// YR[slot][12288]: raw qkv (8192) + z (4096) per position; conv weights tap-major [4][8192]; out QKV[slot][8192]
static inline sycl::event k_gdn_conv(sycl::queue & q, int nc, const int * prm, const float * YR, const float * cw, float eps, float * QKV) {
    return q.parallel_for(sycl::range<1>(nc * 64), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int c = (int) id[0] / 64, g = (int) id[0] % 64; const int p = prm[0] + c; const int ch0 = g * 128;
        simd<float, 16> v[8]; simd<float, 16> ss = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            simd<float, 16> a = 0.0f;
#pragma unroll
            for (int t = 0; t < 4; t++) {
                const float * src = YR + (size_t) ((p - 3 + t) & RMASK) * 12288 + ch0 + 16 * j;
                a += block_load<float, 16>(src) * block_load<float, 16>(cw + t * 8192 + ch0 + 16 * j);
            }
            v[j] = silu16(a); ss += v[j] * v[j];
        }
        float sc = 1.0f;
        if (g < 32) sc = 1.0f / es::sqrt_ieee<float>(reduce<float>(ss, std::plus<>{}) + eps);
        float * dst = QKV + (size_t) (p & RMASK) * 8192 + ch0;
#pragma unroll
        for (int j = 0; j < 8; j++) block_store<float, 16>(dst + 16 * j, v[j] * sc);
    });
}

// ---------------------------------------------------------------- GDN: state update. One thread per (head, row j).
// S[h][j][128] holds the state after position s_pos-1; records of positions [s_pos, P) are applied and written back first,
// then positions [P, P+nc) are computed without writing S (their records go to the rings: speculative until committed).
static inline sycl::event k_gdn_core(sycl::queue & q, int nc, const int * prm, float * S, const float * QKV, const float * G, float * Dr, float * O) {
    return q.parallel_for(sycl::range<1>(32 * 128), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int P = prm[0], s_pos = prm[1]; const int h = (int) id[0] / 128, j = (int) id[0] % 128; const int kh = h % 16;
        float * sp = S + ((size_t) h * 128 + j) * 128;
        simd<float, 16> s[8];
#pragma unroll
        for (int i = 0; i < 8; i++) s[i] = block_load<float, 16>(sp + 16 * i);
        for (int p = s_pos; p < P; p++) {
            const int sl = p & RMASK; const float g = G[sl * 64 + h]; const float dl = Dr[(size_t) sl * 4096 + h * 128 + j];
            const float * k = QKV + (size_t) sl * 8192 + 2048 + kh * 128;
#pragma unroll
            for (int i = 0; i < 8; i++) s[i] = s[i] * g + block_load<float, 16>(k + 16 * i) * dl;
        }
        if (P > s_pos) {
#pragma unroll
            for (int i = 0; i < 8; i++) block_store<float, 16>(sp + 16 * i, s[i]);
        }
        for (int c = 0; c < nc; c++) {
            const int sl = (P + c) & RMASK; const float g = G[sl * 64 + h], beta = G[sl * 64 + 32 + h];
            const float * qv = QKV + (size_t) sl * 8192 + kh * 128; const float * k = qv + 2048;
            const float vj = QKV[(size_t) sl * 8192 + 4096 + h * 128 + j];
            simd<float, 16> kk[8]; simd<float, 16> a = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) { kk[i] = block_load<float, 16>(k + 16 * i); a += s[i] * kk[i]; }
            const float dl = (vj - g * reduce<float>(a, std::plus<>{})) * beta;
            simd<float, 16> o = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) { s[i] = s[i] * g + kk[i] * dl; o += s[i] * block_load<float, 16>(qv + 16 * i); }
            Dr[(size_t) sl * 4096 + h * 128 + j] = dl;
            O[c * 4096 + h * 128 + j] = reduce<float>(o, std::plus<>{}) * 0.08838834764831845f; // 1/sqrt(128)
        }
    });
}

// ---------------------------------------------------------------- GDN: gated RMS norm per head + quantize -> act(4096)
static inline sycl::event k_gdn_post(sycl::queue & q, int nc, const int * prm, const float * O, const float * YR, const float * nw, float eps, Act a) {
    return q.parallel_for(sycl::range<1>(nc * 32), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int c = (int) id[0] / 32, h = (int) id[0] % 32; const float * o = O + c * 4096 + h * 128;
        const float * z = YR + (size_t) ((prm[0] + c) & RMASK) * 12288 + 8192 + h * 128;
        simd<float, 16> v[8]; simd<float, 16> ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; i++) { v[i] = block_load<float, 16>(o + 16 * i); ss += v[i] * v[i]; }
        const float sc = 1.0f / es::sqrt_ieee<float>(reduce<float>(ss, std::plus<>{}) / 128.0f + eps);
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = v[i] * sc * block_load<float, 16>(nw + 16 * i) * silu16(block_load<float, 16>(z + 16 * i));
#pragma unroll
        for (int b = 0; b < 4; b++) quant32(v[2 * b], v[2 * b + 1], a.q + c * 4096 + h * 128 + 32 * b, a.d + c * 128 + h * 4 + b, a.s + c * 256 + h * 8 + 2 * b);
    });
}

// ---------------------------------------------------------------- attention: int8 K/V cache in XMX tile layout
// Per kv head (2) and tile of 16 positions t (tb = kvh * ntile + t):
//   kt[tb][b]  512 B : int8 VNNI tile of block b (32 dims): dword k4 * 16 + n = dims 4 k4 .. 4 k4 + 3 of position 16 t + n
//   ks[tb][b][16]    : K block scales;   vs[tb][b][16]: V block scales
//   vt[tb][g]  256 B : int8 V of dims 16 g .. 16 g + 15 in fp16-VNNI order: byte (k2 * 16 + d) * 2 + j = position 2 k2 + j, dim d
// Queries go to QN[c][kvh][b] = 8 heads x 32 int8 (natural order) and QSN[c][kvh][b][8] = scales / 16 (1/sqrt(256) folded in).
struct KV { int8_t * kt; float * ks; int8_t * vt; float * vs; int ntile; };

static ESIMD_INLINE float quant32_q(simd<float, 16> a0, simd<float, 16> a1, simd<int8_t, 32> & q) {
    const float amax = hmax<float>((es::max)(es::abs(a0), es::abs(a1)));
    const float d = amax / 127.0f; const float id = d > 0.0f ? 1.0f / d : 0.0f;
    q.select<16, 1>(0) = convert<int8_t>(rnde<int, 16>(a0 * id)); q.select<16, 1>(16) = convert<int8_t>(rnde<int, 16>(a1 * id));
    return d;
}

// per-head prep, one thread per (column, unit): 16 q heads, 2 k heads, 2 v heads.
// Y1[c][9216] = q (16 x [q 256 | gate 256]), k (2 x 256), v (2 x 256); rope[c][64] = 32 cos, 32 sin
static inline sycl::event k_att_prep(sycl::queue & q, int nc, const int * prm, const float * Y1, const float * qw, const float * kw, float eps, const float * rope,
                                     int8_t * QN, float * QSN, KV kv) {
    return q.parallel_for(sycl::range<1>(nc * 20), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int P = prm[0]; const int c = (int) id[0] / 20, u = (int) id[0] % 20; const float * y = Y1 + c * 9216;
        const int pos = P + c; const int t = pos / 16, n = pos % 16;
        if (u >= 18) { // v head
            const int kvh = u - 18; const float * v = y + 8704 + kvh * 256; const size_t tb = (size_t) kvh * kv.ntile + t;
            simd<uint32_t, 16> d16(0u, 1u); simd<uint32_t, 16> boff = (((n >> 1) * 16u) + d16) * 2u + (uint32_t) (n & 1);
#pragma unroll
            for (int b = 0; b < 8; b++) {
                simd<int8_t, 32> qb; const float d = quant32_q(block_load<float, 16>(v + 32 * b), block_load<float, 16>(v + 32 * b + 16), qb);
                kv.vs[(tb * 8 + b) * 16 + n] = d;
#pragma unroll
                for (int h = 0; h < 2; h++) { const int g = 2 * b + h;
                    simd<uint8_t, 16> vv = qb.select<16, 1>(16 * h).bit_cast_view<uint8_t>();
                    scatter<uint8_t, 16>((uint8_t *) kv.vt + (tb * 16 + g) * 256, boff, vv); }
            }
            return;
        }
        const bool isk = u >= 16; const float * src = isk ? y + 8192 + (u - 16) * 256 : y + u * 512; const float * nw = isk ? kw : qw;
        simd<float, 16> v[16]; simd<float, 16> ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i++) { v[i] = block_load<float, 16>(src + 16 * i); ss += v[i] * v[i]; }
        const float sc = 1.0f / es::sqrt_ieee<float>(reduce<float>(ss, std::plus<>{}) / 256.0f + eps);
#pragma unroll
        for (int i = 0; i < 16; i++) v[i] = v[i] * sc * block_load<float, 16>(nw + 16 * i);
        const float * rp = rope + c * 64;
#pragma unroll
        for (int i = 0; i < 2; i++) { // pairs (d, d + 32), d < 32
            simd<float, 16> cs = block_load<float, 16>(rp + 16 * i), sn = block_load<float, 16>(rp + 32 + 16 * i);
            simd<float, 16> a = v[i], b = v[i + 2];
            v[i] = a * cs - b * sn; v[i + 2] = a * sn + b * cs;
        }
        if (isk) {
            const int kvh = u - 16; const size_t tb = (size_t) kvh * kv.ntile + t;
            simd<uint32_t, 8> k4(0u, 1u); simd<uint32_t, 8> doff = (k4 * 16u + (uint32_t) n) * 4u;
#pragma unroll
            for (int b = 0; b < 8; b++) {
                simd<int8_t, 32> qb; const float d = quant32_q(v[2 * b], v[2 * b + 1], qb);
                kv.ks[(tb * 8 + b) * 16 + n] = d;
                scatter<uint32_t, 8>((uint32_t *) (kv.kt + (tb * 8 + b) * 512), doff, qb.bit_cast_view<uint32_t>());
            }
        } else {
            const int kvh = u / 8, m = u % 8; int8_t * qn = QN + ((size_t) c * 16 + kvh * 8) * 256 + m * 32; float * qs = QSN + ((size_t) c * 16 + kvh * 8) * 8 + m;
#pragma unroll
            for (int b = 0; b < 8; b++) { simd<int8_t, 32> qb; const float d = quant32_q(v[2 * b], v[2 * b + 1], qb); block_store<int8_t, 32>(qn + b * 256, qb); qs[b * 8] = d * 0.0625f; }
        }
    });
}

// ---------------------------------------------------------------- attention: combine chunks, gate, quantize -> act(4096)
// work-group of 32 threads per (column, head): thread t merges chunks t, t + 32, ... ; the 32 partial results meet in SLM,
// then 8 threads finish 32 dims each.
static inline sycl::event k_att_fin(sycl::queue & q, int nc, const int * prm, int nchb, const float * PART, const float * Y1, Act a) {
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) nc * 16 * 32), sycl::range<1>(32)), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
        slm_init<32 * 260 * 4>();
        const int g = (int) it.get_group(0), t = (int) it.get_local_id(0); const int c = g / 16, h = g % 16;
        const int nch = prm[2]; const float * part = PART + ((size_t) c * 16 + h) * nchb * 272;
        float M = -1e30f;
        for (int ch = t; ch < nch; ch += 32) { const float m = part[(size_t) ch * 272]; if (part[(size_t) ch * 272 + 1] > 0.0f && m > M) M = m; }
        simd<float, 16> o[16];
#pragma unroll
        for (int i = 0; i < 16; i++) o[i] = 0.0f;
        float L = 0.0f;
        for (int ch = t; ch < nch; ch += 32) {
            const float * p = part + (size_t) ch * 272; const float l = p[1];
            if (l > 0.0f) {
                simd<float, 1> fv = es::exp(simd<float, 1>(p[0] - M)); const float f = fv[0]; L += l * f;
#pragma unroll
                for (int i = 0; i < 16; i++) o[i] += block_load<float, 16>(p + 16 + 16 * i) * f;
            }
        }
        const uint32_t base = (uint32_t) t * 260 * 4;
#pragma unroll
        for (int i = 0; i < 16; i++) slm_block_store<float, 16>(base + 16 * 4 * i, o[i]);
        simd<float, 4> ml = 0.0f; ml[0] = M; ml[1] = L; slm_block_store<float, 4>(base + 256 * 4, ml);
        barrier();
        if (t < 8) {
            float Mg = -1e30f;
            for (int j = 0; j < 32; j++) { const float lj = slm_scalar_load<float>((uint32_t) j * 1040 + 257 * 4), mj = slm_scalar_load<float>((uint32_t) j * 1040 + 256 * 4); if (lj > 0.0f && mj > Mg) Mg = mj; }
            simd<float, 32> acc = 0.0f; float Lg = 0.0f;
            for (int j = 0; j < 32; j++) {
                const float lj = slm_scalar_load<float>((uint32_t) j * 1040 + 257 * 4); if (!(lj > 0.0f)) continue;
                simd<float, 1> fv = es::exp(simd<float, 1>(slm_scalar_load<float>((uint32_t) j * 1040 + 256 * 4) - Mg)); const float f = fv[0];
                Lg += lj * f; acc += slm_block_load<float, 32>((uint32_t) j * 1040 + (uint32_t) t * 32 * 4) * f;
            }
            const float inv = 1.0f / Lg; const float * gate = Y1 + c * 9216 + h * 512 + 256 + 32 * t;
            simd<float, 16> o0 = acc.select<16, 1>(0) * inv * sigmoid16(block_load<float, 16>(gate)), o1 = acc.select<16, 1>(16) * inv * sigmoid16(block_load<float, 16>(gate + 16));
            quant32(o0, o1, a.q + c * 4096 + h * 256 + 32 * t, a.d + c * 128 + h * 8 + t, a.s + c * 256 + h * 16 + 2 * t);
        }
    });
}

// ---------------------------------------------------------------- router: softmax + top-8 (one thread per column)
// RL[c][272]: 256 router logits + shared expert gate logit at [256]. Outputs (device + host): ids[c][8], w[c][8] (normalized), shw[c]
static inline sycl::event k_topk(sycl::queue & q, int nc, const float * RL, int * ids_d, float * w_d, float * shw_d, int * ids_h, float * w_h, const int * prm = nullptr, bool win = false) {
    return q.parallel_for(sycl::range<1>(nc), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int c = (int) id[0]; const float * l = RL + c * 272;
        simd<float, 16> v[16]; float mx = -1e30f;
#pragma unroll
        for (int i = 0; i < 16; i++) { v[i] = block_load<float, 16>(l + 16 * i); const float m = hmax<float>(v[i]); if (m > mx) mx = m; }
        simd<uint32_t, 16> key[16]; simd<uint32_t, 16> lane(0u, 1u);
#pragma unroll
        for (int i = 0; i < 16; i++) {
            v[i] = es::exp(v[i] - mx); // in (0, 1]: bit pattern order == value order
            simd<uint32_t, 16> bits = v[i].bit_cast_view<uint32_t>();
            key[i] = (bits & 0xFFFFFF00u) | (simd<uint32_t, 16>(255u - 16u * i) - lane);
        }
        float wsel[8]; int isel[8]; float wsum = 0.0f;
        for (int u = 0; u < 8; u++) {
            uint32_t best = 0;
#pragma unroll
            for (int i = 0; i < 16; i++) { const uint32_t k = hmax<uint32_t>(key[i]); if (k > best) best = k; }
            const int e = 255 - (int) (best & 0xFFu);
            isel[u] = e; const float pv = v[e >> 4][e & 15]; wsel[u] = pv; wsum += pv;
            key[e >> 4][e & 15] = 0u;
        }
        if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f;
        for (int u = 0; u < 8; u++) {
            const float w = wsel[u] / wsum;
            ids_d[c * 8 + u] = isel[u]; w_d[c * 8 + u] = w;
            if (win) ids_h[(size_t) (prm[5] + c) * 8 + u] = isel[u]; else { ids_h[c * 8 + u] = isel[u]; w_h[c * 8 + u] = w; }
        }
        simd<float, 1> sg(l[256]); simd<float, 1> r = simd<float, 1>(1.0f) / (es::exp(-sg) + 1.0f);
        shw_d[c] = r[0];
    });
}

// ---------------------------------------------------------------- shared expert gate/up (Q8_0 rows interleaved [gate_i | up_i]) + SwiGLU -> ACT[c][512]
static inline sycl::event k_sh_glu(sycl::queue & q, const uint8_t * W, int nc, Act a, float * ACT, int astride) {
    const int n = 2048, rb = 2176;
    return q.parallel_for(sycl::range<1>(512), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int r = (int) id[0]; const uint8_t * wg = W + (size_t) (2 * r) * rb; const uint8_t * wu = wg + rb;
        simd<float, 16> ag[NCMAX], au[NCMAX];
        for (int c = 0; c < nc; c++) { ag[c] = 0.0f; au[c] = 0.0f; }
        for (int c0 = 0; c0 < n; c0 += 256) {
            simd<int8_t, 256> g8 = block_load<int8_t, 256>((const int8_t *) wg + c0), u8 = block_load<int8_t, 256>((const int8_t *) wu + c0);
            simd<int, 64> g32 = g8.bit_cast_view<int>(), u32 = u8.bit_cast_view<int>();
            simd<float, 8> gsc = convert<float>(block_load<sycl::half, 8>((const sycl::half *) (wg + n) + c0 / 32));
            simd<float, 8> usc = convert<float>(block_load<sycl::half, 8>((const sycl::half *) (wu + n) + c0 / 32));
            for (int c = 0; c < nc; c++) {
                simd<int8_t, 256> x8 = block_load<int8_t, 256>(a.q + c * n + c0); simd<int, 64> x32 = x8.bit_cast_view<int>();
                simd<float, 8> xd = block_load<float, 8>(a.d + c * 64 + c0 / 32);
                simd<float, 8> sg = gsc * xd, su = usc * xd;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    simd<int, 16> xa = x32.select<16, 1>(16 * j);
                    simd<float, 16> pg = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), simd<int, 16>(g32.select<16, 1>(16 * j)), xa));
                    simd<float, 16> pu = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), simd<int, 16>(u32.select<16, 1>(16 * j)), xa));
                    simd<float, 16> s; float s0 = sg[2 * j], s1 = sg[2 * j + 1];
                    s.select<8, 1>(0) = s0; s.select<8, 1>(8) = s1; ag[c] += pg * s;
                    s0 = su[2 * j]; s1 = su[2 * j + 1];
                    s.select<8, 1>(0) = s0; s.select<8, 1>(8) = s1; au[c] += pu * s;
                }
            }
        }
        for (int c = 0; c < nc; c++) {
            simd<float, 1> g(reduce<float>(ag[c], std::plus<>{})); simd<float, 1> sl = g / (es::exp(-g) + 1.0f);
            ACT[c * astride + r] = sl[0] * reduce<float>(au[c], std::plus<>{});
        }
    });
}

// ---------------------------------------------------------------- quantize nv vectors of 512 floats per column: ACT[c][v][512] -> act
// GU != nullptr: vector 0 of every column is the shared expert, SwiGLU of GU[c][1024] (interleaved gate_i, up_i)
static inline sycl::event k_act_quant(sycl::queue & q, int nc, int nv, int vmax, const float * ACT, Act a, const float * GU = nullptr) {
    return q.parallel_for(sycl::range<1>((size_t) nc * nv * 16), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int b = (int) id[0] % 16; const int v = (int) (id[0] / 16) % nv; const int c = (int) (id[0] / 16) / nv;
        const size_t vec = (size_t) c * vmax + v; const float * src = ACT + vec * 512 + 32 * b;
        simd<float, 16> v0, v1;
        if (GU && v == 0) { const float * g = GU + (size_t) c * 1024 + 64 * b; simd<float, 32> t0 = block_load<float, 32>(g), t1 = block_load<float, 32>(g + 32);
            v0 = silu16(t0.select<16, 2>(0)) * t0.select<16, 2>(1); v1 = silu16(t1.select<16, 2>(0)) * t1.select<16, 2>(1); }
        else { v0 = block_load<float, 16>(src); v1 = block_load<float, 16>(src + 16); }
        quant32(v0, v1, a.q + vec * 512 + 32 * b, a.d + vec * 16 + b, a.s + vec * 32 + 2 * b);
    });
}

// ---------------------------------------------------------------- shared expert down (Q8_0, 512 -> 2048), scaled by the shared gate -> Y2[c][2048]
static inline sycl::event k_sh_down(sycl::queue & q, const uint8_t * W, int nc, int vmax, Act a, const float * shw, float * Y2) {
    const int n = 512, rb = 544;
    return q.parallel_for(sycl::range<1>(2048), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int r = (int) id[0]; const uint8_t * w = W + (size_t) r * rb;
        for (int c = 0; c < nc; c++) {
            const size_t vec = (size_t) c * vmax; simd<float, 16> acc = 0.0f;
#pragma unroll
            for (int c0 = 0; c0 < n; c0 += 256) {
                simd<int8_t, 256> w8 = block_load<int8_t, 256>((const int8_t *) w + c0); simd<int, 64> w32 = w8.bit_cast_view<int>();
                simd<float, 8> sc = convert<float>(block_load<sycl::half, 8>((const sycl::half *) (w + n) + c0 / 32)) * block_load<float, 8>(a.d + vec * 16 + c0 / 32);
                simd<int8_t, 256> x8 = block_load<int8_t, 256>(a.q + vec * 512 + c0); simd<int, 64> x32 = x8.bit_cast_view<int>();
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    simd<float, 16> p = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), simd<int, 16>(w32.select<16, 1>(16 * j)), simd<int, 16>(x32.select<16, 1>(16 * j))));
                    simd<float, 16> s; const float s0 = sc[2 * j], s1 = sc[2 * j + 1];
                    s.select<8, 1>(0) = s0; s.select<8, 1>(8) = s1; acc += p * s;
                }
            }
            Y2[c * 2048 + r] = reduce<float>(acc, std::plus<>{}) * shw[c];
        }
    });
}

// ---------------------------------------------------------------- LM head: Q6_K row-SoA (n = 2048: ql 1024 | qh 512 | scales 128 | d 16), one thread per row
// if rowidx != nullptr the kernel computes rows rowidx[i] (draft vocabulary subset)
template <int NC>
static inline sycl::event k_head_q6k(sycl::queue & q, const uint8_t * W, int rows, Act a, float * LOG, int lstride, const int * rowidx) {
    const int n = 2048, rb = 1680;
    return q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int i = (int) id[0]; const int r = rowidx ? rowidx[i] : i; const uint8_t * w = W + (size_t) r * rb;
        simd<float, 8> dv = convert<float>(block_load<sycl::half, 8>((const sycl::half *) (w + 1664)));
        simd<float, 16> acc[NC]; float corr[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) { acc[c] = 0.0f; corr[c] = 0.0f; }
        for (int b = 0; b < 8; b++) {
            const float d = dv[b];
            simd<uint8_t, 128> ql = block_load<uint8_t, 128>(w + 128 * b); simd<uint8_t, 64> qh = block_load<uint8_t, 64>(w + 1024 + 64 * b);
            simd<int8_t, 16> sc8 = block_load<int8_t, 16>((const int8_t *) w + 1536 + 16 * b);
            simd<float, 16> scf = convert<float>(sc8) * d; // 16 sub-block scales (16 weights each)
            simd<uint32_t, 32> ql32 = ql.bit_cast_view<uint32_t>(); simd<uint32_t, 16> qh32 = qh.bit_cast_view<uint32_t>();
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                simd<uint32_t, 8> A = ql32.select<8, 1>(16 * hf), B = ql32.select<8, 1>(16 * hf + 8), H = qh32.select<8, 1>(8 * hf);
                simd<uint32_t, 16> qa, qb; // weights [0..63] and [64..127] of this half, 6-bit values as bytes
                qa.select<8, 1>(0) = (A & 0x0F0F0F0Fu) | ((H & 0x03030303u) << 4);
                qa.select<8, 1>(8) = (B & 0x0F0F0F0Fu) | (((H >> 2) & 0x03030303u) << 4);
                qb.select<8, 1>(0) = ((A >> 4) & 0x0F0F0F0Fu) | (((H >> 4) & 0x03030303u) << 4);
                qb.select<8, 1>(8) = ((B >> 4) & 0x0F0F0F0Fu) | (((H >> 6) & 0x03030303u) << 4);
                simd<int, 16> qai = qa.bit_cast_view<int>(), qbi = qb.bit_cast_view<int>();
                simd<float, 16> sa, sb; // per-lane weight scale: 4 lanes per sub-block of 16
#pragma unroll
                for (int k = 0; k < 4; k++) { const float f0 = scf[8 * hf + k], f1 = scf[8 * hf + 4 + k]; sa.select<4, 1>(4 * k) = f0; sb.select<4, 1>(4 * k) = f1; }
                const int x0 = 256 * b + 128 * hf;
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    simd<int8_t, 128> x8 = block_load<int8_t, 128>(a.q + c * n + x0); simd<int, 32> x32 = x8.bit_cast_view<int>();
                    simd<float, 4> xd = block_load<float, 4>(a.d + c * 64 + x0 / 32); simd<float, 8> xs = block_load<float, 8>(a.s + c * 128 + x0 / 16);
                    simd<float, 16> da, db; const float d0 = xd[0], d1 = xd[1], d2 = xd[2], d3 = xd[3];
                    da.select<8, 1>(0) = d0; da.select<8, 1>(8) = d1; db.select<8, 1>(0) = d2; db.select<8, 1>(8) = d3;
                    simd<float, 16> pa = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), qai, simd<int, 16>(x32.select<16, 1>(0))));
                    simd<float, 16> pb = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), qbi, simd<int, 16>(x32.select<16, 1>(16))));
                    acc[c] += pa * sa * da + pb * sb * db;
                    simd<float, 8> s8 = scf.select<8, 1>(8 * hf);
                    corr[c] += reduce<float>(s8 * xs, std::plus<>{});
                }
            }
        }
#pragma unroll
        for (int c = 0; c < NC; c++) LOG[(size_t) c * lstride + i] = reduce<float>(acc[c], std::plus<>{}) - 32.0f * corr[c];
    });
}

// ================================================================ routed experts on the GPU (the VRAM expert cache)
// TAB[256] = device address of the expert record of this layer, 0 if the expert is not cached (the CPU computes it).
// record: 512 x [gate row | up row] (Q4_K row-SoA, 1152 B each), then 2048 x down row (Q5_K 352 B or Q6_K 432 B)

// get_scale_min_k4 for one block: 12 packed bytes -> 8 scales * d and 8 mins * dmin
static ESIMD_INLINE void k4_unpack(simd<uint8_t, 12> sc12, float d, float dmin, simd<float, 8> & scale_f, simd<float, 8> & min_f) {
    simd<uint8_t, 8> sc = 0, m = 0;
    simd<uint8_t, 4> slo = sc12.select<4, 1>(0), mlo = sc12.select<4, 1>(4), hi = sc12.select<4, 1>(8);
    sc.select<4, 1>(0) = slo & simd<uint8_t, 4>(0x3F);
    sc.select<4, 1>(4) = (hi & simd<uint8_t, 4>(0x0F)) | ((slo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
    m.select<4, 1>(0) = mlo & simd<uint8_t, 4>(0x3F);
    m.select<4, 1>(4) = (hi >> simd<uint8_t, 4>(4)) | ((mlo >> simd<uint8_t, 4>(6)) << simd<uint8_t, 4>(4));
    scale_f = convert<float>(sc) * d; min_f = convert<float>(m) * dmin;
}

// one Q4_K block (256 weights) of a row against one activation block; q32 = 32 dwords of packed nibbles
static ESIMD_INLINE float q4k_block(simd<uint32_t, 32> q32, simd<float, 8> scale_f, simd<float, 8> min_f, simd<int, 64> x32, simd<float, 8> xd, simd<float, 16> xs) {
    simd<float, 8> sd = scale_f * xd; simd<float, 16> acc = 0.0f;
#pragma unroll
    for (int g = 0; g < 4; g++) {
        simd<uint32_t, 8> Q = q32.select<8, 1>(8 * g); simd<uint32_t, 16> qq;
        qq.select<8, 1>(0) = Q & 0x0F0F0F0Fu; qq.select<8, 1>(8) = (Q >> 4) & 0x0F0F0F0Fu;
        simd<int, 16> qi = qq.bit_cast_view<int>();
        simd<float, 16> p = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), qi, simd<int, 16>(x32.select<16, 1>(16 * g))));
        simd<float, 16> s; const float s0 = sd[2 * g], s1 = sd[2 * g + 1];
        s.select<8, 1>(0) = s0; s.select<8, 1>(8) = s1; acc += p * s;
    }
    simd<float, 8> xs32 = xs.select<8, 2>(0) + xs.select<8, 2>(1);
    return reduce<float>(acc, std::plus<>{}) - reduce<float>(min_f * xs32, std::plus<>{});
}

// gate/up + SwiGLU for the cached experts: one thread per (column, used expert u, hidden unit i) -> ACT[c][1 + u][i] (0 if not cached)
// fused version: groups of 32 threads (32 hidden units of one column / slot), the values meet in SLM and are quantized into
// aq (vector c * 9 + 1 + u); groups past nc * 8 * 16 quantize the shared expert (vector c * 9) from GU. Replaces k_act_quant.
static inline sycl::event k_exp_glu_q(sycl::queue & q, int nc, const uint64_t * TAB, const int * ids, Act a, const float * GU, Act aq) {
    const size_t ng = (size_t) nc * 8 * 16 + (size_t) nc * 16;
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>(ng * 32), sycl::range<1>(32)), [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
        slm_init<32 * 4>();
        const size_t gi = it.get_group(0); const int t = (int) it.get_local_id(0);
        if (gi >= (size_t) nc * 8 * 16) { // shared expert: one group = 32 values of vector 0 of column c
            if (t != 0) return;
            const size_t k = gi - (size_t) nc * 8 * 16; const int c = (int) (k / 16), b = (int) (k % 16);
            const float * gp = GU + (size_t) c * 1024 + 64 * b; simd<float, 32> t0 = block_load<float, 32>(gp), t1 = block_load<float, 32>(gp + 32);
            const size_t vec = (size_t) c * 9;
            quant32(silu16(t0.select<16, 2>(0)) * t0.select<16, 2>(1), silu16(t1.select<16, 2>(0)) * t1.select<16, 2>(1), aq.q + vec * 512 + 32 * b, aq.d + vec * 16 + b, aq.s + vec * 32 + 2 * b);
            return;
        }
        const int b = (int) (gi % 16); const int u = (int) (gi / 16) % 8; const int c = (int) (gi / 16) / 8; const int i = 32 * b + t;
        const uint8_t * rec = (const uint8_t *) TAB[ids[c * 8 + u]];
        float val = 0.0f;
        if (rec) {
            const uint8_t * wg = rec + (size_t) (2 * i) * 1152; const uint8_t * wu = wg + 1152;
            simd<uint8_t, 128> mg = block_load<uint8_t, 128>(wg + 1024), mu = block_load<uint8_t, 128>(wu + 1024);
            simd<sycl::half, 16> dmg = block_load<sycl::half, 16>((const sycl::half *) (wg + 1120)), dmu = block_load<sycl::half, 16>((const sycl::half *) (wu + 1120));
            simd<float, 16> fg = convert<float>(dmg), fu = convert<float>(dmu);
            float sg = 0.0f, su = 0.0f;
            for (int bb = 0; bb < 8; bb++) {
                simd<int8_t, 256> x8 = block_load<int8_t, 256>(a.q + c * 2048 + 256 * bb); simd<int, 64> x32 = x8.bit_cast_view<int>();
                simd<float, 8> xd = block_load<float, 8>(a.d + c * 64 + 8 * bb); simd<float, 16> xs = block_load<float, 16>(a.s + c * 128 + 16 * bb);
                simd<float, 8> scf, mnf;
                simd<uint8_t, 128> qg = block_load<uint8_t, 128>(wg + 128 * bb);
                k4_unpack(mg.select<12, 1>(12 * bb), fg[2 * bb], fg[2 * bb + 1], scf, mnf);
                sg += q4k_block(qg.bit_cast_view<uint32_t>(), scf, mnf, x32, xd, xs);
                simd<uint8_t, 128> qu = block_load<uint8_t, 128>(wu + 128 * bb);
                k4_unpack(mu.select<12, 1>(12 * bb), fu[2 * bb], fu[2 * bb + 1], scf, mnf);
                su += q4k_block(qu.bit_cast_view<uint32_t>(), scf, mnf, x32, xd, xs);
            }
            simd<float, 1> g(sg); simd<float, 1> sl = g / (es::exp(-g) + 1.0f); val = sl[0] * su;
        }
        slm_scalar_store<float>(t * 4, val);
        barrier();
        if (t == 0) {
            const size_t vec = (size_t) c * 9 + 1 + u;
            quant32(slm_block_load<float, 16>(0), slm_block_load<float, 16>(64), aq.q + vec * 512 + 32 * b, aq.d + vec * 16 + b, aq.s + vec * 32 + 2 * b);
        }
    });
}

static inline sycl::event k_exp_glu(sycl::queue & q, int nc, const uint64_t * TAB, const int * ids, Act a, float * ACT) {
    return q.parallel_for(sycl::range<1>((size_t) nc * 8 * 512), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int i = (int) id[0] % 512; const int u = (int) (id[0] / 512) % 8; const int c = (int) (id[0] / 512) / 8;
        const uint8_t * rec = (const uint8_t *) TAB[ids[c * 8 + u]];
        float * dst = ACT + ((size_t) c * 9 + 1 + u) * 512 + i;
        if (!rec) { *dst = 0.0f; return; }
        const uint8_t * wg = rec + (size_t) (2 * i) * 1152; const uint8_t * wu = wg + 1152;
        simd<uint8_t, 128> mg = block_load<uint8_t, 128>(wg + 1024), mu = block_load<uint8_t, 128>(wu + 1024); // 96 B scales + 32 B d/dmin
        simd<sycl::half, 16> dmg = block_load<sycl::half, 16>((const sycl::half *) (wg + 1120)), dmu = block_load<sycl::half, 16>((const sycl::half *) (wu + 1120));
        simd<float, 16> fg = convert<float>(dmg), fu = convert<float>(dmu);
        float sg = 0.0f, su = 0.0f;
        for (int b = 0; b < 8; b++) {
            simd<int8_t, 256> x8 = block_load<int8_t, 256>(a.q + c * 2048 + 256 * b); simd<int, 64> x32 = x8.bit_cast_view<int>();
            simd<float, 8> xd = block_load<float, 8>(a.d + c * 64 + 8 * b); simd<float, 16> xs = block_load<float, 16>(a.s + c * 128 + 16 * b);
            simd<float, 8> scf, mnf;
            simd<uint8_t, 128> qg = block_load<uint8_t, 128>(wg + 128 * b);
            k4_unpack(mg.select<12, 1>(12 * b), fg[2 * b], fg[2 * b + 1], scf, mnf);
            sg += q4k_block(qg.bit_cast_view<uint32_t>(), scf, mnf, x32, xd, xs);
            simd<uint8_t, 128> qu = block_load<uint8_t, 128>(wu + 128 * b);
            k4_unpack(mu.select<12, 1>(12 * b), fu[2 * b], fu[2 * b + 1], scf, mnf);
            su += q4k_block(qu.bit_cast_view<uint32_t>(), scf, mnf, x32, xd, xs);
        }
        simd<float, 1> g(sg); simd<float, 1> sl = g / (es::exp(-g) + 1.0f);
        *dst = sl[0] * su;
    });
}

// MoE output: one thread per output row. Y2[c][r] = shw[c] * sh_down[r].act[c][0] + sum_u w[c][u] * down_u[r].act[c][1+u] over the cached experts
template <bool Q6>
static inline sycl::event k_down(sycl::queue & q, const uint8_t * Wsh, int nc, const uint64_t * TAB, const int * ids, const float * wts, const float * shw, Act a, float * Y2) {
    const size_t down_off = (size_t) 1024 * 1152; const int rd = Q6 ? 432 : 352;
    return q.parallel_for(sycl::range<1>(2048), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int r = (int) id[0]; const uint8_t * w = Wsh + (size_t) r * 544;
        for (int c = 0; c < nc; c++) {
            const size_t vec = (size_t) c * 9; simd<float, 16> acc = 0.0f;
#pragma unroll
            for (int c0 = 0; c0 < 512; c0 += 256) { // shared expert, Q8_0
                simd<int8_t, 256> w8 = block_load<int8_t, 256>((const int8_t *) w + c0); simd<int, 64> w32 = w8.bit_cast_view<int>();
                simd<float, 8> sc = convert<float>(block_load<sycl::half, 8>((const sycl::half *) (w + 512) + c0 / 32)) * block_load<float, 8>(a.d + vec * 16 + c0 / 32);
                simd<int8_t, 256> x8 = block_load<int8_t, 256>(a.q + vec * 512 + c0); simd<int, 64> x32 = x8.bit_cast_view<int>();
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    simd<float, 16> p = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), simd<int, 16>(w32.select<16, 1>(16 * j)), simd<int, 16>(x32.select<16, 1>(16 * j))));
                    simd<float, 16> s; const float s0 = sc[2 * j], s1 = sc[2 * j + 1];
                    s.select<8, 1>(0) = s0; s.select<8, 1>(8) = s1; acc += p * s;
                }
            }
            float y = reduce<float>(acc, std::plus<>{}) * shw[c];
            for (int u = 0; u < 8; u++) {
                const uint8_t * rec = (const uint8_t *) TAB[ids[c * 8 + u]];
                if (!rec) continue;
                const uint8_t * wd = rec + down_off + (size_t) r * rd; const size_t ev = vec + 1 + u; float sum = 0.0f;
                if constexpr (Q6) { // ql 256 | qh 128 | scales 32 | d 2x half
                    simd<float, 2> dv = convert<float>(block_load<sycl::half, 2>((const sycl::half *) (wd + 416)));
                    simd<float, 16> accq = 0.0f; float corr = 0.0f;
#pragma unroll
                    for (int b = 0; b < 2; b++) {
                        const float d = dv[b];
                        simd<uint8_t, 128> ql = block_load<uint8_t, 128>(wd + 128 * b); simd<uint8_t, 64> qh = block_load<uint8_t, 64>(wd + 256 + 64 * b);
                        simd<float, 16> scf = convert<float>(block_load<int8_t, 16>((const int8_t *) wd + 384 + 16 * b)) * d;
                        simd<uint32_t, 32> ql32 = ql.bit_cast_view<uint32_t>(); simd<uint32_t, 16> qh32 = qh.bit_cast_view<uint32_t>();
#pragma unroll
                        for (int hf = 0; hf < 2; hf++) {
                            simd<uint32_t, 8> A = ql32.select<8, 1>(16 * hf), B = ql32.select<8, 1>(16 * hf + 8), H = qh32.select<8, 1>(8 * hf);
                            simd<uint32_t, 16> qa, qb;
                            qa.select<8, 1>(0) = (A & 0x0F0F0F0Fu) | ((H & 0x03030303u) << 4);
                            qa.select<8, 1>(8) = (B & 0x0F0F0F0Fu) | (((H >> 2) & 0x03030303u) << 4);
                            qb.select<8, 1>(0) = ((A >> 4) & 0x0F0F0F0Fu) | (((H >> 4) & 0x03030303u) << 4);
                            qb.select<8, 1>(8) = ((B >> 4) & 0x0F0F0F0Fu) | (((H >> 6) & 0x03030303u) << 4);
                            simd<int, 16> qai = qa.bit_cast_view<int>(), qbi = qb.bit_cast_view<int>();
                            simd<float, 16> sa, sb;
#pragma unroll
                            for (int k = 0; k < 4; k++) { const float f0 = scf[8 * hf + k], f1 = scf[8 * hf + 4 + k]; sa.select<4, 1>(4 * k) = f0; sb.select<4, 1>(4 * k) = f1; }
                            const int x0 = 256 * b + 128 * hf;
                            simd<int8_t, 128> x8 = block_load<int8_t, 128>(a.q + ev * 512 + x0); simd<int, 32> x32 = x8.bit_cast_view<int>();
                            simd<float, 4> xd = block_load<float, 4>(a.d + ev * 16 + x0 / 32); simd<float, 8> xs = block_load<float, 8>(a.s + ev * 32 + x0 / 16);
                            simd<float, 16> da, db; const float d0 = xd[0], d1 = xd[1], d2 = xd[2], d3 = xd[3];
                            da.select<8, 1>(0) = d0; da.select<8, 1>(8) = d1; db.select<8, 1>(0) = d2; db.select<8, 1>(8) = d3;
                            simd<float, 16> pa = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), qai, simd<int, 16>(x32.select<16, 1>(0))));
                            simd<float, 16> pb = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), qbi, simd<int, 16>(x32.select<16, 1>(16))));
                            accq += pa * sa * da + pb * sb * db;
                            simd<float, 8> s8 = scf.select<8, 1>(8 * hf);
                            corr += reduce<float>(s8 * xs, std::plus<>{});
                        }
                    }
                    sum = reduce<float>(accq, std::plus<>{}) - 32.0f * corr;
                } else { // Q5_K: qs 256 | qh 64 | scales 24 | d,dmin 2x2 half
                    simd<uint8_t, 32> meta = block_load<uint8_t, 32>(wd + 320);
                    simd<float, 4> dm = convert<float>(block_load<sycl::half, 4>((const sycl::half *) (wd + 344)));
                    simd<uint8_t, 64> qh = block_load<uint8_t, 64>(wd + 256); simd<uint32_t, 16> qh32 = qh.bit_cast_view<uint32_t>();
#pragma unroll
                    for (int b = 0; b < 2; b++) {
                        simd<float, 8> scf, mnf; k4_unpack(meta.select<12, 1>(12 * b), dm[2 * b], dm[2 * b + 1], scf, mnf);
                        simd<uint8_t, 128> qs = block_load<uint8_t, 128>(wd + 128 * b); simd<uint32_t, 32> q32 = qs.bit_cast_view<uint32_t>();
                        simd<uint32_t, 8> H = qh32.select<8, 1>(8 * b);
                        simd<int8_t, 256> x8 = block_load<int8_t, 256>(a.q + ev * 512 + 256 * b); simd<int, 64> x32 = x8.bit_cast_view<int>();
                        simd<float, 8> sd = scf * block_load<float, 8>(a.d + ev * 16 + 8 * b); simd<float, 16> xs = block_load<float, 16>(a.s + ev * 32 + 16 * b);
                        simd<float, 16> acc5 = 0.0f;
#pragma unroll
                        for (int g = 0; g < 4; g++) {
                            simd<uint32_t, 8> Q = q32.select<8, 1>(8 * g); simd<uint32_t, 16> qq;
                            qq.select<8, 1>(0) = (Q & 0x0F0F0F0Fu) | (((H >> (2 * g)) & 0x01010101u) << 4);
                            qq.select<8, 1>(8) = ((Q >> 4) & 0x0F0F0F0Fu) | (((H >> (2 * g + 1)) & 0x01010101u) << 4);
                            simd<int, 16> qi = qq.bit_cast_view<int>();
                            simd<float, 16> p = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), qi, simd<int, 16>(x32.select<16, 1>(16 * g))));
                            simd<float, 16> s; const float s0 = sd[2 * g], s1 = sd[2 * g + 1];
                            s.select<8, 1>(0) = s0; s.select<8, 1>(8) = s1; acc5 += p * s;
                        }
                        simd<float, 8> xs32 = xs.select<8, 2>(0) + xs.select<8, 2>(1);
                        sum += reduce<float>(acc5, std::plus<>{}) - reduce<float>(mnf * xs32, std::plus<>{});
                    }
                }
                y += wts[c * 8 + u] * sum;
            }
            Y2[c * 2048 + r] = y;
        }
    });
}

// ---------------------------------------------------------------- logits -> sampling candidates on the GPU (plain SYCL)
// per column a block of ints: [0] number of candidates found (may exceed the capacity), [1] maximum logit (float bits),
// then (index, value bits) pairs of every logit >= max - W, in no particular order.
constexpr int CAND_CAP = 8192, CAND_BLK = 2 + 2 * CAND_CAP;
static inline sycl::event k_cand(sycl::queue & q, const float * L, int n, int ld, int nc, float W, int * blk) {
    constexpr int per = 4; constexpr size_t wg = 256; const size_t ni = ((size_t) n + per - 1) / per, ng = (ni + wg - 1) / wg;
    q.parallel_for(sycl::range<1>(nc), [=](sycl::id<1> c) { blk[c * CAND_BLK] = 0; ((float *) blk)[c * CAND_BLK + 1] = -1e30f; });
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nc * ng * wg), sycl::range<1>(wg)), [=](sycl::nd_item<1> it) {
        const size_t g = it.get_group(0); const size_t c = g / ng; const size_t i0 = ((g % ng) * wg + it.get_local_id(0)) * per;
        const float * l = L + c * ld; float m = -1e30f;
        for (int k = 0; k < per; k++) if (i0 + k < (size_t) n) m = sycl::fmax(m, l[i0 + k]);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        if (it.get_local_id(0) == 0) sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(((float *) blk)[c * CAND_BLK + 1]).fetch_max(m);
    });
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>(nc * ng * wg), sycl::range<1>(wg)), [=](sycl::nd_item<1> it) {
        const size_t g = it.get_group(0); const size_t c = g / ng; const size_t i0 = ((g % ng) * wg + it.get_local_id(0)) * per;
        const float * l = L + c * ld; int * b = blk + c * CAND_BLK; const float thr = ((const float *) b)[1] - W;
        for (int k = 0; k < per; k++) {
            const size_t i = i0 + k; if (i >= (size_t) n) break; const float v = l[i];
            if (v >= thr) {
                const int s = sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(b[0]).fetch_add(1);
                if (s < CAND_CAP) { b[2 + 2 * s] = (int) i; ((float *) b)[3 + 2 * s] = v; }
            }
        }
    });
}

} // namespace q36g
