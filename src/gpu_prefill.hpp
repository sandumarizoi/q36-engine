// q36 — kernels of the layer-major prefill: whole chunks of positions per kernel, the MoE expert by expert.
// The arithmetic (and its order) is the same as in the decode kernels, only the work is cut differently:
//   * dense mat-vecs: one thread per (row, 8 columns), threads of one row are neighbours -> the row stays in the cache
//   * delta-net state: one thread per (head, row) walks through all positions of the chunk with the state in registers
//   * routed experts: (token, slot) pairs are grouped by expert on the host; one thread unpacks a weight row once
//     and applies it to up to 8 tokens
#pragma once
#include "gpu_kernels.hpp"
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

namespace q36g {

static inline Act act_at(Act a, size_t c, int n) { return Act{ a.q + c * n, a.d + c * (n / 32), a.s + c * (n / 16) }; }

// ---------------------------------------------------------------- Q8_0 row-SoA mat-vec, thread = (row, group of 8 columns)
// MODE 0: y[col][r] = W[r].act[col]      MODE 1: y[col][r] = (W[r].act[col]) * colsc[col]
template <int MODE>
static inline sycl::event p_gemv_q8(sycl::queue & q, const uint8_t * W, int rows, int n, Act a, float * y, int ng, int ystride, const float * colsc = nullptr) {
    const int rb = n + n / 16; const int nb = n / 32;
    return q.parallel_for(sycl::range<1>((size_t) rows * ng), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int g = (int) (id[0] % ng), r = (int) (id[0] / ng); const size_t c8 = (size_t) g * 8;
        const uint8_t * w = W + (size_t) r * rb; const sycl::half * wd = (const sycl::half *) (w + n);
        simd<float, 16> acc[8];
#pragma unroll
        for (int c = 0; c < 8; c++) acc[c] = 0.0f;
        for (int c0 = 0; c0 < n; c0 += 256) {
            simd<int8_t, 256> w8 = block_load<int8_t, 256>((const int8_t *) w + c0);
            simd<int, 64> w32 = w8.template bit_cast_view<int>();
            simd<float, 8> wsc = convert<float>(block_load<sycl::half, 8>(wd + c0 / 32));
#pragma unroll
            for (int c = 0; c < 8; c++) {
                simd<int8_t, 256> x8 = block_load<int8_t, 256>(a.q + (c8 + c) * n + c0);
                simd<int, 64> x32 = x8.template bit_cast_view<int>();
                simd<float, 8> sc = wsc * block_load<float, 8>(a.d + (c8 + c) * nb + c0 / 32);
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
#pragma unroll
        for (int c = 0; c < 8; c++) {
            float v = reduce<float>(acc[c], std::plus<>{});
            if constexpr (MODE == 1) v *= colsc[c8 + c];
            y[(c8 + c) * ystride + r] = v;
        }
    });
}

// ---------------------------------------------------------------- f32 matrix product (router, alpha/beta): thread = 4 rows x 8 tokens
// y[tok][r] = W[r] . x[tok]; ntok % 8 == 0; rows need not be a multiple of 4
static inline sycl::event p_gemv_f32(sycl::queue & q, const float * W, int rows, int n, const float * x, float * y, int ng, int ystride) {
    const int nr = (rows + 3) / 4;
    return q.parallel_for(sycl::range<1>((size_t) nr * ng), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int g = (int) (id[0] % ng), r0 = (int) (id[0] / ng) * 4; const size_t c8 = (size_t) g * 8;
        const float * w[4];
#pragma unroll
        for (int r = 0; r < 4; r++) w[r] = W + (size_t) std::min(r0 + r, rows - 1) * n;
        const float * xb = x + c8 * n;
        simd<float, 16> acc[4][8];
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
            for (int c = 0; c < 8; c++) acc[r][c] = 0.0f;
        for (int i = 0; i < n; i += 16) {
            simd<float, 16> wv[4];
#pragma unroll
            for (int r = 0; r < 4; r++) wv[r] = block_load<float, 16>(w[r] + i);
#pragma unroll
            for (int c = 0; c < 8; c++) {
                simd<float, 16> xv = block_load<float, 16>(xb + (size_t) c * n + i);
#pragma unroll
                for (int r = 0; r < 4; r++) acc[r][c] += wv[r] * xv;
            }
        }
#pragma unroll
        for (int r = 0; r < 4; r++) {
            if (r0 + r >= rows) break;
#pragma unroll
            for (int c = 0; c < 8; c++) y[(c8 + c) * ystride + r0 + r] = reduce<float>(acc[r][c], std::plus<>{});
        }
    });
}

// ---------------------------------------------------------------- GDN on a chunk of n positions (plain buffers, row = position in the chunk)
static inline sycl::event p_gdn_gates(sycl::queue & q, int n, const float * AB, const float * par_dt, const float * par_a, float * G) {
    return q.parallel_for(sycl::range<1>((size_t) n * 2), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const size_t c = id[0] / 2; const int hf = (int) (id[0] % 2); float * g = G + c * 64;
        simd<float, 16> al = block_load<float, 16>(AB + c * 64 + hf * 16) + block_load<float, 16>(par_dt + hf * 16);
        simd<float, 16> sp = es::log(es::exp(al) + 1.0f);
        sp.merge(al, al > 20.0f);
        block_store<float, 16>(g + hf * 16, es::exp(sp * block_load<float, 16>(par_a + hf * 16)));
        block_store<float, 16>(g + 32 + hf * 16, sigmoid16(block_load<float, 16>(AB + c * 64 + 32 + hf * 16)));
    });
}

// YR has 3 rows of history in front: the conv of chunk position c reads rows c .. c+3
static inline sycl::event p_gdn_conv(sycl::queue & q, int n, const float * YR, const float * cw, float eps, float * QKV) {
    return q.parallel_for(sycl::range<1>((size_t) n * 64), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const size_t c = id[0] / 64; const int g = (int) (id[0] % 64); const int ch0 = g * 128;
        simd<float, 16> v[8]; simd<float, 16> ss = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            simd<float, 16> a = 0.0f;
#pragma unroll
            for (int t = 0; t < 4; t++) {
                const float * src = YR + (c + t) * 12288 + ch0 + 16 * j;
                a += block_load<float, 16>(src) * block_load<float, 16>(cw + t * 8192 + ch0 + 16 * j);
            }
            v[j] = silu16(a); ss += v[j] * v[j];
        }
        float sc = 1.0f;
        if (g < 32) sc = 1.0f / es::sqrt_ieee<float>(reduce<float>(ss, std::plus<>{}) + eps);
        float * dst = QKV + c * 8192 + ch0;
#pragma unroll
        for (int j = 0; j < 8; j++) block_store<float, 16>(dst + 16 * j, v[j] * sc);
    });
}

// S holds the state before the chunk and is advanced over its n positions
static inline sycl::event p_gdn_core(sycl::queue & q, int n, float * S, const float * QKV, const float * G, float * O) {
    return q.parallel_for(sycl::range<1>(32 * 128), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int h = (int) id[0] / 128, j = (int) id[0] % 128; const int kh = h % 16;
        float * sp = S + ((size_t) h * 128 + j) * 128;
        simd<float, 16> s[8];
#pragma unroll
        for (int i = 0; i < 8; i++) s[i] = block_load<float, 16>(sp + 16 * i);
        for (int c = 0; c < n; c++) {
            const float g = G[(size_t) c * 64 + h], beta = G[(size_t) c * 64 + 32 + h];
            const float * qv = QKV + (size_t) c * 8192 + kh * 128; const float * k = qv + 2048;
            const float vj = QKV[(size_t) c * 8192 + 4096 + h * 128 + j];
            simd<float, 16> kk[8]; simd<float, 16> a = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) { kk[i] = block_load<float, 16>(k + 16 * i); a += s[i] * kk[i]; }
            const float dl = (vj - g * reduce<float>(a, std::plus<>{})) * beta;
            simd<float, 16> o = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) { s[i] = s[i] * g + kk[i] * dl; o += s[i] * block_load<float, 16>(qv + 16 * i); }
            O[(size_t) c * 4096 + h * 128 + j] = reduce<float>(o, std::plus<>{}) * 0.08838834764831845f;
        }
#pragma unroll
        for (int i = 0; i < 8; i++) block_store<float, 16>(sp + 16 * i, s[i]);
    });
}

// YRz = the chunk rows of YR (without the 3 history rows)
static inline sycl::event p_gdn_post(sycl::queue & q, int n, const float * O, const float * YRz, const float * nw, float eps, Act a) {
    return q.parallel_for(sycl::range<1>((size_t) n * 32), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const size_t c = id[0] / 32; const int h = (int) (id[0] % 32); const float * o = O + c * 4096 + h * 128;
        const float * z = YRz + c * 12288 + 8192 + h * 128;
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

// ---------------------------------------------------------------- shared expert: GU[c][1024] = interleaved (gate_i, up_i) -> SwiGLU -> act(512)
static inline sycl::event p_swiglu_quant(sycl::queue & q, int n, const float * GU, Act a) {
    return q.parallel_for(sycl::range<1>((size_t) n * 16), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const size_t c = id[0] / 16; const int b = (int) (id[0] % 16); const float * src = GU + c * 1024 + 64 * b;
        simd<float, 32> t0 = block_load<float, 32>(src), t1 = block_load<float, 32>(src + 32);
        simd<float, 16> g0 = t0.select<16, 2>(0), u0 = t0.select<16, 2>(1), g1 = t1.select<16, 2>(0), u1 = t1.select<16, 2>(1);
        quant32(silu16(g0) * u0, silu16(g1) * u1, a.q + c * 512 + 32 * b, a.d + c * 16 + b, a.s + c * 32 + 2 * b);
    });
}

// ---------------------------------------------------------------- routed experts, expert-major
// GRP[group][16] = { expert, n (1..8), pair[8] } with pair = token * 8 + slot. TAB[256] = device address of the expert records.
static ESIMD_INLINE void q4_split(simd<uint32_t, 32> q32, simd<int, 16> * w) {
#pragma unroll
    for (int g = 0; g < 4; g++) {
        simd<uint32_t, 8> Q = q32.select<8, 1>(8 * g); simd<uint32_t, 16> qq;
        qq.select<8, 1>(0) = Q & 0x0F0F0F0Fu; qq.select<8, 1>(8) = (Q >> 4) & 0x0F0F0F0Fu;
        w[g] = qq.bit_cast_view<int>();
    }
}
static ESIMD_INLINE float q4k_dot(const simd<int, 16> * w, simd<float, 8> scale_f, simd<float, 8> min_f, simd<int, 64> x32, simd<float, 8> xd, simd<float, 16> xs) {
    simd<float, 8> sd = scale_f * xd; simd<float, 16> acc = 0.0f;
#pragma unroll
    for (int g = 0; g < 4; g++) {
        simd<float, 16> p = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), w[g], simd<int, 16>(x32.select<16, 1>(16 * g))));
        simd<float, 16> s; const float s0 = sd[2 * g], s1 = sd[2 * g + 1];
        s.select<8, 1>(0) = s0; s.select<8, 1>(8) = s1; acc += p * s;
    }
    simd<float, 8> xs32 = xs.select<8, 2>(0) + xs.select<8, 2>(1);
    return reduce<float>(acc, std::plus<>{}) - reduce<float>(min_f * xs32, std::plus<>{});
}

// gate/up + SwiGLU: thread = (group, hidden unit) -> ACTE[pair][512]. a = post-norm activations of the window (row = token)
static inline sycl::event p_exp_glu(sycl::queue & q, int ng, const int * GRP, const uint64_t * TAB, Act a, float * ACTE) {
    return q.parallel_for(sycl::range<1>((size_t) ng * 512), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int i = (int) (id[0] % 512); const int * grp = GRP + (id[0] / 512) * 16; const int n = grp[1];
        const uint8_t * rec = (const uint8_t *) TAB[grp[0]];
        const uint8_t * wg = rec + (size_t) (2 * i) * 1152; const uint8_t * wu = wg + 1152;
        simd<uint8_t, 128> mg = block_load<uint8_t, 128>(wg + 1024), mu = block_load<uint8_t, 128>(wu + 1024);
        simd<sycl::half, 16> dmg = block_load<sycl::half, 16>((const sycl::half *) (wg + 1120)), dmu = block_load<sycl::half, 16>((const sycl::half *) (wu + 1120));
        simd<float, 16> fg = convert<float>(dmg), fu = convert<float>(dmu);
        float sg[8], su[8];
#pragma unroll
        for (int t = 0; t < 8; t++) { sg[t] = 0.0f; su[t] = 0.0f; }
        for (int b = 0; b < 8; b++) {
            simd<float, 8> scg, mng, scu, mnu; simd<int, 16> qg[4], qu[4];
            k4_unpack(mg.select<12, 1>(12 * b), fg[2 * b], fg[2 * b + 1], scg, mng);
            k4_unpack(mu.select<12, 1>(12 * b), fu[2 * b], fu[2 * b + 1], scu, mnu);
            simd<uint8_t, 128> bg = block_load<uint8_t, 128>(wg + 128 * b), bu = block_load<uint8_t, 128>(wu + 128 * b);
            q4_split(bg.bit_cast_view<uint32_t>(), qg); q4_split(bu.bit_cast_view<uint32_t>(), qu);
            for (int t = 0; t < n; t++) {
                const size_t tok = (size_t) (grp[2 + t] >> 3);
                simd<int8_t, 256> x8 = block_load<int8_t, 256>(a.q + tok * 2048 + 256 * b); simd<int, 64> x32 = x8.bit_cast_view<int>();
                simd<float, 8> xd = block_load<float, 8>(a.d + tok * 64 + 8 * b); simd<float, 16> xs = block_load<float, 16>(a.s + tok * 128 + 16 * b);
                sg[t] += q4k_dot(qg, scg, mng, x32, xd, xs);
                su[t] += q4k_dot(qu, scu, mnu, x32, xd, xs);
            }
        }
        for (int t = 0; t < n; t++) {
            simd<float, 1> g(sg[t]); simd<float, 1> sl = g / (es::exp(-g) + 1.0f);
            ACTE[(size_t) grp[2 + t] * 512 + i] = sl[0] * su[t];
        }
    });
}

// down projection: thread = (group, 4 output rows). Y2[token][r] += w[pair] * down[r].act[pair]
// All pairs of one call must belong to different tokens (the caller runs one call per slot), so the rows of Y2 do not collide.
template <bool Q6>
static inline sycl::event p_exp_down(sycl::queue & q, int ng, const int * GRP, const uint64_t * TAB, Act a, const float * wts, float * Y2) {
    const size_t down_off = (size_t) 1024 * 1152; const int rd = Q6 ? 432 : 352;
    return q.parallel_for(sycl::range<1>((size_t) ng * 512), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int r4 = (int) (id[0] % 512) * 4; const int * grp = GRP + (id[0] / 512) * 16; const int n = grp[1];
        const uint8_t * rec = (const uint8_t *) TAB[grp[0]];
        for (int rr = 0; rr < 4; rr++) {
            const int r = r4 + rr; const uint8_t * wd = rec + down_off + (size_t) r * rd;
            if constexpr (Q6) { // ql 256 | qh 128 | scales 32 | d 2x half
                simd<float, 2> dv = convert<float>(block_load<sycl::half, 2>((const sycl::half *) (wd + 416)));
                simd<int, 16> qa[4], qb[4]; simd<float, 16> sa[4], sb[4]; simd<float, 8> s8[4];
#pragma unroll
                for (int b = 0; b < 2; b++) {
                    const float d = dv[b];
                    simd<uint8_t, 128> ql = block_load<uint8_t, 128>(wd + 128 * b); simd<uint8_t, 64> qh = block_load<uint8_t, 64>(wd + 256 + 64 * b);
                    simd<float, 16> scf = convert<float>(block_load<int8_t, 16>((const int8_t *) wd + 384 + 16 * b)) * d;
                    simd<uint32_t, 32> ql32 = ql.bit_cast_view<uint32_t>(); simd<uint32_t, 16> qh32 = qh.bit_cast_view<uint32_t>();
#pragma unroll
                    for (int hf = 0; hf < 2; hf++) {
                        simd<uint32_t, 8> A = ql32.select<8, 1>(16 * hf), B = ql32.select<8, 1>(16 * hf + 8), H = qh32.select<8, 1>(8 * hf);
                        simd<uint32_t, 16> ua, ub;
                        ua.select<8, 1>(0) = (A & 0x0F0F0F0Fu) | ((H & 0x03030303u) << 4);
                        ua.select<8, 1>(8) = (B & 0x0F0F0F0Fu) | (((H >> 2) & 0x03030303u) << 4);
                        ub.select<8, 1>(0) = ((A >> 4) & 0x0F0F0F0Fu) | (((H >> 4) & 0x03030303u) << 4);
                        ub.select<8, 1>(8) = ((B >> 4) & 0x0F0F0F0Fu) | (((H >> 6) & 0x03030303u) << 4);
                        const int k2 = 2 * b + hf;
                        qa[k2] = ua.bit_cast_view<int>(); qb[k2] = ub.bit_cast_view<int>();
#pragma unroll
                        for (int k = 0; k < 4; k++) { const float f0 = scf[8 * hf + k], f1 = scf[8 * hf + 4 + k]; sa[k2].select<4, 1>(4 * k) = f0; sb[k2].select<4, 1>(4 * k) = f1; }
                        s8[k2] = scf.select<8, 1>(8 * hf);
                    }
                }
                for (int t = 0; t < n; t++) {
                    const size_t ev = (size_t) grp[2 + t]; simd<float, 16> accq = 0.0f; float corr = 0.0f;
#pragma unroll
                    for (int k2 = 0; k2 < 4; k2++) {
                        const int x0 = 128 * k2;
                        simd<int8_t, 128> x8 = block_load<int8_t, 128>(a.q + ev * 512 + x0); simd<int, 32> x32 = x8.bit_cast_view<int>();
                        simd<float, 4> xd = block_load<float, 4>(a.d + ev * 16 + x0 / 32); simd<float, 8> xs = block_load<float, 8>(a.s + ev * 32 + x0 / 16);
                        simd<float, 16> da, db; const float d0 = xd[0], d1 = xd[1], d2 = xd[2], d3 = xd[3];
                        da.select<8, 1>(0) = d0; da.select<8, 1>(8) = d1; db.select<8, 1>(0) = d2; db.select<8, 1>(8) = d3;
                        simd<float, 16> pa = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), qa[k2], simd<int, 16>(x32.select<16, 1>(0))));
                        simd<float, 16> pb = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), qb[k2], simd<int, 16>(x32.select<16, 1>(16))));
                        accq += pa * sa[k2] * da + pb * sb[k2] * db;
                        corr += reduce<float>(s8[k2] * xs, std::plus<>{});
                    }
                    const float sum = reduce<float>(accq, std::plus<>{}) - 32.0f * corr;
                    float * y = Y2 + (ev >> 3) * 2048 + r; *y = *y + wts[ev] * sum;
                }
            } else { // Q5_K: qs 256 | qh 64 | scales 24 | d,dmin 2x2 half
                simd<uint8_t, 32> meta = block_load<uint8_t, 32>(wd + 320);
                simd<float, 4> dm = convert<float>(block_load<sycl::half, 4>((const sycl::half *) (wd + 344)));
                simd<uint8_t, 64> qh = block_load<uint8_t, 64>(wd + 256); simd<uint32_t, 16> qh32 = qh.bit_cast_view<uint32_t>();
                simd<int, 16> wq[8]; simd<float, 8> scf[2], mnf[2];
#pragma unroll
                for (int b = 0; b < 2; b++) {
                    k4_unpack(meta.select<12, 1>(12 * b), dm[2 * b], dm[2 * b + 1], scf[b], mnf[b]);
                    simd<uint8_t, 128> qs = block_load<uint8_t, 128>(wd + 128 * b); simd<uint32_t, 32> q32 = qs.bit_cast_view<uint32_t>();
                    simd<uint32_t, 8> H = qh32.select<8, 1>(8 * b);
#pragma unroll
                    for (int g = 0; g < 4; g++) {
                        simd<uint32_t, 8> Q = q32.select<8, 1>(8 * g); simd<uint32_t, 16> qq;
                        qq.select<8, 1>(0) = (Q & 0x0F0F0F0Fu) | (((H >> (2 * g)) & 0x01010101u) << 4);
                        qq.select<8, 1>(8) = ((Q >> 4) & 0x0F0F0F0Fu) | (((H >> (2 * g + 1)) & 0x01010101u) << 4);
                        wq[4 * b + g] = qq.bit_cast_view<int>();
                    }
                }
                for (int t = 0; t < n; t++) {
                    const size_t ev = (size_t) grp[2 + t]; float sum = 0.0f;
#pragma unroll
                    for (int b = 0; b < 2; b++) {
                        simd<int8_t, 256> x8 = block_load<int8_t, 256>(a.q + ev * 512 + 256 * b); simd<int, 64> x32 = x8.bit_cast_view<int>();
                        simd<float, 8> sd = scf[b] * block_load<float, 8>(a.d + ev * 16 + 8 * b); simd<float, 16> xs = block_load<float, 16>(a.s + ev * 32 + 16 * b);
                        simd<float, 16> acc5 = 0.0f;
#pragma unroll
                        for (int g = 0; g < 4; g++) {
                            simd<float, 16> p = convert<float>(dp4a<int, int, int, int, 16>(simd<int, 16>(0), wq[4 * b + g], simd<int, 16>(x32.select<16, 1>(16 * g))));
                            simd<float, 16> s; const float s0 = sd[2 * g], s1 = sd[2 * g + 1];
                            s.select<8, 1>(0) = s0; s.select<8, 1>(8) = s1; acc5 += p * s;
                        }
                        simd<float, 8> xs32 = xs.select<8, 2>(0) + xs.select<8, 2>(1);
                        sum += reduce<float>(acc5, std::plus<>{}) - reduce<float>(mnf[b] * xs32, std::plus<>{});
                    }
                    float * y = Y2 + (ev >> 3) * 2048 + r; *y = *y + wts[ev] * sum;
                }
            }
        }
    });
}


// ================================================================ XMX (dpas) matrix products for the prefill
// Activations are re-laid out in "VNNI tiles" first: tile (token group g of 16, block kb of 32) = 512 bytes where dword
// k4 * 16 + n = bytes 4 k4 .. 4 k4 + 3 of block kb of token 16 g + n, plus the 16 block scales XD[g][kb][16].
// Then dpas(B = tile, A = 8 weight rows x 32 int8) gives int32 dot products of 8 rows x 16 tokens for one Q8_0 block.
struct Tiles { int8_t * q; float * d; float * s = nullptr; }; // s: sums of the 32 dequantized values (K-quant mins)

static inline sycl::event p_act_tiles(sycl::queue & q, int ntok, int n, Act a, Tiles t) {
    const int nb = n / 32, ng = (ntok + 15) / 16;
    return q.parallel_for(sycl::range<1>((size_t) ng * nb), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int kb = (int) (id[0] % nb), g = (int) (id[0] / nb);
        simd<uint32_t, 128> out = 0u; simd<float, 16> xd = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const int tok = g * 16 + i;
            if (tok < ntok) {
                simd<uint32_t, 8> r = block_load<uint32_t, 8>((const uint32_t *) (a.q + (size_t) tok * n + kb * 32));
                out.select<8, 16>(i) = r; xd[i] = a.d[(size_t) tok * nb + kb];
            }
        }
        uint32_t * dst = (uint32_t *) (t.q + ((size_t) g * nb + kb) * 512);
        block_store<uint32_t, 64>(dst, out.select<64, 1>(0)); block_store<uint32_t, 64>(dst + 64, out.select<64, 1>(64));
        block_store<float, 16>(t.d + ((size_t) g * nb + kb) * 16, xd);
    });
}

// y[tok][r] = W[r] . x[tok] (MODE 1: times colsc[tok]); W = Q8_0 row-SoA (n int8 + n/32 half per row), rows % 16 == 0.
// thread = 16 rows x 32 tokens
template <int MODE>
static inline sycl::event p_mm_q8(sycl::queue & q, const uint8_t * W, int rows, int n, Tiles t, int ntok, float * y, int ystride, const float * colsc = nullptr) {
    const int rb = n + n / 16, nb = n / 32, ng = (ntok + 31) / 32, nr = rows / 16;
    return q.parallel_for(sycl::range<1>((size_t) nr * ng), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int g = (int) (id[0] % ng), r0 = (int) (id[0] / ng) * 16;
        simd<float, 128> acc[2][2];
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
            for (int h = 0; h < 2; h++) acc[r][h] = 0.0f;
        const uint8_t * w0 = W + (size_t) r0 * rb;
        for (int kb = 0; kb < nb; kb++) {
            simd<int8_t, 256> A[2]; simd<float, 16> ws;
#pragma unroll
            for (int r = 0; r < 2; r++)
#pragma unroll
                for (int m = 0; m < 8; m++) {
                    const uint8_t * wr = w0 + (size_t) (8 * r + m) * rb;
                    A[r].select<32, 1>(32 * m) = block_load<int8_t, 32>((const int8_t *) wr + kb * 32);
                    const float hv = (float) ((const sycl::half *) (wr + n))[kb]; ws[8 * r + m] = hv;
                }
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const size_t tile = (size_t) (2 * g + h) * nb + kb;
                simd<int8_t, 512> B;
                B.select<256, 1>(0) = block_load<int8_t, 256>(t.q + tile * 512); B.select<256, 1>(256) = block_load<int8_t, 256>(t.q + tile * 512 + 256);
                simd<float, 16> xd = block_load<float, 16>(t.d + tile * 16);
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    simd<int, 128> res = xmx::dpas<8, 8, int>(B, A[r]);
                    simd<float, 128> sc;
#pragma unroll
                    for (int m = 0; m < 8; m++) { const float wv = ws[8 * r + m]; sc.select<16, 1>(16 * m) = xd * wv; }
                    acc[r][h] += convert<float>(res) * sc;
                }
            }
        }
        simd<uint32_t, 16> lane(0u, 1u);
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int tok0 = 32 * g + 16 * h; if (tok0 >= ntok) break;
            simd_mask<16> ok = (lane + (uint32_t) tok0) < (uint32_t) ntok;
            simd<float, 16> cs = 1.0f;
            if constexpr (MODE == 1) { simd<uint32_t, 16> ci = (lane + (uint32_t) tok0); ci.merge(simd<uint32_t, 16>((uint32_t) tok0), !ok); cs = gather<float, 16>(colsc, ci * 4u); }
            simd<uint32_t, 16> off = (lane + (uint32_t) tok0) * (uint32_t) (ystride * 4);
#pragma unroll
            for (int r = 0; r < 2; r++)
#pragma unroll
                for (int m = 0; m < 8; m++) {
                    simd<float, 16> v = acc[r][h].select<16, 1>(16 * m) * cs;
                    scatter<float, 16>(y + r0 + 8 * r + m, off, v, ok);
                }
        }
    });
}

// ================================================================ routed experts with XMX
// GX[group][32] = { expert, n (1..16), pair[16] }. Tiles of a group: the activations of its pairs (row = pair >> shift).
static inline sycl::event p_grp_tiles(sycl::queue & q, int ngrp, const int * GX, int n, Act a, int shift, Tiles t) {
    const int nb = n / 32;
    return q.parallel_for(sycl::range<1>((size_t) ngrp * nb), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int kb = (int) (id[0] % nb); const size_t gi = id[0] / nb; if (gi >= (size_t) GX[0]) return; const int * grp = GX + 32 + gi * 32; const int cnt = grp[1];
        simd<uint32_t, 128> out = 0u; simd<float, 16> xd = 0.0f, xs = 0.0f, xh = 0.0f;
        for (int i = 0; i < cnt; i++) {
            const size_t row = (size_t) (grp[2 + i] >> shift);
            simd<uint32_t, 8> r = block_load<uint32_t, 8>((const uint32_t *) (a.q + row * n + kb * 32));
            out.select<8, 16>(i) = r; xd[i] = a.d[row * nb + kb]; xs[i] = a.s[row * 2 * nb + 2 * kb]; xh[i] = a.s[row * 2 * nb + 2 * kb + 1];
        }
        const size_t tile = gi * nb + kb; uint32_t * dst = (uint32_t *) (t.q + tile * 512);
        block_store<uint32_t, 64>(dst, out.select<64, 1>(0)); block_store<uint32_t, 64>(dst + 64, out.select<64, 1>(64));
        block_store<float, 16>(t.d + tile * 16, xd); block_store<float, 16>(t.s + tile * 32, xs); block_store<float, 16>(t.s + tile * 32 + 16, xh);
    });
}

// gate/up (Q4_K, rows interleaved [gate_i | up_i], 1152 B each) + SwiGLU: thread = (group, 16 rows = 8 hidden units) -> ACTE[pair][512]
static inline sycl::event p_exp_glu_x(sycl::queue & q, int ngrp, const int * GX, const uint64_t * TAB, Tiles t, float * ACTE) {
    return q.parallel_for(sycl::range<1>((size_t) ngrp * 64), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int rbk = (int) (id[0] % 64); const size_t gi = id[0] / 64; if (gi >= (size_t) GX[0]) return; const int * grp = GX + 32 + gi * 32; const int cnt = grp[1];
        const uint8_t * rec = (const uint8_t *) TAB[grp[0]] + (size_t) (16 * rbk) * 1152;
        simd<float, 128> acc[2]; acc[0] = 0.0f; acc[1] = 0.0f;
        for (int b = 0; b < 8; b++) {
            simd<float, 128> SC, MN; // [row][subblock]
#pragma unroll
            for (int rr = 0; rr < 16; rr++) {
                const uint8_t * w = rec + (size_t) rr * 1152;
                simd<uint8_t, 16> sc16 = block_load<uint8_t, 16>(w + 1024 + 12 * b); simd<uint8_t, 12> sc12 = sc16.select<12, 1>(0);
                simd<float, 2> dm = convert<float>(block_load<sycl::half, 2>((const sycl::half *) (w + 1120) + 2 * b));
                simd<float, 8> scf, mnf; k4_unpack(sc12, dm[0], dm[1], scf, mnf);
                SC.select<8, 1>(8 * rr) = scf; MN.select<8, 1>(8 * rr) = mnf;
            }
#pragma unroll
            for (int g = 0; g < 4; g++) {
                simd<uint32_t, 64> raw[2];
#pragma unroll
                for (int r = 0; r < 2; r++)
#pragma unroll
                    for (int m = 0; m < 8; m++) raw[r].select<8, 1>(8 * m) = block_load<uint32_t, 8>((const uint32_t *) (rec + (size_t) (8 * r + m) * 1152 + 128 * b + 32 * g));
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int sb = 2 * g + h; const size_t tile = gi * 64 + 8 * b + sb;
                    simd<int8_t, 512> B;
                    B.select<256, 1>(0) = block_load<int8_t, 256>(t.q + tile * 512); B.select<256, 1>(256) = block_load<int8_t, 256>(t.q + tile * 512 + 256);
                    simd<float, 16> xd = block_load<float, 16>(t.d + tile * 16), xs = block_load<float, 16>(t.s + tile * 32) + block_load<float, 16>(t.s + tile * 32 + 16);
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        simd<uint32_t, 64> av = (raw[r] >> (4 * h)) & 0x0F0F0F0Fu;
                        simd<int8_t, 256> A = av.bit_cast_view<int8_t>();
                        simd<int, 128> res = xmx::dpas<8, 8, int>(B, A);
#pragma unroll
                        for (int m = 0; m < 8; m++) {
                            const float sc = SC[(8 * r + m) * 8 + sb], mn = MN[(8 * r + m) * 8 + sb];
                            acc[r].select<16, 1>(16 * m) += convert<float>(simd<int, 16>(res.select<16, 1>(16 * m))) * (xd * sc) - xs * mn;
                        }
                    }
                }
            }
        }
        simd<uint32_t, 16> lane(0u, 1u); simd_mask<16> ok = lane < (uint32_t) cnt;
        simd<uint32_t, 16> pr = 0u;
        for (int i = 0; i < cnt; i++) pr[i] = (uint32_t) grp[2 + i];
        simd<uint32_t, 16> off = pr * (512u * 4u);
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
            for (int k = 0; k < 4; k++) {
                simd<float, 16> gt = acc[r].select<16, 1>(32 * k), up = acc[r].select<16, 1>(32 * k + 16);
                scatter<float, 16>(ACTE + 8 * rbk + 4 * r + k, off, silu16(gt) * up, ok);
            }
    });
}

// down (Q5_K, 352 B rows): thread = (group, 16 rows) -> YP[pair][2048] = w[pair] * down . act[pair]
static inline sycl::event p_exp_down_x(sycl::queue & q, int ngrp, const int * GX, const uint64_t * TAB, Tiles t, const float * wts, float * YP) {
    const size_t down_off = (size_t) 1024 * 1152;
    return q.parallel_for(sycl::range<1>((size_t) ngrp * 128), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int rbk = (int) (id[0] % 128); const size_t gi = id[0] / 128; if (gi >= (size_t) GX[0]) return; const int * grp = GX + 32 + gi * 32; const int cnt = grp[1];
        const uint8_t * rec = (const uint8_t *) TAB[grp[0]] + down_off + (size_t) (16 * rbk) * 352;
        simd<float, 128> acc[2]; acc[0] = 0.0f; acc[1] = 0.0f;
#pragma unroll
        for (int b = 0; b < 2; b++) {
            simd<float, 128> SC, MN; simd<uint32_t, 64> qh[2];
#pragma unroll
            for (int rr = 0; rr < 16; rr++) {
                const uint8_t * w = rec + (size_t) rr * 352;
                simd<uint8_t, 16> sc16 = block_load<uint8_t, 16>(w + 320 + 12 * b); simd<uint8_t, 12> sc12 = sc16.select<12, 1>(0);
                simd<float, 2> dm = convert<float>(block_load<sycl::half, 2>((const sycl::half *) (w + 344) + 2 * b));
                simd<float, 8> scf, mnf; k4_unpack(sc12, dm[0], dm[1], scf, mnf);
                SC.select<8, 1>(8 * rr) = scf; MN.select<8, 1>(8 * rr) = mnf;
                qh[rr / 8].select<8, 1>(8 * (rr % 8)) = block_load<uint32_t, 8>((const uint32_t *) (w + 256 + 32 * b));
            }
#pragma unroll
            for (int g = 0; g < 4; g++) {
                simd<uint32_t, 64> raw[2];
#pragma unroll
                for (int r = 0; r < 2; r++)
#pragma unroll
                    for (int m = 0; m < 8; m++) raw[r].select<8, 1>(8 * m) = block_load<uint32_t, 8>((const uint32_t *) (rec + (size_t) (8 * r + m) * 352 + 128 * b + 32 * g));
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int sb = 2 * g + h; const size_t tile = gi * 16 + 8 * b + sb;
                    simd<int8_t, 512> B;
                    B.select<256, 1>(0) = block_load<int8_t, 256>(t.q + tile * 512); B.select<256, 1>(256) = block_load<int8_t, 256>(t.q + tile * 512 + 256);
                    simd<float, 16> xd = block_load<float, 16>(t.d + tile * 16), xs = block_load<float, 16>(t.s + tile * 32) + block_load<float, 16>(t.s + tile * 32 + 16);
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        simd<uint32_t, 64> av = ((raw[r] >> (4 * h)) & 0x0F0F0F0Fu) | (((qh[r] >> sb) & 0x01010101u) << 4);
                        simd<int8_t, 256> A = av.bit_cast_view<int8_t>();
                        simd<int, 128> res = xmx::dpas<8, 8, int>(B, A);
#pragma unroll
                        for (int m = 0; m < 8; m++) {
                            const float sc = SC[(8 * r + m) * 8 + sb], mn = MN[(8 * r + m) * 8 + sb];
                            acc[r].select<16, 1>(16 * m) += convert<float>(simd<int, 16>(res.select<16, 1>(16 * m))) * (xd * sc) - xs * mn;
                        }
                    }
                }
            }
        }
        simd<uint32_t, 16> lane(0u, 1u); simd_mask<16> ok = lane < (uint32_t) cnt;
        simd<uint32_t, 16> pr = 0u; simd<float, 16> wv = 0.0f;
        for (int i = 0; i < cnt; i++) { pr[i] = (uint32_t) grp[2 + i]; wv[i] = wts[grp[2 + i]]; }
        simd<uint32_t, 16> off = pr * (2048u * 4u);
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
            for (int m = 0; m < 8; m++) scatter<float, 16>(YP + 16 * rbk + 8 * r + m, off, acc[r].select<16, 1>(16 * m) * wv, ok);
    });
}

// Y2[tok] += sum over the 8 slots of YP[tok * 8 + u] (slot order, like the decode kernel)
static inline sycl::event p_moe_reduce(sycl::queue & q, int ntok, const float * YP, float * Y2) {
    return q.parallel_for(sycl::range<1>((size_t) ntok * 128), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const size_t tok = id[0] / 128; const int i = (int) (id[0] % 128) * 16; float * y = Y2 + tok * 2048 + i;
        simd<float, 16> v = block_load<float, 16>(y);
#pragma unroll
        for (int u = 0; u < 8; u++) v += block_load<float, 16>(YP + (tok * 8 + u) * 2048 + i);
        block_store<float, 16>(y, v);
    });
}

// down (Q6_K, 432 B rows: ql 256 | qh 128 | 16 int8 scales per 256 | d per 256): scales per 16 values -> two dpas per block of 32
// value = d * sc16 * (q - 32), q in 0..63
static inline sycl::event p_exp_down6_x(sycl::queue & q, int ngrp, const int * GX, const uint64_t * TAB, Tiles t, const float * wts, float * YP) {
    const size_t down_off = (size_t) 1024 * 1152;
    return q.parallel_for(sycl::range<1>((size_t) ngrp * 128), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int rbk = (int) (id[0] % 128); const size_t gi = id[0] / 128; if (gi >= (size_t) GX[0]) return; const int * grp = GX + 32 + gi * 32; const int cnt = grp[1];
        const uint8_t * rec = (const uint8_t *) TAB[grp[0]] + down_off + (size_t) (16 * rbk) * 432;
        simd<float, 128> acc[2]; acc[0] = 0.0f; acc[1] = 0.0f;
        simd<uint32_t, 64> lomask; // dwords 0..3 of every row of 8
#pragma unroll
        for (int m = 0; m < 8; m++) { lomask.select<4, 1>(8 * m) = 0xFFFFFFFFu; lomask.select<4, 1>(8 * m + 4) = 0u; }
#pragma unroll
        for (int b = 0; b < 2; b++) {
            simd<float, 256> SC; // [row][16 scales] * d
#pragma unroll
            for (int rr = 0; rr < 16; rr++) {
                const uint8_t * w = rec + (size_t) rr * 432;
                simd<float, 2> dv = convert<float>(block_load<sycl::half, 2>((const sycl::half *) (w + 416)));
                const float d = dv[b];
                SC.select<16, 1>(16 * rr) = convert<float>(block_load<int8_t, 16>((const int8_t *) w + 384 + 16 * b)) * d;
            }
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                simd<uint32_t, 64> RA[2], RB[2], RH[2];
#pragma unroll
                for (int r = 0; r < 2; r++)
#pragma unroll
                    for (int m = 0; m < 8; m++) {
                        const uint8_t * w = rec + (size_t) (8 * r + m) * 432;
                        RA[r].select<8, 1>(8 * m) = block_load<uint32_t, 8>((const uint32_t *) (w + 128 * b + 64 * hf));
                        RB[r].select<8, 1>(8 * m) = block_load<uint32_t, 8>((const uint32_t *) (w + 128 * b + 64 * hf + 32));
                        RH[r].select<8, 1>(8 * m) = block_load<uint32_t, 8>((const uint32_t *) (w + 256 + 64 * b + 32 * hf));
                    }
#pragma unroll
                for (int qq = 0; qq < 4; qq++) {
                    const int sb = 4 * hf + qq; const size_t tile = gi * 16 + 8 * b + sb;
                    simd<int8_t, 512> B;
                    B.select<256, 1>(0) = block_load<int8_t, 256>(t.q + tile * 512); B.select<256, 1>(256) = block_load<int8_t, 256>(t.q + tile * 512 + 256);
                    simd<float, 16> xd = block_load<float, 16>(t.d + tile * 16), xl = block_load<float, 16>(t.s + tile * 32), xh = block_load<float, 16>(t.s + tile * 32 + 16);
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        simd<uint32_t, 64> av;
                        if (qq == 0) av = (RA[r] & 0x0F0F0F0Fu) | ((RH[r] & 0x03030303u) << 4);
                        else if (qq == 1) av = (RB[r] & 0x0F0F0F0Fu) | (((RH[r] >> 2) & 0x03030303u) << 4);
                        else if (qq == 2) av = ((RA[r] >> 4) & 0x0F0F0F0Fu) | (((RH[r] >> 4) & 0x03030303u) << 4);
                        else av = ((RB[r] >> 4) & 0x0F0F0F0Fu) | (((RH[r] >> 6) & 0x03030303u) << 4);
                        simd<uint32_t, 64> alo = av & lomask, ahi = av & ~lomask;
                        simd<int8_t, 256> Alo = alo.bit_cast_view<int8_t>(), Ahi = ahi.bit_cast_view<int8_t>();
                        simd<int, 128> rl = xmx::dpas<8, 8, int>(B, Alo), rh = xmx::dpas<8, 8, int>(B, Ahi);
#pragma unroll
                        for (int m = 0; m < 8; m++) {
                            const float s0 = SC[(8 * r + m) * 16 + 2 * sb], s1 = SC[(8 * r + m) * 16 + 2 * sb + 1];
                            acc[r].select<16, 1>(16 * m) += (convert<float>(simd<int, 16>(rl.select<16, 1>(16 * m))) * s0 + convert<float>(simd<int, 16>(rh.select<16, 1>(16 * m))) * s1) * xd
                                                            - (xl * s0 + xh * s1) * 32.0f;
                        }
                    }
                }
            }
        }
        simd<uint32_t, 16> lane(0u, 1u); simd_mask<16> ok = lane < (uint32_t) cnt;
        simd<uint32_t, 16> pr = 0u; simd<float, 16> wv = 0.0f;
        for (int i = 0; i < cnt; i++) { pr[i] = (uint32_t) grp[2 + i]; wv[i] = wts[grp[2 + i]]; }
        simd<uint32_t, 16> off = pr * (2048u * 4u);
#pragma unroll
        for (int r = 0; r < 2; r++)
#pragma unroll
            for (int m = 0; m < 8; m++) scatter<float, 16>(YP + 16 * rbk + 8 * r + m, off, acc[r].select<16, 1>(16 * m) * wv, ok);
    });
}

// ---------------------------------------------------------------- grouping of the (token, slot) pairs by expert, on the GPU
// ids[np] = expert per pair. GX[0] = number of groups, groups at GX + 32 (32 ints each: expert, n, pair[16]), in expert order.
static inline sycl::event p_group(sycl::queue & q, int np, const int * ids, int * CNT, int * GX) {
    q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int e = (int) id[0]; simd<int, 16> c = 0;
        for (int i = 0; i < np; i += 16) { simd<int, 16> v = block_load<int, 16>(ids + i); simd<int, 16> t = 0; t.merge(simd<int, 16>(1), v == e); c += t; }
        CNT[e] = reduce<int>(c, std::plus<>{});
    });
    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) [[intel::sycl_explicit_simd]] {
        int off = 0;
        for (int e0 = 0; e0 < 256; e0 += 16) {
            simd<int, 16> c = block_load<int, 16>(CNT + e0); simd<int, 16> g = (c + 15) >> 4; simd<int, 16> o;
#pragma unroll
            for (int i = 0; i < 16; i++) { o[i] = off; off += g[i]; }
            block_store<int, 16>(CNT + 256 + e0, o);
        }
        GX[0] = off;
    });
    return q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int e = (int) id[0]; int k = 0; int * g = GX + 32 + (size_t) CNT[256 + e] * 32;
        for (int i = 0; i < np; i += 16) {
            simd<int, 16> v = block_load<int, 16>(ids + i); simd_mask<16> mk = v == e;
            if (!mk.any()) continue;
#pragma unroll
            for (int j = 0; j < 16; j++) if (mk[j]) {
                if ((k & 15) == 0) { if (k) g += 32; g[0] = e; g[1] = 0; }
                g[2 + (k & 15)] = i + j; g[1] = (k & 15) + 1; k++;
            }
        }
    });
}

// ================================================================ flash attention on XMX, reading the tile-layout K/V cache (see KV)
// decode: thread = (column c, kv head, chunk of prm[3] keys (a multiple of 16), half of the head dims); split over chunks
// like the old kernel: writes the chunk partials (m, l, o) to PART for k_att_fin.
static inline sycl::event k_att_t(sycl::queue & q, int nc, const int * prm, int nch, const int8_t * QN, const float * QSN, KV kv, float * PART) {
    return q.parallel_for(sycl::range<1>((size_t) nc * 2 * nch * 2), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int hf = (int) (id[0] % 2); size_t rr = id[0] / 2; const int c = (int) (rr % nc); rr /= nc; const int kvh = (int) (rr % 2); const int ch = (int) (rr / 2);
        if (ch >= prm[2]) return;
        const int P = prm[0], cht = prm[3]; const int t0 = ch * cht;
        int lim = P + c + 1 - t0; if (lim > cht) lim = cht;
        float * out = PART + (((size_t) c * 16 + kvh * 8) * nch + ch) * 272; const size_t ostride = (size_t) nch * 272;
        if (lim <= 0) {
            if (hf == 0) {
#pragma unroll
                for (int i = 0; i < 8; i++) { out[i * ostride] = -1e30f; out[i * ostride + 1] = 0.0f; }
            }
            return;
        }
        const int8_t * qn = QN + ((size_t) c * 16 + kvh * 8) * 256; const float * qsn = QSN + ((size_t) c * 16 + kvh * 8) * 8;
        simd<float, 128> O[8];
#pragma unroll
        for (int g = 0; g < 8; g++) O[g] = 0.0f;
        simd<float, 8> M = -1e30f, Ls = 0.0f;
        simd<uint32_t, 16> lane(0u, 1u);
        for (int kt = 0; kt < lim; kt += 16) {
            const size_t tb = (size_t) kvh * kv.ntile + (t0 + kt) / 16;
            simd_mask<16> ok = lane < (uint32_t) (lim - kt);
            simd<float, 128> S = 0.0f;
#pragma unroll
            for (int b = 0; b < 8; b++) {
                simd<int8_t, 256> A = block_load<int8_t, 256>(qn + b * 256); simd<float, 8> qs = block_load<float, 8>(qsn + b * 8);
                simd<int8_t, 512> B; const int8_t * kp = kv.kt + (tb * 8 + b) * 512;
                B.select<256, 1>(0) = block_load<int8_t, 256>(kp); B.select<256, 1>(256) = block_load<int8_t, 256>(kp + 256);
                simd<float, 16> ks = block_load<float, 16>(kv.ks + (tb * 8 + b) * 16);
                simd<int, 128> res = xmx::dpas<8, 8, int>(B, A);
#pragma unroll
                for (int m = 0; m < 8; m++) { const float qv = qs[m]; S.select<16, 1>(16 * m) += convert<float>(simd<int, 16>(res.select<16, 1>(16 * m))) * (ks * qv); }
            }
            simd<float, 128> Pm;
#pragma unroll
            for (int m = 0; m < 8; m++) {
                simd<float, 16> r = S.select<16, 1>(16 * m); r.merge(simd<float, 16>(-1e30f), !ok);
                const float mx = hmax<float>(r); const float mo = M[m]; const float mn = mx > mo ? mx : mo;
                simd<float, 1> cr = es::exp(simd<float, 1>(mo - mn)); const float corr = cr[0];
                simd<float, 16> e = es::exp(r - mn); Pm.select<16, 1>(16 * m) = e;
                M[m] = mn; Ls[m] = Ls[m] * corr + reduce<float>(e, std::plus<>{});
#pragma unroll
                for (int g = 0; g < 8; g++) O[g].select<16, 1>(16 * m) *= corr;
            }
#pragma unroll
            for (int bb = 0; bb < 4; bb++) {
                const int b = 4 * hf + bb; simd<float, 16> vs = block_load<float, 16>(kv.vs + (tb * 8 + b) * 16); vs.merge(simd<float, 16>(0.0f), !ok);
                simd<sycl::half, 128> A;
#pragma unroll
                for (int m = 0; m < 8; m++) A.select<16, 1>(16 * m) = convert<sycl::half>(simd<float, 16>(Pm.select<16, 1>(16 * m)) * vs);
#pragma unroll
                for (int gg = 0; gg < 2; gg++) {
                    const int g = 2 * b + gg; simd<sycl::half, 256> Bv = convert<sycl::half>(block_load<int8_t, 256>(kv.vt + (tb * 16 + g) * 256));
                    O[2 * bb + gg] = xmx::dpas<8, 8, float>(O[2 * bb + gg], Bv, A);
                }
            }
        }
#pragma unroll
        for (int m = 0; m < 8; m++) {
#pragma unroll
            for (int g = 0; g < 8; g++) block_store<float, 16>(out + m * ostride + 16 + 128 * hf + 16 * g, O[g].select<16, 1>(16 * m));
            if (hf == 0) { out[m * ostride] = M[m]; out[m * ostride + 1] = Ls[m]; }
        }
    });
}

// thread = (query c of the chunk, kv head, half of the head dims). Rows = the 8 query heads of the kv head.
// S = Q.K (int8 dpas per 32-dim block, x scales), online softmax, O += P.V (fp16 dpas, V scales folded into P).
// Output: gated (sigmoid of the gate half of Y1) and quantized into act(4096) like k_att_fin.
static inline sycl::event p_flash(sycl::queue & q, int n, int P0, const int8_t * QN, const float * QSN, KV at, const float * Y1, Act ao) {
    return q.parallel_for(sycl::range<1>((size_t) n * 4), [=](sycl::id<1> id) [[intel::sycl_explicit_simd]] {
        const int hf = (int) (id[0] % 2), kvh = (int) (id[0] / 2 % 2); const size_t c = id[0] / 4; const int pos = P0 + (int) c;
        const int ntl = pos / 16 + 1;
        simd<float, 128> O[8];
#pragma unroll
        for (int g = 0; g < 8; g++) O[g] = 0.0f;
        simd<float, 8> M = -1e30f, Lsum = 0.0f;
        simd<uint32_t, 16> lane(0u, 1u);
        const int8_t * qn = QN + (c * 16 + kvh * 8) * 256; const float * qsn = QSN + (c * 16 + kvh * 8) * 8;
        for (int t = 0; t < ntl; t++) {
            const size_t tb = (size_t) kvh * at.ntile + t;
            simd<float, 128> S = 0.0f;
#pragma unroll
            for (int b = 0; b < 8; b++) {
                simd<int8_t, 256> A = block_load<int8_t, 256>(qn + b * 256);
                simd<int8_t, 512> B; const int8_t * kt = at.kt + (tb * 8 + b) * 512;
                B.select<256, 1>(0) = block_load<int8_t, 256>(kt); B.select<256, 1>(256) = block_load<int8_t, 256>(kt + 256);
                simd<float, 16> ks = block_load<float, 16>(at.ks + (tb * 8 + b) * 16); simd<float, 8> qs = block_load<float, 8>(qsn + b * 8);
                simd<int, 128> res = xmx::dpas<8, 8, int>(B, A);
#pragma unroll
                for (int m = 0; m < 8; m++) { const float qv = qs[m]; S.select<16, 1>(16 * m) += convert<float>(simd<int, 16>(res.select<16, 1>(16 * m))) * (ks * qv); }
            }
            if (t == ntl - 1) { simd_mask<16> bad = (lane + (uint32_t) (16 * t)) > (uint32_t) pos;
#pragma unroll
                for (int m = 0; m < 8; m++) { simd<float, 16> r = S.select<16, 1>(16 * m); r.merge(simd<float, 16>(-1e30f), bad); S.select<16, 1>(16 * m) = r; } }
            simd<float, 128> Pm;
#pragma unroll
            for (int m = 0; m < 8; m++) {
                simd<float, 16> r = S.select<16, 1>(16 * m); const float mx = hmax<float>(r); const float mo = M[m]; const float mn = mx > mo ? mx : mo;
                simd<float, 1> cr = es::exp(simd<float, 1>(mo - mn)); const float corr = cr[0];
                simd<float, 16> e = es::exp(r - mn); Pm.select<16, 1>(16 * m) = e;
                M[m] = mn; Lsum[m] = Lsum[m] * corr + reduce<float>(e, std::plus<>{});
#pragma unroll
                for (int g = 0; g < 8; g++) O[g].select<16, 1>(16 * m) *= corr;
            }
#pragma unroll
            for (int bb = 0; bb < 4; bb++) { // 32-dim block of this half
                const int b = 4 * hf + bb; simd<float, 16> vs = block_load<float, 16>(at.vs + (tb * 8 + b) * 16);
                if (t == ntl - 1) vs.merge(simd<float, 16>(0.0f), (lane + (uint32_t) (16 * t)) > (uint32_t) pos); // positions past this query: stale cache entries
                simd<sycl::half, 128> A;
#pragma unroll
                for (int m = 0; m < 8; m++) A.select<16, 1>(16 * m) = convert<sycl::half>(simd<float, 16>(Pm.select<16, 1>(16 * m)) * vs);
#pragma unroll
                for (int gg = 0; gg < 2; gg++) {
                    const int g = 2 * b + gg; simd<sycl::half, 256> B = convert<sycl::half>(block_load<int8_t, 256>(at.vt + (tb * 16 + g) * 256));
                    O[2 * bb + gg] = xmx::dpas<8, 8, float>(O[2 * bb + gg], B, A);
                }
            }
        }
        // finish: normalize, gate, quantize per 32 dims
#pragma unroll
        for (int m = 0; m < 8; m++) {
            const int h = kvh * 8 + m; const float inv = 1.0f / Lsum[m]; const float * gate = Y1 + c * 9216 + h * 512 + 256 + 128 * hf;
#pragma unroll
            for (int bb = 0; bb < 4; bb++) {
                simd<float, 16> o0 = O[2 * bb].select<16, 1>(16 * m) * inv * sigmoid16(block_load<float, 16>(gate + 32 * bb));
                simd<float, 16> o1 = O[2 * bb + 1].select<16, 1>(16 * m) * inv * sigmoid16(block_load<float, 16>(gate + 32 * bb + 16));
                const int blk = 4 * hf + bb;
                quant32(o0, o1, ao.q + c * 4096 + h * 256 + 32 * blk, ao.d + c * 128 + h * 8 + blk, ao.s + c * 256 + h * 16 + 2 * blk);
            }
        }
    });
}

} // namespace q36g
