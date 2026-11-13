// q36: CPU kernels (AVX2/FMA/F16C) - exact float math on the quantized weights, no activation quantization.
// y[r] = sum_i dequant(W[r][i]) * x[i], computed as sum over sub-blocks of scale * (sum q*x) - min * (sum x).
#pragma once
#include "gguf.hpp"
#include <immintrin.h>
#include <cmath>
#include <thread>
#include <atomic>
#include <functional>
#include <algorithm>

namespace q36 {

// ---------------------------------------------------------------- thread pool (spinning workers)
struct Pool {
    int nth = 1;
    std::vector<std::thread> th;
    std::atomic<uint64_t> gen{0};
    std::atomic<int> pending{0};
    std::atomic<bool> stop{false};
    std::function<void(int)> job; // job(thread index)

    explicit Pool(int n) : nth(std::max(1, n)) {
        for (int i = 1; i < nth; i++) th.emplace_back([this, i] { worker(i); });
    }
    ~Pool() { stop = true; gen++; for (auto & t : th) t.join(); }
    void worker(int i) {
        uint64_t seen = 0;
        for (;;) {
            int spins = 0;
            while (gen.load(std::memory_order_acquire) == seen) {
                if (++spins < 20000) _mm_pause(); else std::this_thread::yield();
            }
            seen = gen.load(std::memory_order_acquire);
            if (stop) return;
            job(i);
            pending.fetch_sub(1, std::memory_order_release);
        }
    }
    void run(const std::function<void(int)> & f) {
        if (nth == 1) { f(0); return; }
        job = f; pending.store(nth - 1, std::memory_order_release); gen.fetch_add(1, std::memory_order_release);
        f(0);
        while (pending.load(std::memory_order_acquire) != 0) _mm_pause();
    }
    // split [0, n) into contiguous chunks, one per thread
    template <class F> void pfor(int64_t n, F && f) {
        if (n <= 0) return;
        int k = (int) std::min<int64_t>(nth, n);
        if (k == 1) { f((int64_t) 0, n); return; }
        run([&](int i) { if (i < k) { int64_t a = n * i / k, b = n * (i + 1) / k; f(a, b); } });
    }
};

// ---------------------------------------------------------------- helpers
static inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s)); s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
static inline float bf2f(uint16_t h) { uint32_t u = (uint32_t) h << 16; float f; memcpy(&f, &u, 4); return f; }

static inline __m256 u8x8_ps(__m128i v) { return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v)); }
// sum of 16 u8 (low 128 bits) * x[0..15] as 8 partial sums, added to acc
static inline __m256 dot16_u8(__m128i q, const float * x, __m256 acc) {
    acc = _mm256_fmadd_ps(u8x8_ps(q), _mm256_loadu_ps(x), acc);
    return _mm256_fmadd_ps(u8x8_ps(_mm_srli_si128(q, 8)), _mm256_loadu_ps(x + 8), acc);
}
static inline __m256 dot32_u8(__m256i q, const float * x) {
    __m256 a = dot16_u8(_mm256_castsi256_si128(q), x, _mm256_setzero_ps());
    return dot16_u8(_mm256_extracti128_si256(q, 1), x + 16, a);
}

static inline float dot_f32(const float * a, const float * b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0; int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    float s = hsum(_mm256_add_ps(s0, s1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

// sums of x over consecutive groups of 16
static inline void xsum16(const float * x, int n, float * out) {
    for (int i = 0; i < n; i += 16) out[i / 16] = hsum(_mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(x + i + 8)));
}

// ---------------------------------------------------------------- one row of a quantized matrix times x
static inline float row_q8_0(const uint8_t * w, int n, const float * x) {
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n / 32; b++, w += 34, x += 32) {
        uint16_t dh; memcpy(&dh, w, 2);
        const __m128i * q = (const __m128i *) (w + 2);
        __m128i q0 = _mm_loadu_si128(q), q1 = _mm_loadu_si128(q + 1);
        __m256 s = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0)), _mm256_loadu_ps(x));
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8))), _mm256_loadu_ps(x + 8), s);
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1)), _mm256_loadu_ps(x + 16), s);
        s = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8))), _mm256_loadu_ps(x + 24), s);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(h2f(dh)), s, acc);
    }
    return hsum(acc);
}

static inline void k4_scales(const uint8_t * q, float d, float dmin, float * sc, float * mn) {
    for (int j = 0; j < 8; j++) {
        uint8_t s, m;
        if (j < 4) { s = q[j] & 63; m = q[j + 4] & 63; }
        else { s = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
        sc[j] = d * s; mn[j] = dmin * m;
    }
}

// xs = sums of x over groups of 16
static inline float row_q4_k(const uint8_t * w, int n, const float * x, const float * xs) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps(); float msum = 0;
    for (int b = 0; b < n / 256; b++, w += 144, x += 256, xs += 16) {
        uint16_t dh, mh; memcpy(&dh, w, 2); memcpy(&mh, w + 2, 2);
        float sc[8], mn[8]; k4_scales(w + 4, h2f(dh), h2f(mh), sc, mn);
        const uint8_t * qs = w + 16;
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

static inline float row_q5_k(const uint8_t * w, int n, const float * x, const float * xs) {
    const __m256i m4 = _mm256_set1_epi8(0xF), one = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps(); float msum = 0;
    for (int b = 0; b < n / 256; b++, w += 176, x += 256, xs += 16) {
        uint16_t dh, mh; memcpy(&dh, w, 2); memcpy(&mh, w + 2, 2);
        float sc[8], mn[8]; k4_scales(w + 4, h2f(dh), h2f(mh), sc, mn);
        const __m256i qh = _mm256_loadu_si256((const __m256i *) (w + 16));
        const uint8_t * qs = w + 48;
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

static inline float row_q6_k(const uint8_t * w, int n, const float * x, const float * xs) {
    const __m256i m4 = _mm256_set1_epi8(0xF), m2 = _mm256_set1_epi8(3);
    __m256 acc = _mm256_setzero_ps(); float msum = 0;
    for (int b = 0; b < n / 256; b++, w += 210, x += 256, xs += 16) {
        const int8_t * scl = (const int8_t *) (w + 192);
        uint16_t dh; memcpy(&dh, w + 208, 2); const float d = h2f(dh);
        for (int hf = 0; hf < 2; hf++) {
            __m256i A = _mm256_loadu_si256((const __m256i *) (w + 64 * hf));
            __m256i B = _mm256_loadu_si256((const __m256i *) (w + 64 * hf + 32));
            __m256i H = _mm256_loadu_si256((const __m256i *) (w + 128 + 32 * hf));
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

static inline float row_bf16(const uint8_t * w, int n, const float * x) {
    float s = 0; const uint16_t * h = (const uint16_t *) w;
    for (int i = 0; i < n; i++) s += bf2f(h[i]) * x[i];
    return s;
}
static inline float row_f16(const uint8_t * w, int n, const float * x) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) acc = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (w + 2 * i))), _mm256_loadu_ps(x + i), acc);
    return hsum(acc);
}

// ---------------------------------------------------------------- matrix view + mat-vec
struct Mat {
    uint32_t type = 0; const uint8_t * data = nullptr; int n_in = 0; int64_t n_out = 0; uint64_t row_bytes = 0;
    Mat() {}
    Mat(const GTensor & t) : type(t.type), data(t.data), n_in((int) t.dims[0]), n_out((int64_t) (t.nelem() / t.dims[0])), row_bytes(type_row_bytes(t.type, t.dims[0])) {}
    Mat rows(int64_t r0, int64_t n) const { Mat m = *this; m.data = data + r0 * row_bytes; m.n_out = n; return m; }
};

static inline float mat_row(const Mat & m, int64_t r, const float * x, const float * xs) {
    const uint8_t * w = m.data + r * m.row_bytes;
    switch (m.type) {
        case T_Q8_0: return row_q8_0(w, m.n_in, x);
        case T_Q4_K: return row_q4_k(w, m.n_in, x, xs);
        case T_Q5_K: return row_q5_k(w, m.n_in, x, xs);
        case T_Q6_K: return row_q6_k(w, m.n_in, x, xs);
        case T_F32:  return dot_f32((const float *) w, x, m.n_in);
        case T_BF16: return row_bf16(w, m.n_in, x);
        case T_F16:  return row_f16(w, m.n_in, x);
    }
    return 0;
}

// y = M x (single thread, rows [r0, r1))
static inline void matvec_range(const Mat & m, const float * x, const float * xs, float * y, int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; r++) y[r] = mat_row(m, r, x, xs);
}
static inline void matvec(Pool & pool, const Mat & m, const float * x, float * y) {
    std::vector<float> xs(m.n_in / 16 + 1); xsum16(x, m.n_in, xs.data());
    pool.pfor(m.n_out, [&](int64_t a, int64_t b) { matvec_range(m, x, xs.data(), y, a, b); });
}

// dequantize one row to f32 (used for embeddings and tests)
static inline void dequant_row(const Mat & m, int64_t r, float * out) {
    const uint8_t * w = m.data + r * m.row_bytes; const int n = m.n_in;
    switch (m.type) {
        case T_F32: memcpy(out, w, 4 * (size_t) n); return;
        case T_F16: for (int i = 0; i < n; i++) { uint16_t h; memcpy(&h, w + 2 * i, 2); out[i] = h2f(h); } return;
        case T_BF16: for (int i = 0; i < n; i++) { uint16_t h; memcpy(&h, w + 2 * i, 2); out[i] = bf2f(h); } return;
        case T_Q8_0:
            for (int b = 0; b < n / 32; b++, w += 34) { uint16_t dh; memcpy(&dh, w, 2); float d = h2f(dh); for (int i = 0; i < 32; i++) out[b * 32 + i] = d * (int8_t) w[2 + i]; }
            return;
        case T_Q4_K:
            for (int b = 0; b < n / 256; b++, w += 144, out += 256) {
                uint16_t dh, mh; memcpy(&dh, w, 2); memcpy(&mh, w + 2, 2); float sc[8], mn[8]; k4_scales(w + 4, h2f(dh), h2f(mh), sc, mn);
                for (int g = 0; g < 4; g++) for (int l = 0; l < 32; l++) {
                    uint8_t v = w[16 + 32 * g + l];
                    out[64 * g + l] = sc[2 * g] * (v & 0xF) - mn[2 * g]; out[64 * g + 32 + l] = sc[2 * g + 1] * (v >> 4) - mn[2 * g + 1];
                }
            }
            return;
        case T_Q5_K:
            for (int b = 0; b < n / 256; b++, w += 176, out += 256) {
                uint16_t dh, mh; memcpy(&dh, w, 2); memcpy(&mh, w + 2, 2); float sc[8], mn[8]; k4_scales(w + 4, h2f(dh), h2f(mh), sc, mn);
                for (int g = 0; g < 4; g++) for (int l = 0; l < 32; l++) {
                    uint8_t v = w[48 + 32 * g + l], h = w[16 + l];
                    out[64 * g + l] = sc[2 * g] * ((v & 0xF) + ((h >> (2 * g)) & 1) * 16) - mn[2 * g];
                    out[64 * g + 32 + l] = sc[2 * g + 1] * ((v >> 4) + ((h >> (2 * g + 1)) & 1) * 16) - mn[2 * g + 1];
                }
            }
            return;
        case T_Q6_K:
            for (int b = 0; b < n / 256; b++, w += 210, out += 256) {
                const int8_t * sc = (const int8_t *) (w + 192); uint16_t dh; memcpy(&dh, w + 208, 2); float d = h2f(dh);
                for (int hf = 0; hf < 2; hf++) for (int l = 0; l < 32; l++) {
                    const uint8_t * ql = w + 64 * hf, * qh = w + 128 + 32 * hf; int is = l / 16 + 8 * hf; float * y = out + 128 * hf;
                    int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32, q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32,  q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l] = d * sc[is] * q1; y[l + 32] = d * sc[is + 2] * q2; y[l + 64] = d * sc[is + 4] * q3; y[l + 96] = d * sc[is + 6] * q4;
                }
            }
            return;
    }
}

static inline void rms_norm(const float * x, const float * w, float * y, int n, float eps) {
    double s = 0; for (int i = 0; i < n; i++) s += (double) x[i] * x[i];
    const float sc = 1.0f / sqrtf((float) (s / n) + eps);
    if (w) for (int i = 0; i < n; i++) y[i] = x[i] * sc * w[i]; else for (int i = 0; i < n; i++) y[i] = x[i] * sc;
}
static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

} // namespace q36
