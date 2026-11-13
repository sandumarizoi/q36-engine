// q36: Qwen3.6-35B-A3B (qwen35moe) reference forward pass on the CPU. One sequence, one token at a time.
#pragma once
#include "cpu.hpp"
#include <chrono>

namespace q36 {

struct HP {
    int n_layer = 40, n_embd = 2048, n_vocab = 248320;
    int n_head = 16, n_head_kv = 2, head_dim = 256, n_rot = 64;
    int n_expert = 256, n_used = 8, n_ff = 512;
    int gdn_kh = 16, gdn_vh = 32, gdn_d = 128, conv_k = 4;
    float eps = 1e-6f, rope_base = 1e7f;
    int qkv_dim() const { return 2 * gdn_kh * gdn_d + gdn_vh * gdn_d; } // 8192
};

struct Layer {
    bool recr = false;
    const float * attn_norm = nullptr, * post_norm = nullptr;
    // gated delta net
    Mat qkv, zgate, ssm_out, alpha, beta;
    const float * conv = nullptr, * dt_bias = nullptr, * ssm_a = nullptr, * ssm_norm = nullptr;
    // attention
    Mat wq, wk, wv, wo; const float * q_norm = nullptr, * k_norm = nullptr;
    // MoE
    Mat router, sh_gate_inp, sh_gate, sh_up, sh_down, e_gate, e_up, e_down;
    // MTP block only
    Mat eh_proj; const float * enorm = nullptr, * hnorm = nullptr, * head_norm = nullptr;
};

struct Model {
    GGUF g; HP hp;
    std::vector<Layer> layers; // 40 trunk + 1 MTP
    Mat tok_embd, output; const float * output_norm = nullptr;

    static const float * f32(const GTensor & t) { if (t.type != T_F32) throw std::runtime_error("expected F32: " + t.name); return (const float *) t.data; }

    void load(const std::string & path, bool vocab = false) {
        g.open(path, vocab);
        const std::string a = "qwen35moe.";
        hp.n_layer = (int) g.num(a + "block_count") - (int) g.num(a + "nextn_predict_layers");
        hp.n_embd = (int) g.num(a + "embedding_length");
        hp.n_head = (int) g.num(a + "attention.head_count"); hp.n_head_kv = (int) g.num(a + "attention.head_count_kv");
        hp.head_dim = (int) g.num(a + "attention.key_length"); hp.n_rot = (int) g.num(a + "rope.dimension_count");
        hp.n_expert = (int) g.num(a + "expert_count"); hp.n_used = (int) g.num(a + "expert_used_count");
        hp.n_ff = (int) g.num(a + "expert_feed_forward_length");
        hp.gdn_kh = (int) g.num(a + "ssm.group_count"); hp.gdn_vh = (int) g.num(a + "ssm.time_step_rank"); hp.gdn_d = (int) g.num(a + "ssm.state_size");
        hp.conv_k = (int) g.num(a + "ssm.conv_kernel");
        hp.eps = (float) g.num(a + "attention.layer_norm_rms_epsilon"); hp.rope_base = (float) g.num(a + "rope.freq_base");
        const int interval = (int) g.num(a + "full_attention_interval", 4);
        tok_embd = Mat(g.get("token_embd.weight")); output = Mat(g.get("output.weight")); output_norm = f32(g.get("output_norm.weight"));
        hp.n_vocab = (int) tok_embd.n_out;
        layers.resize(hp.n_layer + 1);
        for (int il = 0; il <= hp.n_layer; il++) {
            Layer & L = layers[il]; const std::string p = "blk." + std::to_string(il) + ".";
            if (il == hp.n_layer && !g.find(p + "nextn.eh_proj.weight")) { layers.pop_back(); break; }
            L.recr = il < hp.n_layer && ((il + 1) % interval != 0);
            L.attn_norm = f32(g.get(p + "attn_norm.weight")); L.post_norm = f32(g.get(p + "post_attention_norm.weight"));
            if (L.recr) {
                L.qkv = Mat(g.get(p + "attn_qkv.weight")); L.zgate = Mat(g.get(p + "attn_gate.weight")); L.ssm_out = Mat(g.get(p + "ssm_out.weight"));
                L.alpha = Mat(g.get(p + "ssm_alpha.weight")); L.beta = Mat(g.get(p + "ssm_beta.weight"));
                L.conv = f32(g.get(p + "ssm_conv1d.weight")); L.dt_bias = f32(g.get(p + "ssm_dt.bias")); L.ssm_a = f32(g.get(p + "ssm_a")); L.ssm_norm = f32(g.get(p + "ssm_norm.weight"));
            } else {
                L.wq = Mat(g.get(p + "attn_q.weight")); L.wk = Mat(g.get(p + "attn_k.weight")); L.wv = Mat(g.get(p + "attn_v.weight")); L.wo = Mat(g.get(p + "attn_output.weight"));
                L.q_norm = f32(g.get(p + "attn_q_norm.weight")); L.k_norm = f32(g.get(p + "attn_k_norm.weight"));
            }
            L.router = Mat(g.get(p + "ffn_gate_inp.weight")); L.sh_gate_inp = Mat(g.get(p + "ffn_gate_inp_shexp.weight"));
            L.sh_gate = Mat(g.get(p + "ffn_gate_shexp.weight")); L.sh_up = Mat(g.get(p + "ffn_up_shexp.weight")); L.sh_down = Mat(g.get(p + "ffn_down_shexp.weight"));
            L.e_gate = Mat(g.get(p + "ffn_gate_exps.weight")); L.e_up = Mat(g.get(p + "ffn_up_exps.weight")); L.e_down = Mat(g.get(p + "ffn_down_exps.weight"));
            if (il == hp.n_layer) {
                L.eh_proj = Mat(g.get(p + "nextn.eh_proj.weight")); L.enorm = f32(g.get(p + "nextn.enorm.weight")); L.hnorm = f32(g.get(p + "nextn.hnorm.weight"));
                auto hn = g.find(p + "nextn.shared_head_norm.weight"); L.head_norm = hn ? f32(*hn) : output_norm;
            }
        }
    }
    bool has_mtp() const { return (int) layers.size() == hp.n_layer + 1; }
};

// per-sequence state
struct State {
    std::vector<std::vector<float>> conv;   // [layer] (conv_k-1) x qkv_dim, oldest first
    std::vector<std::vector<float>> S;      // [layer] vh x d x d, stored transposed: S[h][j*d + i] = S_ij
    std::vector<std::vector<float>> K, V;   // [layer] n_tok x n_head_kv x head_dim (K after norm + rope)
    int n_past = 0;
    void init(const Model & m) {
        const HP & hp = m.hp; const int nl = (int) m.layers.size();
        conv.assign(nl, {}); S.assign(nl, {}); K.assign(nl, {}); V.assign(nl, {}); n_past = 0;
        for (int il = 0; il < nl; il++) if (m.layers[il].recr) {
            conv[il].assign((size_t) (hp.conv_k - 1) * hp.qkv_dim(), 0.0f);
            S[il].assign((size_t) hp.gdn_vh * hp.gdn_d * hp.gdn_d, 0.0f);
        }
    }
};

struct Trace { // optional per-token observation hooks
    std::vector<uint8_t> * ids = nullptr;       // appends n_used expert ids per layer
    std::vector<float> * probs = nullptr;       // appends n_used normalized weights per layer
};

struct Engine {
    const Model & m; Pool & pool; const HP & hp;
    std::vector<float> xn, qkv, z, tmp, attn, a8, b8, logits_r, hbuf, gate, up, eo;
    std::vector<float> rope_cos, rope_sin;
    double t_dense = 0, t_exp = 0, t_gdn = 0, t_attn = 0, t_head = 0;

    Engine(const Model & model, Pool & p) : m(model), pool(p), hp(model.hp) {
        xn.resize(hp.n_embd); qkv.resize(hp.qkv_dim()); z.resize(4096); tmp.resize(8192); attn.resize(4096);
        logits_r.resize(hp.n_expert); hbuf.resize(hp.n_embd); gate.resize(hp.n_ff); up.resize(hp.n_ff); eo.resize((size_t) hp.n_used * hp.n_embd);
        rope_cos.resize(hp.n_rot / 2); rope_sin.resize(hp.n_rot / 2);
    }
    static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

    void rope_prepare(int pos) {
        const float theta_scale = powf(hp.rope_base, -2.0f / hp.n_rot);
        float theta = (float) pos;
        for (int i = 0; i < hp.n_rot / 2; i++) { rope_cos[i] = cosf(theta); rope_sin[i] = sinf(theta); theta *= theta_scale; }
    }
    void rope(float * v) const { // pairs (i, i + n_rot/2), remaining dims untouched
        const int h = hp.n_rot / 2;
        for (int i = 0; i < h; i++) { float x0 = v[i], x1 = v[i + h]; v[i] = x0 * rope_cos[i] - x1 * rope_sin[i]; v[i + h] = x0 * rope_sin[i] + x1 * rope_cos[i]; }
    }

    // y += gated delta net block output for the normalized input xn
    void gdn(const Layer & L, State & st, int il, const float * x, float * y) {
        const int D = hp.gdn_d, KH = hp.gdn_kh, VH = hp.gdn_vh, C = hp.qkv_dim(), CK = hp.conv_k;
        double t0 = now();
        matvec(pool, L.qkv, x, qkv.data());
        matvec(pool, L.zgate, x, z.data());
        float al[64], be[64];
        matvec(pool, L.alpha, x, al); matvec(pool, L.beta, x, be);
        t_dense += now() - t0; t0 = now();
        // causal conv over the last CK inputs per channel + SiLU, then roll the conv state
        float * cs = st.conv[il].data(); float * c = tmp.data();
        for (int ch = 0; ch < C; ch++) {
            float s = qkv[ch] * L.conv[(CK - 1) + ch * CK];
            for (int i = 0; i < CK - 1; i++) s += cs[(size_t) i * C + ch] * L.conv[i + ch * CK];
            c[ch] = silu(s);
        }
        memmove(cs, cs + C, sizeof(float) * (size_t) (CK - 2) * C);
        memcpy(cs + (size_t) (CK - 2) * C, qkv.data(), sizeof(float) * C);
        float * q = c, * k = c + KH * D, * v = c + 2 * KH * D;
        for (int h = 0; h < 2 * KH; h++) { // L2 norm of q and k heads
            float * p = c + h * D; double s = 0; for (int i = 0; i < D; i++) s += (double) p[i] * p[i];
            const float sc = 1.0f / sqrtf((float) s + hp.eps); for (int i = 0; i < D; i++) p[i] *= sc;
        }
        const float qs = 1.0f / sqrtf((float) D);
        float * S = st.S[il].data(); float * o = attn.data();
        pool.pfor(VH, [&](int64_t h0, int64_t h1) {
            for (int64_t h = h0; h < h1; h++) {
                const float * qh = q + (h % KH) * D, * kh = k + (h % KH) * D, * vh = v + h * D;
                const float g = expf(softplus(al[h] + L.dt_bias[h]) * L.ssm_a[h]), b = sigmoid(be[h]);
                float * Sh = S + (size_t) h * D * D;
                for (int j = 0; j < D; j++) {
                    float * r = Sh + (size_t) j * D;
                    const float delta = (vh[j] - g * dot_f32(r, kh, D)) * b;
                    const __m256 vg = _mm256_set1_ps(g), vd = _mm256_set1_ps(delta);
                    __m256 acc = _mm256_setzero_ps();
                    for (int i = 0; i < D; i += 8) {
                        __m256 s = _mm256_fmadd_ps(vd, _mm256_loadu_ps(kh + i), _mm256_mul_ps(vg, _mm256_loadu_ps(r + i)));
                        _mm256_storeu_ps(r + i, s);
                        acc = _mm256_fmadd_ps(s, _mm256_loadu_ps(qh + i), acc);
                    }
                    o[h * D + j] = hsum(acc) * qs;
                }
            }
        });
        for (int h = 0; h < VH; h++) { // gated RMS norm: rms_norm(o) * w * silu(z)
            float * p = o + h * D; double s = 0; for (int i = 0; i < D; i++) s += (double) p[i] * p[i];
            const float sc = 1.0f / sqrtf((float) (s / D) + hp.eps);
            for (int i = 0; i < D; i++) p[i] = p[i] * sc * L.ssm_norm[i] * silu(z[h * D + i]);
        }
        t_gdn += now() - t0; t0 = now();
        matvec(pool, L.ssm_out, o, y);
        t_dense += now() - t0;
    }

    // full attention block (also used by the MTP block with its own K/V)
    void attention(const Layer & L, State & st, int il, const float * x, float * y, bool kv_only = false) {
        const int HD = hp.head_dim, NH = hp.n_head, NKV = hp.n_head_kv;
        double t0 = now();
        float kv[2][1024];
        matvec(pool, L.wk, x, kv[0]); matvec(pool, L.wv, x, kv[1]);
        for (int h = 0; h < NKV; h++) { float * p = kv[0] + h * HD; rms_norm(p, L.k_norm, p, HD, hp.eps); rope(p); }
        st.K[il].insert(st.K[il].end(), kv[0], kv[0] + NKV * HD); st.V[il].insert(st.V[il].end(), kv[1], kv[1] + NKV * HD);
        if (kv_only) { t_dense += now() - t0; return; }
        matvec(pool, L.wq, x, tmp.data()); // per head: [q (HD) | gate (HD)]
        t_dense += now() - t0; t0 = now();
        const int n = (int) (st.K[il].size() / (NKV * HD));
        const float * Kc = st.K[il].data(), * Vc = st.V[il].data(); float * o = attn.data();
        const float scale = 1.0f / sqrtf((float) HD);
        pool.pfor(NH, [&](int64_t h0, int64_t h1) {
            std::vector<float> sc(n); float qh[1024];
            for (int64_t h = h0; h < h1; h++) {
                const float * qf = tmp.data() + h * 2 * HD; const int kvh = (int) (h / (NH / NKV));
                rms_norm(qf, L.q_norm, qh, HD, hp.eps); rope(qh);
                float mx = -1e30f;
                for (int t = 0; t < n; t++) { sc[t] = dot_f32(qh, Kc + ((size_t) t * NKV + kvh) * HD, HD) * scale; mx = std::max(mx, sc[t]); }
                double sum = 0; for (int t = 0; t < n; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
                float * oh = o + h * HD; memset(oh, 0, sizeof(float) * HD);
                for (int t = 0; t < n; t++) {
                    const __m256 w = _mm256_set1_ps((float) (sc[t] / sum)); const float * vt = Vc + ((size_t) t * NKV + kvh) * HD;
                    for (int i = 0; i < HD; i += 8) _mm256_storeu_ps(oh + i, _mm256_fmadd_ps(w, _mm256_loadu_ps(vt + i), _mm256_loadu_ps(oh + i)));
                }
                for (int i = 0; i < HD; i++) oh[i] *= sigmoid(qf[HD + i]);
            }
        });
        t_attn += now() - t0; t0 = now();
        matvec(pool, L.wo, o, y);
        t_dense += now() - t0;
    }

    // y = MoE(x): routed top-k experts + gated shared expert
    void moe(const Layer & L, const float * x, float * y, Trace * tr) {
        const int E = hp.n_expert, U = hp.n_used, F = hp.n_ff, N = hp.n_embd;
        double t0 = now();
        matvec(pool, L.router, x, logits_r.data());
        float mx = -1e30f; for (int e = 0; e < E; e++) mx = std::max(mx, logits_r[e]);
        double sum = 0; for (int e = 0; e < E; e++) { logits_r[e] = expf(logits_r[e] - mx); sum += logits_r[e]; }
        int ids[16]; float w[16]; float wsum = 0;
        std::vector<int> ord(E); for (int e = 0; e < E; e++) ord[e] = e;
        std::partial_sort(ord.begin(), ord.begin() + U, ord.end(), [&](int a, int b) { return logits_r[a] > logits_r[b] || (logits_r[a] == logits_r[b] && a < b); });
        for (int u = 0; u < U; u++) { ids[u] = ord[u]; w[u] = (float) (logits_r[ids[u]] / sum); wsum += w[u]; }
        wsum = std::max(wsum, 6.103515625e-5f);
        for (int u = 0; u < U; u++) w[u] /= wsum;
        if (tr && tr->ids) for (int u = 0; u < U; u++) tr->ids->push_back((uint8_t) ids[u]);
        if (tr && tr->probs) for (int u = 0; u < U; u++) tr->probs->push_back(w[u]);
        float xs[256]; xsum16(x, N, xs);
        // shared expert
        float sg[1]; matvec_range(L.sh_gate_inp, x, xs, sg, 0, 1); const float shw = sigmoid(sg[0]);
        std::vector<float> hs(F), us(F), hsum16(F / 16);
        pool.pfor(F, [&](int64_t a, int64_t b) { matvec_range(L.sh_gate, x, xs, hs.data(), a, b); matvec_range(L.sh_up, x, xs, us.data(), a, b); });
        for (int i = 0; i < F; i++) hs[i] = silu(hs[i]) * us[i];
        xsum16(hs.data(), F, hsum16.data());
        pool.pfor(N, [&](int64_t a, int64_t b) { matvec_range(L.sh_down, hs.data(), hsum16.data(), y, a, b); });
        for (int i = 0; i < N; i++) y[i] *= shw;
        t_dense += now() - t0; t0 = now();
        // routed experts: one thread per expert when possible
        std::vector<float> act((size_t) U * F), act_s((size_t) U * (F / 16)), up_((size_t) U * F);
        pool.pfor((int64_t) U * F, [&](int64_t a, int64_t b) {
            for (int64_t r = a; r < b; r++) {
                const int u = (int) (r / F); const int64_t row = (int64_t) ids[u] * F + r % F;
                act[r] = silu(mat_row(L.e_gate, row, x, xs)) * mat_row(L.e_up, row, x, xs);
            }
        });
        for (int u = 0; u < U; u++) xsum16(act.data() + (size_t) u * F, F, act_s.data() + (size_t) u * (F / 16));
        pool.pfor((int64_t) U * N, [&](int64_t a, int64_t b) {
            for (int64_t r = a; r < b; r++) {
                const int u = (int) (r / N);
                eo[r] = mat_row(L.e_down, (int64_t) ids[u] * N + r % N, act.data() + (size_t) u * F, act_s.data() + (size_t) u * (F / 16));
            }
        });
        for (int u = 0; u < U; u++) for (int i = 0; i < N; i++) y[i] += w[u] * eo[(size_t) u * N + i];
        t_exp += now() - t0;
    }

    void embed(int tok, float * x) const { dequant_row(m.tok_embd, tok, x); }

    // one trunk step: x (token embedding) -> h = output_norm(last hidden). Appends to the K/V caches and updates the GDN state.
    void step(State & st, const float * emb, float * h_out, Trace * tr = nullptr) {
        const int N = hp.n_embd; std::vector<float> x(emb, emb + N), y(N);
        rope_prepare(st.n_past);
        for (int il = 0; il < hp.n_layer; il++) {
            const Layer & L = m.layers[il];
            rms_norm(x.data(), L.attn_norm, xn.data(), N, hp.eps);
            if (L.recr) gdn(L, st, il, xn.data(), y.data()); else attention(L, st, il, xn.data(), y.data());
            for (int i = 0; i < N; i++) x[i] += y[i];
            rms_norm(x.data(), L.post_norm, xn.data(), N, hp.eps);
            moe(L, xn.data(), y.data(), tr);
            for (int i = 0; i < N; i++) x[i] += y[i];
        }
        rms_norm(x.data(), m.output_norm, h_out, N, hp.eps);
        st.n_past++;
    }

    void head(const float * h, float * logits) {
        double t0 = now(); matvec(pool, m.output, h, logits); t_head += now() - t0;
    }

    // MTP block: (embedding of the token at position p, hidden state that predicted it) -> hidden for the token at p+1.
    // Uses layer index n_layer of the state for its own K/V; pos is the position of `tok`.
    void mtp_step(State & st, const float * emb, const float * h_prev, int pos, float * h_out, bool kv_only = false) {
        const int N = hp.n_embd; const int il = hp.n_layer; const Layer & L = m.layers[il];
        std::vector<float> cat(2 * N), x(N), y(N);
        rms_norm(emb, L.enorm, cat.data(), N, hp.eps); rms_norm(h_prev, L.hnorm, cat.data() + N, N, hp.eps);
        matvec(pool, L.eh_proj, cat.data(), x.data());
        rope_prepare(pos);
        rms_norm(x.data(), L.attn_norm, xn.data(), N, hp.eps);
        attention(L, st, il, xn.data(), y.data(), kv_only);
        if (kv_only) return;
        for (int i = 0; i < N; i++) x[i] += y[i];
        rms_norm(x.data(), L.post_norm, xn.data(), N, hp.eps);
        moe(L, xn.data(), y.data(), nullptr);
        for (int i = 0; i < N; i++) x[i] += y[i];
        rms_norm(x.data(), L.head_norm, h_out, N, hp.eps);
    }
};

} // namespace q36
