// q36 engine: hybrid GPU (dense blocks, hot experts) + CPU (cold experts) decode for Qwen3.6-35B-A3B on the Arc B580.
#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include "pack.hpp"
#include "gpu_prefill.hpp"
#include <algorithm>
#include <chrono>
#include <thread>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <map>
#include <string>

namespace q36 {
using q36g::Act; using q36g::NCMAX; using q36g::RING; using q36g::RMASK;

static inline double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------- CPU side: routed experts from the pack (exact float math)
struct CpuExperts {
    struct L { const uint8_t * base = nullptr; size_t rec = 0; uint32_t tg = 0, td = 0, rg = 0, rd = 0; };
    std::vector<L> layers; Pool & pool;
    std::vector<float> act, acts, eo;
    CpuExperts(const Pack & p, int n_layers, Pool & pl) : pool(pl) {
        layers.resize(n_layers);
        for (int il = 0; il < n_layers; il++) {
            const PEntry & e = p.get(std::to_string(il) + ".exp");
            layers[il] = { e.data, e.row_bytes, e.type >> 8, e.type & 0xFF, e.n_in >> 16, e.n_in & 0xFFFF };
        }
        act.resize(64 * 512); acts.resize(64 * 32); eo.resize((size_t) 64 * 2048);
    }
    // out[2048] += sum_u w[u] * expert(ids[u])(x) for n experts (n <= 64); x = normalized input (2048).
    // int8 activations (as on the GPU), threads split the rows of all experts
    std::vector<int8_t> aq; std::vector<float> ad, as_, as32;
    void run(int il, const float * x, const int * ids, const float * w, int n, float * out) {
        if (n == 0) return;
        const L & l = layers[il];
        int8_t xq[2048]; float xd[64], xs[128], xs32[64]; quant_act(x, 2048, xq, xd, xs); for (int i = 0; i < 64; i++) xs32[i] = xs[2 * i] + xs[2 * i + 1];
        if (aq.empty()) { aq.resize(64 * 512); ad.resize(64 * 16); as_.resize(64 * 32); as32.resize(64 * 16); }
        pool.pfor((int64_t) n * 512, [&](int64_t a, int64_t b) {
            for (int64_t r = a; r < b; r++) {
                const int u = (int) (r / 512), i = (int) (r % 512); const uint8_t * rec = l.base + (size_t) ids[u] * l.rec + (size_t) (2 * i) * l.rg;
                act[r] = silu(qrow(l.tg, rec, 2048, xq, xd, xs, xs32)) * qrow(l.tg, rec + l.rg, 2048, xq, xd, xs, xs32);
            }
        });
        for (int u = 0; u < n; u++) { quant_act(act.data() + (size_t) u * 512, 512, aq.data() + (size_t) u * 512, ad.data() + (size_t) u * 16, as_.data() + (size_t) u * 32);
            for (int i = 0; i < 16; i++) as32[(size_t) u * 16 + i] = as_[(size_t) u * 32 + 2 * i] + as_[(size_t) u * 32 + 2 * i + 1]; }
        pool.pfor((int64_t) n * 2048, [&](int64_t a, int64_t b) {
            for (int64_t r = a; r < b; r++) {
                const int u = (int) (r / 2048), i = (int) (r % 2048); const uint8_t * rec = l.base + (size_t) ids[u] * l.rec + (size_t) 1024 * l.rg + (size_t) i * l.rd;
                eo[r] = qrow(l.td, rec, 512, aq.data() + (size_t) u * 512, ad.data() + (size_t) u * 16, as_.data() + (size_t) u * 32, as32.data() + (size_t) u * 16);
            }
        });
        for (int u = 0; u < n; u++) { const float * e = eo.data() + (size_t) u * 2048; const float wu = w[u]; for (int i = 0; i < 2048; i++) out[i] += wu * e[i]; }
    }
    // several columns at once: misses (col[k], ids[k], w[k]), k < n <= 64, in column order. An expert needed by several columns
    // is read once (its rows stay in the cache while the columns are computed). out[c][2048] += ...
    std::vector<int8_t> xq_m; std::vector<float> xd_m, xs_m, xs32_m;
    void run_multi(int il, int nc, const float * X, const int * col, const int * ids, const float * w, int n, float * out) {
        if (n == 0) return;
        const L & l = layers[il];
        if (xq_m.empty()) { xq_m.resize(8 * 2048); xd_m.resize(8 * 64); xs_m.resize(8 * 128); xs32_m.resize(8 * 64); }
        if (aq.empty()) { aq.resize(64 * 512); ad.resize(64 * 16); as_.resize(64 * 32); as32.resize(64 * 16); }
        bool need[8] = { false }; for (int k = 0; k < n; k++) need[col[k]] = true;
        for (int c = 0; c < nc; c++) if (need[c]) { quant_act(X + (size_t) c * 2048, 2048, xq_m.data() + c * 2048, xd_m.data() + c * 64, xs_m.data() + c * 128);
            for (int i = 0; i < 64; i++) xs32_m[c * 64 + i] = xs_m[c * 128 + 2 * i] + xs_m[c * 128 + 2 * i + 1]; }
        // unique experts, each with the list of misses using it
        int ue[64], nu = 0, first[64], nxt[64]; for (int k = 0; k < n; k++) nxt[k] = -1;
        { int last[64]; for (int k = 0; k < n; k++) { int j = 0; while (j < nu && ue[j] != ids[k]) j++; if (j == nu) { ue[nu] = ids[k]; first[nu] = k; last[nu] = k; nu++; } else { nxt[last[j]] = k; last[j] = k; } } }
        pool.pfor((int64_t) nu * 512, [&](int64_t a, int64_t b) {
            for (int64_t r = a; r < b; r++) {
                const int j = (int) (r / 512), i = (int) (r % 512); const uint8_t * rec = l.base + (size_t) ue[j] * l.rec + (size_t) (2 * i) * l.rg;
                for (int k = first[j]; k >= 0; k = nxt[k]) { const int c = col[k];
                    const int8_t * xq = xq_m.data() + c * 2048; const float * xd = xd_m.data() + c * 64, * xs = xs_m.data() + c * 128, * x32 = xs32_m.data() + c * 64;
                    act[(size_t) k * 512 + i] = silu(qrow(l.tg, rec, 2048, xq, xd, xs, x32)) * qrow(l.tg, rec + l.rg, 2048, xq, xd, xs, x32); }
            }
        });
        for (int u = 0; u < n; u++) { quant_act(act.data() + (size_t) u * 512, 512, aq.data() + (size_t) u * 512, ad.data() + (size_t) u * 16, as_.data() + (size_t) u * 32);
            for (int i = 0; i < 16; i++) as32[(size_t) u * 16 + i] = as_[(size_t) u * 32 + 2 * i] + as_[(size_t) u * 32 + 2 * i + 1]; }
        pool.pfor((int64_t) nu * 2048, [&](int64_t a, int64_t b) {
            for (int64_t r = a; r < b; r++) {
                const int j = (int) (r / 2048), i = (int) (r % 2048); const uint8_t * rec = l.base + (size_t) ue[j] * l.rec + (size_t) 1024 * l.rg + (size_t) i * l.rd;
                for (int k = first[j]; k >= 0; k = nxt[k])
                    eo[(size_t) k * 2048 + i] = qrow(l.td, rec, 512, aq.data() + (size_t) k * 512, ad.data() + (size_t) k * 16, as_.data() + (size_t) k * 32, as32.data() + (size_t) k * 16);
            }
        });
        for (int k = 0; k < n; k++) { const float * e = eo.data() + (size_t) k * 2048; float * o = out + (size_t) col[k] * 2048; const float wk = w[k]; for (int i = 0; i < 2048; i++) o[i] += wk * e[i]; }
    }
};

// ---------------------------------------------------------------- VRAM expert cache
// Hot experts live in VRAM slots; a device table maps (layer, expert) -> record address (0 = not cached, CPU computes it).
// Scores are exponentially decayed use counts. At step boundaries (queue idle) the hottest uncached experts replace the
// coldest cached ones: the victim is unpublished first, the upload runs on a second queue, the new expert is published
// when its upload has finished. Kernels of one step therefore always see a consistent table.
struct ExpertCache {
    struct Cls { size_t rec = 0; std::vector<uint8_t *> chunks; int per_chunk = 0, cap = 0; std::vector<int> free_slots; std::vector<int> owner; };
    struct Pending { int key, cls, slot; };
    sycl::queue & q; sycl::queue qup; const CpuExperts & cpu; int nl;
    Cls cls[2]; std::vector<uint8_t> layer_cls; std::vector<int> slot_of; std::vector<uint64_t> tab_h; uint64_t * tab_d = nullptr;
    std::vector<float> score; std::vector<uint32_t> total; std::vector<uint8_t> is_pending;
    float decay = 0.9954f, margin = 1.25f; int rate = 4, rate_max = 40; double prior_tokens = 60.0; // decay: half-life 150 steps; uploads per step
    long hits = 0, misses = 0, uploads = 0; size_t bytes = 0; bool dirty = false;

    ExpertCache(sycl::queue & q_, const CpuExperts & c, int n_layers, double budget_gb) : q(q_), qup(q_.get_context(), q_.get_device()), cpu(c), nl(n_layers) {
        layer_cls.resize(nl); slot_of.assign((size_t) nl * 256, -1); tab_h.assign((size_t) nl * 256, 0); score.assign((size_t) nl * 256, 0.0f);
        total.assign((size_t) nl * 256, 0); is_pending.assign((size_t) nl * 256, 0);
        tab_d = sycl::malloc_device<uint64_t>((size_t) nl * 256, q); q.memcpy(tab_d, tab_h.data(), tab_h.size() * 8).wait();
        size_t rec[2] = { 0, 0 }; int cnt[2] = { 0, 0 };
        for (int il = 0; il < nl; il++) { const int k = c.layers[il].td == T_Q6_K ? 1 : 0; layer_cls[il] = (uint8_t) k; rec[k] = c.layers[il].rec; cnt[k]++; }
        const double all = (double) cnt[0] * rec[0] + (double) cnt[1] * rec[1];
        for (int k = 0; k < 2; k++) if (cnt[k]) {
            Cls & C = cls[k]; C.rec = rec[k]; C.per_chunk = 64;
            const int want = (int) (budget_gb * 1e9 * ((double) cnt[k] * rec[k] / all) / rec[k]);
            const int n_chunks = std::max(0, std::min(want, cnt[k] * 256) / C.per_chunk);
            for (int i = 0; i < n_chunks; i++) {
                uint8_t * p = sycl::malloc_device<uint8_t>(C.rec * C.per_chunk, q);
                if (!p) break;
                C.chunks.push_back(p); bytes += C.rec * C.per_chunk;
            }
            C.cap = (int) C.chunks.size() * C.per_chunk; C.owner.assign(C.cap, -1);
            for (int i = C.cap - 1; i >= 0; i--) C.free_slots.push_back(i);
        }
    }
    uint8_t * addr(int k, int slot) const { return cls[k].chunks[slot / cls[k].per_chunk] + (size_t) (slot % cls[k].per_chunk) * cls[k].rec; }
    const uint8_t * src(int key) const { const auto & l = cpu.layers[key >> 8]; return l.base + (size_t) (key & 255) * l.rec; }
    bool cached(int il, int e) const { return slot_of[(size_t) il * 256 + e] >= 0; }
    void touch(int il, int e, bool hit) { const size_t k = (size_t) il * 256 + e; score[k] += 1.0f; total[k]++; if (hit) hits++; else misses++; }
    void flush_table() { if (dirty) { q.memcpy(tab_d, tab_h.data(), tab_h.size() * 8); dirty = false; } }

    // fill the cache with the experts that have the highest score (prior profile); synchronous
    void prefill() {
        std::vector<int> ord((size_t) nl * 256); for (size_t i = 0; i < ord.size(); i++) ord[i] = (int) i;
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return score[a] > score[b]; });
        std::vector<sycl::event> evs;
        for (int key : ord) {
            Cls & C = cls[layer_cls[key >> 8]]; if (C.free_slots.empty()) continue;
            const int slot = C.free_slots.back(); C.free_slots.pop_back(); C.owner[slot] = key;
            evs.push_back(qup.memcpy(addr(layer_cls[key >> 8], slot), src(key), C.rec));
            slot_of[key] = slot; tab_h[key] = (uint64_t) addr(layer_cls[key >> 8], slot);
            if (evs.size() >= 64) { sycl::event::wait(evs); evs.clear(); }
        }
        sycl::event::wait(evs); dirty = true; flush_table(); q.wait();
    }

    // uploads run on a worker thread (the copy call itself costs ~0.2 ms of host time per expert)
    std::thread up_thread; std::mutex up_m; std::condition_variable up_cv; std::deque<Pending> up_jobs; std::vector<Pending> up_done; bool up_stop = false; int up_busy = 0;
    void up_worker() {
        for (;;) {
            Pending j;
            { std::unique_lock<std::mutex> lk(up_m); up_cv.wait(lk, [&] { return up_stop || !up_jobs.empty(); }); if (up_stop) return; j = up_jobs.front(); up_jobs.pop_front(); up_busy++; }
            qup.memcpy(addr(j.cls, j.slot), src(j.key), cls[j.cls].rec).wait();
            { std::lock_guard<std::mutex> lk(up_m); up_done.push_back(j); up_busy--; }
        }
    }
    ~ExpertCache() { if (up_thread.joinable()) { { std::lock_guard<std::mutex> lk(up_m); up_stop = true; } up_cv.notify_all(); up_thread.join(); } }

    // called between steps with the main queue idle (no kernel is running, so a victim slot can be reused right away:
    // the table update is queued in order before the kernels of the next step)
    void maintain(const std::vector<int> & missed) {
        if (!up_thread.joinable()) up_thread = std::thread([this] { up_worker(); });
        size_t n_pending;
        { std::lock_guard<std::mutex> lk(up_m);
          for (auto & p : up_done) { slot_of[p.key] = p.slot; tab_h[p.key] = (uint64_t) addr(p.cls, p.slot); is_pending[p.key] = 0; dirty = true; uploads++; }
          up_done.clear(); n_pending = up_jobs.size(); }
        for (auto & v : score) v *= decay;
        const int rate_now = rate > 0 ? std::min(rate_max, std::max(rate, (int) missed.size() / 3)) : 0;
        if (rate > 0 && (int) n_pending < 2 * rate_now && !missed.empty()) {
            std::vector<int> cand; cand.reserve(missed.size());
            for (int k : missed) if (slot_of[k] < 0 && !is_pending[k]) cand.push_back(k);
            std::sort(cand.begin(), cand.end()); cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
            std::sort(cand.begin(), cand.end(), [&](int a, int b) { return score[a] > score[b]; });
            int done = 0;
            for (int key : cand) {
                if (done >= rate_now) break;
                const int k = layer_cls[key >> 8]; Cls & C = cls[k]; int slot = -1;
                if (!C.free_slots.empty()) { slot = C.free_slots.back(); C.free_slots.pop_back(); }
                else {
                    int vs = -1; float vmin = 1e30f;
                    for (int sl = 0; sl < C.cap; sl++) { const int o = C.owner[sl]; if (o >= 0 && slot_of[o] == sl && score[o] < vmin) { vmin = score[o]; vs = sl; } }
                    if (vs < 0 || !(score[key] > vmin * margin + 0.5f)) continue;
                    const int victim = C.owner[vs]; slot_of[victim] = -1; tab_h[victim] = 0; dirty = true; slot = vs;
                }
                C.owner[slot] = key; is_pending[key] = 1; done++;
                { std::lock_guard<std::mutex> lk(up_m); up_jobs.push_back({ key, k, slot }); }
            }
            if (done) up_cv.notify_one();
        }
        flush_table();
    }
    // synchronous bulk update (main queue idle): bring the cached set close to the best-scored experts, at most max_swaps uploads
    int refill(int max_swaps) {
        int n = 0;
        for (int k = 0; k < 2; k++) {
            Cls & C = cls[k]; if (C.cap == 0) continue;
            std::vector<int> keys; for (int il = 0; il < nl; il++) if (layer_cls[il] == k) for (int e = 0; e < 256; e++) keys.push_back(il * 256 + e);
            std::sort(keys.begin(), keys.end(), [&](int a, int b) { return score[a] > score[b]; });
            std::vector<int> in, out; // wanted but not cached (best first), cached but not wanted (worst first)
            for (size_t i = 0; i < keys.size(); i++) { const int key = keys[i]; if ((int) i < C.cap) { if (slot_of[key] < 0 && !is_pending[key]) in.push_back(key); } else if (slot_of[key] >= 0) out.push_back(key); }
            std::reverse(out.begin(), out.end());
            size_t io = 0;
            for (int key : in) {
                if (n >= max_swaps) break;
                int slot = -1;
                if (!C.free_slots.empty()) { slot = C.free_slots.back(); C.free_slots.pop_back(); }
                else { if (io >= out.size() || !(score[key] > score[out[io]] * margin + 0.5f)) break; const int v = out[io++]; slot = slot_of[v]; slot_of[v] = -1; tab_h[v] = 0; }
                C.owner[slot] = key; q.memcpy(addr(k, slot), src(key), C.rec); slot_of[key] = slot; tab_h[key] = (uint64_t) addr(k, slot); n++; uploads++;
            }
        }
        if (n) { dirty = true; flush_table(); q.wait(); }
        return n;
    }
    // the layer-major prefill borrows the last n chunks of class 0 as its staging area (their experts are dropped)
    void settle() { // wait for the upload worker, apply what it finished
        for (;;) { { std::lock_guard<std::mutex> lk(up_m); if (up_jobs.empty() && up_busy == 0) break; } std::this_thread::sleep_for(std::chrono::microseconds(200)); }
        std::lock_guard<std::mutex> lk(up_m);
        for (auto & p : up_done) { slot_of[p.key] = p.slot; tab_h[p.key] = (uint64_t) addr(p.cls, p.slot); is_pending[p.key] = 0; dirty = true; uploads++; }
        up_done.clear();
    }
    int lent = 0;
    std::vector<uint8_t *> lend(int n) { // returns the chunk base addresses
        settle(); Cls & C = cls[0]; std::vector<uint8_t *> out;
        n = std::min(n, (int) C.chunks.size()); if (n <= 0) return out;
        const int cap = C.cap - n * C.per_chunk;
        for (int sl = cap; sl < C.cap; sl++) { const int o = C.owner[sl]; if (o >= 0) { if (slot_of[o] == sl) { slot_of[o] = -1; tab_h[o] = 0; dirty = true; } C.owner[sl] = -1; } }
        C.free_slots.erase(std::remove_if(C.free_slots.begin(), C.free_slots.end(), [&](int x) { return x >= cap; }), C.free_slots.end());
        C.cap = cap; lent = n; flush_table(); q.wait();
        for (int i = 0; i < n; i++) out.push_back(C.chunks[C.chunks.size() - n + i]);
        return out;
    }
    void give_back() { Cls & C = cls[0]; if (!lent) return; const int cap = C.cap + lent * C.per_chunk; for (int sl = cap - 1; sl >= C.cap; sl--) C.free_slots.push_back(sl); C.cap = cap; lent = 0; }
    double hit_rate() const { return hits + misses ? (double) hits / (hits + misses) : 0.0; }
    void save_profile(const std::string & path) const { FILE * f = fopen(path.c_str(), "wb"); if (f) { fwrite(total.data(), 4, total.size(), f); fclose(f); } }
    bool load_profile(const std::string & path) {
        FILE * f = fopen(path.c_str(), "rb"); if (!f) return false;
        std::vector<uint32_t> t(total.size()); const bool ok = fread(t.data(), 4, t.size(), f) == t.size(); fclose(f); if (!ok) return false;
        double sum = 0; for (auto v : t) sum += v; if (sum <= 0) return false;
        // the prior weighs like prior_tokens tokens of history
        const double sc = prior_tokens * nl * 8 / sum; for (size_t i = 0; i < t.size(); i++) { score[i] = (float) (t[i] * sc); total[i] = t[i]; }
        return true;
    }
};

// ---------------------------------------------------------------- GPU engine
struct Engine {
    const Pack & pk; Pool & pool; sycl::queue q; CpuExperts cpu; std::unique_ptr<ExpertCache> cache; std::vector<int> missed;
    int n_layer = 40, n_vocab = 0, n_ctx = 0; bool has_mtp = false; float eps = 1e-6f, rope_base = 1e7f; int n_rot = 64;

    struct LW { // device pointers per layer
        bool recr = false;
        uint8_t * w_in = nullptr, * w_out = nullptr, * sh_gu = nullptr, * sh_down = nullptr; float * ab = nullptr, * par = nullptr, * router = nullptr;
        float * S = nullptr, * YR = nullptr, * G = nullptr, * QKV = nullptr, * Dr = nullptr;  // GDN state + rings
        q36g::KV kv = { nullptr, nullptr, nullptr, nullptr };                                   // attention cache (int8)
    };
    std::vector<LW> L;
    uint8_t * head = nullptr; float * gpar = nullptr;
    // activations
    float * GU = nullptr, * X, * XN, * Y1, * AB, * O, * Y, * RL, * ACT, * Y2, * QS, * PART, * LOG, * shw_d, * w_d; int * ids_d; int8_t * QQ;
    Act a_x, a_o, a_act, a_xm;
    float * Xm, * XNm, * HT, * Hp, * HM, * HIN, * h_embm; uint8_t * eh_proj = nullptr; int * draft_rows_d = nullptr; int n_draft_rows = 0; std::vector<int> draft_rows_h;
    struct Stream { float * X; float * XN; Act a; }; Stream st_t, st_m;
    // layer-major prefill
    int pf_window = 0; size_t stage_rec = 0;
    float * Xw = nullptr, * HTw = nullptr, * h_embw = nullptr, * h_wtw = nullptr; int * h_idsw = nullptr; uint8_t * stage = nullptr;
    float * pY2, * pWT, * pSHW, * pROPE, * pGU, * pACTE, * cXN, * cY, * cYR, * cAB, * cG, * cQKV, * cO, * cY1, * cRL, * cQS; int8_t * cQQ; int * pID, * pPRM, * pGRP; uint64_t * pTAB;
    Act pAX, pAS, pAE, cAO; q36g::Tiles pT, pTG; q36g::AttT pAT; int8_t * pQN = nullptr; float * pQSN = nullptr; bool pf_oldatt = false; int * pGX = nullptr, * pCNT = nullptr; float * pYP = nullptr;
    // host USM
    float * h_emb, * h_xn, * h_miss, * h_w; int * h_ids; uint8_t * h_step, * d_step; // step params: int[16] then rope float[NCMAX][64]
    const PEntry * tok_embd = nullptr;
    int n_past = 0, s_pos = 0; size_t vram = 0;

    double t_gpu_wait = 0, t_cpu = 0, t_submit = 0, t_head = 0, t_maint = 0; long n_steps = 0;
    bool prof = false; std::map<std::string, std::pair<double, long>> kprof;
    static sycl::property_list qprops() { if (getenv("Q36_GPROF")) return { sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{} }; return { sycl::property::queue::in_order{} }; }
    // Q36_GPROF=1 with eager submission (--no-graph): device timestamps of every kernel; busy time vs span of each step
    bool gprof = getenv("Q36_GPROF") != nullptr; std::vector<std::pair<const char *, sycl::event>> gev; std::map<std::string, std::pair<double, long>> gsum; double g_busy = 0, g_span = 0; long g_steps = 0;
    void gprof_collect() {
        if (gev.empty()) return; uint64_t t0 = ~0ull, t1 = 0; double busy = 0;
        for (auto & pe : gev) { const uint64_t a = pe.second.get_profiling_info<sycl::info::event_profiling::command_start>(), b = pe.second.get_profiling_info<sycl::info::event_profiling::command_end>();
            t0 = std::min(t0, a); t1 = std::max(t1, b); busy += (b - a) * 1e-9; auto & r = gsum[pe.first]; r.first += (b - a) * 1e-9; r.second++; }
        g_busy += busy; g_span += (t1 - t0) * 1e-9; g_steps++; gev.clear();
    }
    void gprof_print(long tokens) { for (auto & kv : gsum) printf("  %-14s %7.1f us/call  %6.1f calls/tok  %7.3f ms/tok (device time)\n", kv.first.c_str(), kv.second.first / kv.second.second * 1e6, (double) kv.second.second / tokens, kv.second.first / tokens * 1e3);
        printf("  device busy %.3f ms/tok, span of the steps %.3f ms/tok (gaps %.3f)\n", g_busy / tokens * 1e3, g_span / tokens * 1e3, (g_span - g_busy) / tokens * 1e3); }
    template <class F> sycl::event K(const char * name, F && f) { // submit a kernel; in profiling mode wait for it and book the time
        if (gprof && !prof) { sycl::event e = f(); gev.push_back({ name, e }); return e; }
        if (!prof) return f();
        const double t0 = now_s(); sycl::event e = f(); e.wait(); auto & r = kprof[name]; r.first += now_s() - t0; r.second++; return e;
    }
    void print_prof(long tokens) { double tot = 0; for (auto & kv : kprof) tot += kv.second.first;
        for (auto & kv : kprof) printf("  %-14s %7.1f us/call  %6.0f calls/tok  %7.3f ms/tok\n", kv.first.c_str(), kv.second.first / kv.second.second * 1e6, (double) kv.second.second / tokens, kv.second.first / tokens * 1e3);
        printf("  total %.3f ms/tok (each call includes one host round trip)\n", tot / tokens * 1e3); }

    template <class T> T * dalloc(size_t n) { T * p = sycl::malloc_device<T>(n, q); if (!p) throw std::runtime_error("device alloc failed"); vram += n * sizeof(T); return p; }
    // prefill buffers live in chunks of the expert cache that are lent for the duration of a window
    std::vector<std::pair<void **, size_t>> pf_reqs; size_t pf_bytes = 0; int pf_nchunks = 0;
    template <class T> void palloc(T *& p, size_t n) { p = nullptr; pf_reqs.push_back({ (void **) &p, (n * sizeof(T) + 255) / 256 * 256 }); pf_bytes += pf_reqs.back().second; }
    void pact(Act & a, size_t vecs, int n) { palloc(a.q, vecs * n); palloc(a.d, vecs * (n / 32)); palloc(a.s, vecs * (n / 16)); }
    int pf_chunks_needed(size_t chunk_bytes) const { int k = 1; size_t off = 0; for (auto & r : pf_reqs) { if (off + r.second > chunk_bytes) { k++; off = 0; } off += r.second; } return k; }
    void pf_carve(const std::vector<uint8_t *> & ch, size_t chunk_bytes) { size_t k = 0, off = 0; for (auto & r : pf_reqs) { if (off + r.second > chunk_bytes) { k++; off = 0; } *r.first = ch[k] + off; off += r.second; } }
    template <class T> T * halloc(size_t n) { T * p = sycl::malloc_host<T>(n, q); if (!p) throw std::runtime_error("host USM alloc failed"); memset(p, 0, n * sizeof(T)); return p; }
    template <class T> T * upload(const PEntry & e) { T * p = (T *) dalloc<uint8_t>(e.size); q.memcpy(p, e.data, e.size); return p; }
    Act act_alloc(size_t vecs, int n) { Act a; a.q = dalloc<int8_t>(vecs * n); a.d = dalloc<float>(vecs * (n / 32)); a.s = dalloc<float>(vecs * (n / 16)); return a; }

    Engine(const Pack & p, Pool & pl, int ctx, double cache_gb = 0.0, int prefill_window = 0) : pf_window(prefill_window), pk(p), pool(pl), q(sycl::gpu_selector_v, qprops()), cpu(p, p.hpi("n_layer") + p.hpi("has_mtp"), pl) {
        n_layer = p.hpi("n_layer"); n_vocab = p.hpi("n_vocab"); has_mtp = p.hpi("has_mtp") != 0; n_ctx = ctx;
        eps = (float) p.hp.at("eps"); rope_base = (float) p.hp.at("rope_base"); n_rot = p.hpi("n_rot");
        const int nl = n_layer + (has_mtp ? 1 : 0); L.resize(nl);
        for (int il = 0; il < nl; il++) {
            LW & l = L[il]; const std::string s = std::to_string(il) + "."; l.recr = p.has(s + "qkvz");
            l.w_in = upload<uint8_t>(p.get(s + (l.recr ? "qkvz" : "qkv"))); l.w_out = upload<uint8_t>(p.get(s + "out"));
            l.par = upload<float>(p.get(s + "params")); l.router = upload<float>(p.get(s + "router"));
            l.sh_gu = upload<uint8_t>(p.get(s + "sh_gu")); l.sh_down = upload<uint8_t>(p.get(s + "sh_down"));
            if (l.recr) {
                l.ab = upload<float>(p.get(s + "ab"));
                l.S = dalloc<float>((size_t) 32 * 128 * 128); l.YR = dalloc<float>((size_t) RING * 12288); l.G = dalloc<float>(RING * 64);
                l.QKV = dalloc<float>((size_t) RING * 8192); l.Dr = dalloc<float>((size_t) RING * 4096);
            } else { l.kv.kq = dalloc<int8_t>((size_t) n_ctx * 512); l.kv.ks = dalloc<float>((size_t) n_ctx * 32); l.kv.vq = dalloc<int8_t>((size_t) n_ctx * 512); l.kv.vs = dalloc<float>((size_t) n_ctx * 16); }
        }
        head = upload<uint8_t>(p.get("head")); gpar = upload<float>(p.get("g.params")); tok_embd = &p.get("tok_embd");
        X = dalloc<float>(NCMAX * 2048); XN = dalloc<float>(NCMAX * 2048); Y1 = dalloc<float>(NCMAX * 9216); AB = dalloc<float>(NCMAX * 64);
        O = dalloc<float>(NCMAX * 4096); Y = dalloc<float>(NCMAX * 2048); RL = dalloc<float>(NCMAX * 272); ACT = dalloc<float>(NCMAX * 9 * 512);
        Y2 = dalloc<float>(NCMAX * 2048); GU = dalloc<float>(NCMAX * 1024); QQ = dalloc<int8_t>(NCMAX * 16 * 256); QS = dalloc<float>(NCMAX * 16 * 16);
        PART = dalloc<float>((size_t) NCMAX * 16 * std::min(q36g::ATT_NCH_MAX, (n_ctx + 15) / 16 + 1) * 272); LOG = dalloc<float>((size_t) NCMAX * n_vocab);
        shw_d = dalloc<float>(NCMAX); w_d = dalloc<float>(NCMAX * 8); ids_d = dalloc<int>(NCMAX * 8);
        a_x = act_alloc(NCMAX, 2048); a_o = act_alloc(NCMAX, 4096); a_act = act_alloc(NCMAX * 9, 512); a_xm = act_alloc(NCMAX, 2048);
        Xm = dalloc<float>(NCMAX * 2048); XNm = dalloc<float>(NCMAX * 2048); HT = dalloc<float>(NCMAX * 2048); Hp = dalloc<float>(2048); HM = dalloc<float>(NCMAX * 2048);
        HIN = dalloc<float>(NCMAX * 2048); h_embm = halloc<float>(NCMAX * 2048);
        st_t = { X, XN, a_x }; st_m = { Xm, XNm, a_xm };
        if (has_mtp) eh_proj = upload<uint8_t>(p.get(std::to_string(n_layer) + ".eh_proj"));
        if (pf_window > 0) { // layer-major prefill: the device buffers are carved out of expert-cache chunks lent for the window (see pf_carve)
            pf_window = std::min(pf_window, 8192) / 8 * 8;
            for (auto & cl : cpu.layers) stage_rec = std::max(stage_rec, cl.rec);
            const size_t W = pf_window, C = PFC;
            palloc(Xw, W * 2048); palloc(HTw, (W + 1) * 2048);
            h_embw = halloc<float>(W * 2048); h_idsw = halloc<int>(W * 8 * (size_t) n_layer); h_wtw = halloc<float>(W * 8);
            palloc(pY2, W * 2048); palloc(pWT, W * 8); palloc(pSHW, W); palloc(pROPE, W * 64); palloc(pGU, (size_t) PFM * 1024); palloc(pACTE, (size_t) PFM * 8 * 512);
            palloc(pID, W * 8); palloc(pPRM, (W / 8 + W / C + 8) * 16); palloc(pTAB, (size_t) nl * 256);
            pact(pAX, W, 2048); pact(pAS, PFM, 512); pact(pAE, (size_t) PFM * 8, 512); pact(cAO, C, 4096); palloc(pT.q, (size_t) PFM * 4096); pf_pin = halloc<uint8_t>(stage_rec * PFPIN);
            { const int nt = (n_ctx + 15) / 16 + 1; pAT.ntile = nt; palloc(pAT.kt, (size_t) 2 * nt * 8 * 512); palloc(pAT.ks, (size_t) 2 * nt * 8 * 16);
              palloc(pAT.vt, (size_t) 2 * nt * 16 * 256); palloc(pAT.vs, (size_t) 2 * nt * 8 * 16);
              palloc(pQN, (size_t) PFC * 16 * 256); palloc(pQSN, (size_t) PFC * 16 * 8); pf_oldatt = getenv("Q36_OLDATT") != nullptr; }
            { const size_t ngx = (size_t) PFM / 2 + 256; palloc(pTG.q, ngx * 64 * 512); palloc(pTG.d, ngx * 64 * 16); palloc(pTG.s, ngx * 64 * 32);
              palloc(pGX, (ngx + 1) * 32); palloc(pCNT, 512); palloc(pYP, (size_t) PFM * 8 * 2048); } palloc(pT.d, (size_t) PFM * 128);
            palloc(cXN, C * 2048); palloc(cY, C * 2048); palloc(cYR, (C + 3) * 12288); palloc(cAB, C * 64); palloc(cG, C * 64);
            palloc(cQKV, C * 8192); palloc(cO, C * 4096); palloc(cY1, C * 9216); palloc(cRL, C * 272); palloc(cQS, C * 256); palloc(cQQ, C * 4096);
            pf_prof = getenv("Q36_PFPROF") != nullptr; pf_kprof = getenv("Q36_PFKPROF") != nullptr;
        }
        h_emb = halloc<float>(NCMAX * 2048); h_xn = halloc<float>(NCMAX * 2048); h_miss = halloc<float>(NCMAX * 2048); h_step = halloc<uint8_t>(64 + NCMAX * 64 * 4); d_step = dalloc<uint8_t>(64 + NCMAX * 64 * 4);
        h_w = halloc<float>(NCMAX * 8); h_ids = halloc<int>(NCMAX * 8);
        q.wait(); reset();
        // cache_gb < 0: -cache_gb is the total VRAM budget of the engine, the cache gets what the rest leaves
        if (cache_gb < 0) cache_gb = std::max(0.0, -cache_gb - vram / 1e9 - 0.1);
        cache.reset(new ExpertCache(q, cpu, nl, cache_gb)); graphs.resize(nl);
        if (pf_window > 0) { const size_t cb = cache->cls[0].rec * cache->cls[0].per_chunk; pf_nchunks = cb ? pf_chunks_needed(cb) : 0;
            if (!cb || (int) cache->cls[0].chunks.size() < PFLEND + pf_nchunks + 2) pf_window = 0; } // expert cache too small to lend the prefill its memory
    }

    void reset() {
        for (auto & l : L) if (l.recr) {
            q.fill(l.S, 0.0f, (size_t) 32 * 128 * 128); q.fill(l.YR, 0.0f, (size_t) RING * 12288); q.fill(l.G, 0.0f, RING * 64);
            q.fill(l.QKV, 0.0f, (size_t) RING * 8192); q.fill(l.Dr, 0.0f, (size_t) RING * 4096);
        }
        q.fill(Hp, 0.0f, 2048); q.wait(); n_past = 0; s_pos = 0;
    }

    void embed(int tok, float * out) const { // Q8_0 row-SoA -> f32
        const uint8_t * w = tok_embd->data + (size_t) tok * tok_embd->row_bytes;
        for (int b = 0; b < 64; b++) { uint16_t dh; memcpy(&dh, w + 2048 + 2 * b, 2); const float d = h2f(dh); for (int i = 0; i < 32; i++) out[b * 32 + i] = d * (int8_t) w[b * 32 + i]; }
    }

    sycl::event gemv_f32(int nc, const float * W, int rows, const float * x, float * y, int ystride) {
        switch (nc) {
            case 1: return q36g::k_gemv_f32n<1>(q, W, rows, 2048, x, y, ystride); case 2: return q36g::k_gemv_f32n<2>(q, W, rows, 2048, x, y, ystride);
            case 3: return q36g::k_gemv_f32n<3>(q, W, rows, 2048, x, y, ystride); case 4: return q36g::k_gemv_f32n<4>(q, W, rows, 2048, x, y, ystride);
            case 5: return q36g::k_gemv_f32n<5>(q, W, rows, 2048, x, y, ystride); case 6: return q36g::k_gemv_f32n<6>(q, W, rows, 2048, x, y, ystride);
            case 7: return q36g::k_gemv_f32n<7>(q, W, rows, 2048, x, y, ystride); default: return q36g::k_gemv_f32n<8>(q, W, rows, 2048, x, y, ystride);
        }
    }
    template <class... A> sycl::event gemv_q8(int nc, A &&... a) {
        switch (nc) {
            case 1: return q36g::k_gemv_q8<1>(q, a...); case 2: return q36g::k_gemv_q8<2>(q, a...); case 3: return q36g::k_gemv_q8<3>(q, a...);
            case 4: return q36g::k_gemv_q8<4>(q, a...); case 5: return q36g::k_gemv_q8<5>(q, a...); case 6: return q36g::k_gemv_q8<6>(q, a...);
            case 7: return q36g::k_gemv_q8<7>(q, a...); default: return q36g::k_gemv_q8<8>(q, a...);
        }
    }

    // part A of layer il: residual add + norm, attention / delta-net block, post norm, router + top-k (the host waits for this).
    // il == n_layer is the MTP block: its input is eh_proj([rms(emb) ; rms(h)]) and it runs on its own residual stream.
    sycl::event sub_A(int il, int nc, int nchb, bool kv_only = false) {
        const LW & l = L[il]; const int * prm = (const int *) d_step; const float * rope = (const float *) (d_step + 64);
        const bool mtp = il == n_layer; const Stream & st = mtp ? st_m : st_t; float * X = st.X; float * XN = st.XN; const Act a_x = st.a;
        if (mtp) {
            K("mtp_in", [&] { return q36g::k_mtp_in(q, nc, h_embm, HIN, l.par + POff::enorm, l.par + POff::hnorm, eps, a_o); });
            K("gemv_ehproj", [&] { return gemv_q8(nc, eh_proj, 2048, 4096, a_o, X, prm, 0, 0x7fffffff, 2048); });
            K("norm_quant", [&] { return q36g::k_norm_quant(q, nc, X, nullptr, nullptr, false, l.par + POff::attn_norm, eps, XN, nullptr, a_x); });
        }
        else if (il == 0) K("norm_quant", [&] { return q36g::k_norm_quant(q, nc, X, h_emb, nullptr, true, l.par + POff::attn_norm, eps, XN, nullptr, a_x); });
        else              K("norm_quant", [&] { return q36g::k_norm_quant(q, nc, X, Y2, h_miss, false, l.par + POff::attn_norm, eps, XN, nullptr, a_x); });
        if (l.recr) {
            K("gemv_qkvz", [&] { return gemv_q8(nc, l.w_in, 12288, 2048, a_x, l.YR, prm, 1, RMASK, 12288); });
            K("ab", [&] { return gemv_f32(nc, l.ab, 64, XN, AB, 64); });
            K("gdn_gates", [&] { return q36g::k_gdn_gates(q, nc, prm, AB, l.par + POff::dt_bias, l.par + POff::ssm_a, l.G); });
            K("gdn_conv", [&] { return q36g::k_gdn_conv(q, nc, prm, l.YR, l.par + POff::conv, eps, l.QKV); });
            K("gdn_core", [&] { return q36g::k_gdn_core(q, nc, prm, l.S, l.QKV, l.G, l.Dr, O); });
            K("gdn_post", [&] { return q36g::k_gdn_post(q, nc, prm, O, l.YR, l.par + POff::ssm_norm, eps, a_o); });
        } else {
            K("gemv_qkv_att", [&] { return gemv_q8(nc, l.w_in, 9216, 2048, a_x, Y1, prm, 0, 0x7fffffff, 9216); });
            sycl::event ep = K("att_prep", [&] { return q36g::k_att_prep(q, nc, prm, Y1, l.par + POff::q_norm, l.par + POff::k_norm, eps, rope, QQ, QS, l.kv); });
            if (kv_only) return ep;
            K("att", [&] { return q36g::k_att(q, nc, prm, nchb, QQ, QS, l.kv, PART); });
            K("att_fin", [&] { return q36g::k_att_fin(q, nc, prm, nchb, PART, Y1, a_o); });
        }
        K("gemv_out", [&] { return gemv_q8(nc, l.w_out, 2048, 4096, a_o, Y, prm, 0, 0x7fffffff, 2048); });
        K("norm_quant", [&] { return q36g::k_norm_quant(q, nc, X, Y, nullptr, false, l.par + POff::post_norm, eps, XN, h_xn, a_x); });
        K("router", [&] { return gemv_f32(nc, l.router, 257, XN, RL, 272); });
        return K("topk", [&] { return q36g::k_topk(q, nc, RL, ids_d, w_d, shw_d, h_ids, h_w); });
    }
    // part B: shared expert + the experts that are cached in VRAM -> Y2
    sycl::event sub_B(int il, int nc) {
        const LW & l = L[il]; const uint64_t * tab = cache->tab_d + (size_t) il * 256; const Act a_x = (il == n_layer ? st_m : st_t).a;
        K("sh_gu", [&] { return gemv_q8(nc, l.sh_gu, 1024, 2048, a_x, GU, (const int *) d_step, 0, 0x7fffffff, 1024); });
        K("exp_glu", [&] { return q36g::k_exp_glu(q, nc, tab, ids_d, a_x, ACT); });
        K("act_quant", [&] { return q36g::k_act_quant(q, nc, 9, 9, ACT, a_act, GU); });
        if (cache->layer_cls[il]) return K("down", [&] { return q36g::k_down<true>(q, l.sh_down, nc, tab, ids_d, w_d, shw_d, a_act, Y2); });
        return K("down", [&] { return q36g::k_down<false>(q, l.sh_down, nc, tab, ids_d, w_d, shw_d, a_act, Y2); });
    }

    typedef sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable> XGraph;
    struct LGraph { std::unique_ptr<XGraph> A[NCMAX + 1], B[NCMAX + 1], KVO[NCMAX + 1]; int bucket[NCMAX + 1] = { 0 }; };
    std::vector<LGraph> graphs; bool use_graph = true; long n_recorded = 0;
    template <class F> XGraph & recorded(std::unique_ptr<XGraph> & slot, F && submit) {
        if (!slot) {
            sycl::ext::oneapi::experimental::command_graph g(q);
            g.begin_recording(q); submit(); g.end_recording();
            slot.reset(new XGraph(g.finalize())); n_recorded++;
        }
        return *slot;
    }

    // CPU side of the MoE of layer il: the experts that are not in VRAM -> h_miss
    void cpu_moe(int il, int nc) {
        int mc[64], mi[64]; float mw[64]; int nm = 0;
        memset(h_miss, 0, sizeof(float) * 2048 * (size_t) nc);
        for (int c = 0; c < nc; c++)
            for (int u = 0; u < 8; u++) {
                const int e = h_ids[c * 8 + u]; const bool hit = cache->cached(il, e); cache->touch(il, e, hit);
                if (!hit) { mc[nm] = c; mi[nm] = e; mw[nm] = h_w[c * 8 + u]; nm++; missed.push_back(il * 256 + e); }
            }
        cpu.run_multi(il, nc, h_xn, mc, mi, mw, nm, h_miss);
    }

    // per-step parameters for the kernels: first position, GDN state position, attention chunking, window column, RoPE tables
    void fill_step(uint8_t * dst, int P, int nc, int spos, int woff, int & nchb) const {
        int cht = 16; while ((P + nc + cht - 1) / cht > 256 && cht < q36g::ATT_CH_MAX) cht *= 2; while ((P + nc + cht - 1) / cht > q36g::ATT_NCH_MAX) cht *= 2; // keys per attention thread
        const int nch = (P + nc + cht - 1) / cht; nchb = 4; while (nchb < nch) nchb *= 2;
        int * hp = (int *) dst; hp[0] = P; hp[1] = spos; hp[2] = nch; hp[3] = cht; hp[4] = 0; hp[5] = woff; float * hr = (float *) (dst + 64);
        const float theta_scale = powf(rope_base, -2.0f / n_rot);
        for (int c = 0; c < nc; c++) { float th = (float) (P + c); for (int i = 0; i < n_rot / 2; i++) { hr[c * 64 + i] = cosf(th); hr[c * 64 + 32 + i] = sinf(th); th *= theta_scale; } }
    }
    void set_step(int P, int nc, int & nchb) { fill_step(h_step, P, nc, s_pos, 0, nchb); q.memcpy(d_step, h_step, 64 + sizeof(float) * 64 * (size_t) nc); }
    // run layer il (A, B, CPU experts); leaves Y2 (GPU) + h_miss (CPU) to be added by the next norm
    void run_layer(int il, int nc, int nchb, double & ts) {
        const bool gr = use_graph && !prof; sycl::event ea;
        if (gr) {
            LGraph & g = graphs[il];
            if (!L[il].recr && g.bucket[nc] != nchb) { g.A[nc].reset(); g.KVO[nc].reset(); g.bucket[nc] = nchb; }
            ea = q.ext_oneapi_graph(recorded(g.A[nc], [&] { sub_A(il, nc, nchb); }));
            q.ext_oneapi_graph(recorded(g.B[nc], [&] { sub_B(il, nc); }));
        } else { ea = sub_A(il, nc, nchb); sub_B(il, nc); }
        double t0 = now_s(); t_submit += t0 - ts;
        ea.wait();
        double t1 = now_s(); t_gpu_wait += t1 - t0;
        cpu_moe(il, nc);
        ts = now_s(); t_cpu += ts - t1;
    }
    template <class... A> sycl::event head_k(int nc, A &&... a) {
        switch (nc) {
            case 1: return q36g::k_head_q6k<1>(q, a...); case 2: return q36g::k_head_q6k<2>(q, a...); case 3: return q36g::k_head_q6k<3>(q, a...);
            case 4: return q36g::k_head_q6k<4>(q, a...); case 5: return q36g::k_head_q6k<5>(q, a...); case 6: return q36g::k_head_q6k<6>(q, a...);
            case 7: return q36g::k_head_q6k<7>(q, a...); default: return q36g::k_head_q6k<8>(q, a...);
        }
    }
    void head_logits(int nc, Act a, float * logits_host, bool draft_vocab) {
        double t0 = now_s();
        if (draft_vocab && draft_rows_d) {
            K("head_draft", [&] { return head_k(nc, head, n_draft_rows, a, LOG, n_vocab, draft_rows_d); });
            q.memcpy(logits_host, LOG, sizeof(float) * n_draft_rows).wait();
        } else {
            K("head_q6k", [&] { return head_k(nc, head, n_vocab, a, LOG, n_vocab, nullptr); });
            q.memcpy(logits_host, LOG, sizeof(float) * (size_t) nc * n_vocab).wait();
        }
        t_head += now_s() - t0;
    }
    // MTP drafts over a row subset of the LM head: the first n_keep token ids plus the ids listed in a text file
    void set_draft_vocab(int n_keep, const std::string & extra_file, const std::vector<int> & extra_ids = {}) {
        std::vector<uint8_t> in(n_vocab, 0); for (int i = 0; i < std::min(n_keep, n_vocab); i++) in[i] = 1;
        for (int id : extra_ids) if (id >= 0 && id < n_vocab) in[id] = 1;
        if (!extra_file.empty()) { FILE * f = fopen(extra_file.c_str(), "rb"); if (f) { int id; while (fscanf(f, "%d", &id) == 1) { if (id >= 0 && id < n_vocab) in[id] = 1; int ch; while ((ch = fgetc(f)) != EOF && (ch == ',' || ch == ' ' || ch == '\n' || ch == '\r')) {} if (ch != EOF) ungetc(ch, f); } fclose(f); } }
        draft_rows_h.clear(); for (int i = 0; i < n_vocab; i++) if (in[i]) draft_rows_h.push_back(i);
        n_draft_rows = (int) draft_rows_h.size(); draft_rows_d = dalloc<int>(n_draft_rows); q.memcpy(draft_rows_d, draft_rows_h.data(), sizeof(int) * (size_t) n_draft_rows).wait();
    }

    // one trunk step over nc tokens at positions n_past .. n_past+nc-1 (embeddings in h_emb). The normalized final hidden
    // states go to HT[c]. Nothing is committed: call commit(n) with the number of positions to keep (GDN state and K/V
    // caches roll back by themselves: speculative positions are simply overwritten).
    int step_P = 0;
    void step(int nc, bool want_logits, float * logits_host) {
        const int P = n_past; if (P + nc > n_ctx) throw std::runtime_error("context full");
        double ts = now_s(); int nchb; set_step(P, nc, nchb);
        for (int il = 0; il < n_layer; il++) run_layer(il, nc, nchb, ts);
        K("norm_quant", [&] { return q36g::k_norm_quant(q, nc, X, Y2, h_miss, false, gpar, eps, HT, nullptr, a_x); });
        t_submit += now_s() - ts;
        if (want_logits) head_logits(nc, a_x, logits_host, false); else q.wait();
        step_P = P; n_steps++; if (gprof) { q.wait(); gprof_collect(); }
        { const double tm = now_s(); cache->maintain(missed); missed.clear(); t_maint += now_s() - tm; }
    }
    void commit(int n_keep) { s_pos = step_P; n_past = step_P + n_keep; }

    // MTP block over nc tokens at positions P..P+nc-1: token embeddings in h_embm, the hidden states that predicted them in HIN.
    // kv_only: just append K/V of the MTP attention (catch-up with the true hidden states). Otherwise the output hidden
    // states go to HM[c] and, if asked, logits of column nc-1... of all columns.
    void mtp_step(int nc, int P, bool kv_only, float * logits_host) {
        const int il = n_layer; double ts = now_s(); int nchb; set_step(P, nc, nchb);
        if (kv_only) {
            LGraph & g = graphs[il]; const bool gr = use_graph && !prof;
            if (g.bucket[nc] != nchb) { g.A[nc].reset(); g.KVO[nc].reset(); g.bucket[nc] = nchb; }
            if (gr) q.ext_oneapi_graph(recorded(g.KVO[nc], [&] { sub_A(il, nc, nchb, true); })); else sub_A(il, nc, nchb, true);
            t_submit += now_s() - ts; return;
        }
        run_layer(il, nc, nchb, ts);
        K("norm_quant", [&] { return q36g::k_norm_quant(q, nc, Xm, Y2, h_miss, false, L[il].par + POff::head_norm, eps, HM, nullptr, a_xm); });
        t_submit += now_s() - ts;
        if (logits_host) head_logits(nc, a_xm, logits_host, true); else q.wait();
        { const double tm = now_s(); cache->maintain(missed); missed.clear(); t_maint += now_s() - tm; }
    }

    // ---- layer-major prefill ------------------------------------------------------------------------------------------
    // A window of W tokens (multiple of 8) goes through the model layer by layer. Inside a layer:
    //   1. the mixer (delta-net / attention), post norm and router run over chunks of PFC positions with chunk-wide kernels;
    //      meanwhile the experts of the layer that are not in the VRAM cache are uploaded to the staging area;
    //   2. the host groups the (token, slot) pairs by expert; the MoE runs expert-major: an expert's weights are read
    //      once and applied to all of its tokens.
    // The residual stream of the window lives in Xw, the MoE output in pY2, the final hidden states in HTw.
    static constexpr int PFC = 256, PFM = 512, PFPIN = 16, PFLEND = 8; uint8_t * pf_pin = nullptr; // PFLEND chunks of 64 experts = the staging ring (512 records)
    std::mutex pf_m; std::condition_variable pf_cv; int pf_moe_sub = 0, pf_up_ready = 0; bool pf_abort = false; std::vector<sycl::event> pf_moe_ev; std::vector<std::vector<sycl::event>> pf_up_ev; std::vector<int> pf_slot_user;
    std::vector<int> pf_hgrp, pf_hgrpx; std::vector<uint64_t> pf_rows; bool pf_prof = false, pf_kprof = false;
    template <class F> void prefill_window(const int * toks, int W, bool mtp, F && progress) {
        using namespace q36g; double pk_t = now_s(); auto PKW = [&](const char * nm) { if (!pf_kprof) return; q.wait(); const double t = now_s(); auto & r = kprof[nm]; r.first += t - pk_t; r.second++; pk_t = t; };
        const int P0 = n_past; if (P0 + W > n_ctx) throw std::runtime_error("context full");
        const double t0 = now_s();
        // staging area: the last PFLEND chunks of the expert cache (lent for the window). Ring of units of one class-0 record;
        // a Q6_K expert (bigger record) takes two neighbouring units of the same chunk.
        std::vector<uint8_t *> lch = cache->lend(PFLEND + pf_nchunks);
        if ((int) lch.size() < PFLEND + pf_nchunks) { cache->give_back(); throw std::runtime_error("expert cache too small for the prefill window"); }
        { const size_t cb = cache->cls[0].rec * cache->cls[0].per_chunk; std::vector<uint8_t *> bc(lch.begin() + PFLEND, lch.end()); pf_carve(bc, cb); lch.resize(PFLEND); }
        struct GiveBack { ExpertCache * c; ~GiveBack() { c->give_back(); } } give_back{ cache.get() };
        const size_t urec = cache->cls[0].rec; const int upc = cache->cls[0].per_chunk, R = (int) lch.size() * upc;
        auto unit = [&](int u) { return lch[u / upc] + (size_t) (u % upc) * urec; };
        pool.pfor(W, [&](int64_t a, int64_t b) { for (int64_t i = a; i < b; i++) embed(toks[i], h_embw + (size_t) i * 2048); });
        // kernel parameters: [0] flush of the pending delta-net records, then one entry per chunk and per attention call
        std::vector<int> hp; auto addp = [&](int P, int sp, int nch, int cht) { const int i = (int) hp.size() / 16; hp.resize(hp.size() + 16, 0); hp[i * 16] = P; hp[i * 16 + 1] = sp; hp[i * 16 + 2] = nch; hp[i * 16 + 3] = cht; return i; };
        struct Att { int c0, nc, prm, nch; }; std::vector<Att> att; std::vector<int> chunk_prm, chunk_att;
        addp(P0, s_pos, 0, 0);
        const int partcap = NCMAX * std::min(ATT_NCH_MAX, (n_ctx + 15) / 16 + 1);
        for (int c0 = 0; c0 < W; c0 += PFC) {
            const int n = std::min(PFC, W - c0); chunk_prm.push_back(addp(P0 + c0, 0, 0, 0)); chunk_att.push_back((int) att.size());
            for (int a0 = 0; a0 < n;) { // attention calls: as many queries as the buffer of partial results takes
                const int P = P0 + c0 + a0; int nc = n - a0;
                while (nc > 8 && nc * ((P + nc + 63) / 64) > partcap) nc = std::max(8, (nc / 2) & ~7);
                const int nch = (P + nc + 63) / 64;
                att.push_back({ c0 + a0, nc, addp(P, 0, nch, 64), nch }); a0 += nc;
            }
        }
        chunk_att.push_back((int) att.size());
        std::vector<float> hr((size_t) W * 64);
        { const float theta_scale = powf(rope_base, -2.0f / n_rot);
          for (int c = 0; c < W; c++) { float th = (float) (P0 + c); for (int i = 0; i < n_rot / 2; i++) { hr[(size_t) c * 64 + i] = cosf(th); hr[(size_t) c * 64 + 32 + i] = sinf(th); th *= theta_scale; } } }
        q.memcpy(pPRM, hp.data(), hp.size() * sizeof(int)); q.memcpy(pROPE, hr.data(), hr.size() * sizeof(float)); q.wait();
        { const float d = powf(cache->decay, (float) (W / 8)); for (auto & v : cache->score) v *= d; }
        const int total = n_layer + (mtp && has_mtp ? 1 : 0); const int nchunk = (int) chunk_prm.size();
        struct UpJob { int il, e, u, nu; }; std::vector<UpJob> jobs; std::vector<int> layer_end(n_layer);
        pf_rows.resize((size_t) n_layer * 256);
        { int head = 0;
          for (int il = 0; il < n_layer; il++) { const int nu = cpu.layers[il].rec > urec ? 2 : 1; int used = 0;
              for (int e = 0; e < 256; e++) { const size_t key = (size_t) il * 256 + e; pf_rows[key] = cache->tab_h[key];
                  if (cache->slot_of[key] >= 0) continue;
                  if (nu == 2 && (head & 1)) { head = (head + 1) % R; used++; }
                  pf_rows[key] = (uint64_t) unit(head); jobs.push_back({ il, e, head, nu }); head = (head + nu) % R; used += nu; }
              if (used > R) throw std::runtime_error("prefill staging area too small");
              layer_end[il] = (int) jobs.size(); } }
        q.memcpy(pTAB, pf_rows.data(), pf_rows.size() * 8);
        // the uploader streams them through pinned bounce buffers; a slot is reused once the MoE of its previous layer is done
        pf_moe_ev.assign(n_layer, sycl::event()); pf_up_ev.assign(n_layer, {}); pf_moe_sub = 0; pf_up_ready = 0; pf_abort = false; pf_slot_user.assign(R, -1);
        std::thread upl([&] {
            int pin_head = 0; std::vector<sycl::event> pin_ev(PFPIN); size_t jx = 0;
            for (int il = 0; il < n_layer; il++) {
                const auto & cl = cpu.layers[il]; std::vector<sycl::event> ev;
                for (; jx < (size_t) layer_end[il]; jx++) {
                    const UpJob & jb = jobs[jx]; std::vector<sycl::event> dep;
                    for (int k = 0; k < jb.nu; k++) { const int user = pf_slot_user[jb.u + k];
                        if (user >= 0) { std::unique_lock<std::mutex> lk(pf_m); pf_cv.wait(lk, [&] { return pf_moe_sub > user || pf_abort; }); if (pf_abort) return; dep.push_back(pf_moe_ev[user]); }
                        pf_slot_user[jb.u + k] = il; }
                    const int ps = pin_head; pin_head = (pin_head + 1) % PFPIN; pin_ev[ps].wait();
                    uint8_t * pb = pf_pin + (size_t) ps * stage_rec; memcpy(pb, cl.base + (size_t) jb.e * cl.rec, cl.rec);
                    pin_ev[ps] = cache->qup.memcpy(unit(jb.u), pb, cl.rec, dep); ev.push_back(pin_ev[ps]);
                }
                { std::lock_guard<std::mutex> lk(pf_m); pf_up_ev[il] = std::move(ev); pf_up_ready = il + 1; } pf_cv.notify_all();
            }
        });
        struct Join { Engine & e; std::thread & t; ~Join() { { std::lock_guard<std::mutex> lk(e.pf_m); e.pf_abort = true; } e.pf_cv.notify_all(); if (t.joinable()) t.join(); } } join_upl{ *this, upl };
        pf_staged += (long) jobs.size();
        // the host never waits for the GPU inside the window, except to stay at most two layers ahead (progress, queue depth)
        for (int il = 0; il < n_layer; il++) {
            const LW & l = L[il]; int * hid = h_idsw + (size_t) il * W * 8;
            if (l.recr) {
                k_gdn_core(q, 0, pPRM, l.S, l.QKV, l.G, l.Dr, cO); // apply the pending records: S = state before the window
                for (int t = 0; t < 3; t++) q.memcpy(cYR + (size_t) t * 12288, l.YR + (size_t) ((P0 - 3 + t) & RMASK) * 12288, 12288 * sizeof(float));
            }
            for (int ci = 0; ci < nchunk; ci++) {
                const int c0 = ci * PFC, n = std::min(PFC, W - c0); float * Xc = Xw + (size_t) c0 * 2048; const Act ax = act_at(pAX, c0, 2048);
                if (il == 0) k_norm_quant(q, n, Xc, h_embw + (size_t) c0 * 2048, nullptr, true, l.par + POff::attn_norm, eps, cXN, nullptr, ax);
                else         k_norm_quant(q, n, Xc, pY2 + (size_t) c0 * 2048, nullptr, false, l.par + POff::attn_norm, eps, cXN, nullptr, ax);
                p_act_tiles(q, n, 2048, ax, pT); PKW("norm + tiles");
                if (l.recr) {
                    float * YRz = cYR + (size_t) 3 * 12288;
                    p_mm_q8<0>(q, l.w_in, 12288, 2048, pT, n, YRz, 12288); PKW("mm qkvz");
                    p_gemv_f32(q, l.ab, 64, 2048, cXN, cAB, n / 8, 64); PKW("f32 ab/router");
                    p_gdn_gates(q, n, cAB, l.par + POff::dt_bias, l.par + POff::ssm_a, cG);
                    p_gdn_conv(q, n, cYR, l.par + POff::conv, eps, cQKV); PKW("gdn gates+conv");
                    p_gdn_core(q, n, l.S, cQKV, cG, cO); PKW("gdn core");
                    p_gdn_post(q, n, cO, YRz, l.par + POff::ssm_norm, eps, cAO); PKW("gdn post");
                    if (c0 + n < W) q.memcpy(cYR, cYR + (size_t) n * 12288, 3 * 12288 * sizeof(float)); // conv history for the next chunk
                    else for (int t = 0; t < 3; t++) q.memcpy(l.YR + (size_t) ((P0 + W - 3 + t) & RMASK) * 12288, cYR + (size_t) (n + t) * 12288, 12288 * sizeof(float));
                } else {
                    p_mm_q8<0>(q, l.w_in, 9216, 2048, pT, n, cY1, 9216); PKW("mm qkv");
                    k_att_prep(q, n, pPRM + chunk_prm[ci] * 16, cY1, l.par + POff::q_norm, l.par + POff::k_norm, eps, pROPE + (size_t) c0 * 64, cQQ, cQS, l.kv); PKW("att prep");
                    if (!pf_oldatt) {
                        if (ci == 0 && P0 > 0) p_kv_tiles(q, l.kv, 0, (P0 + 15) / 16, P0, pAT); // keys before the window
                        const int k0 = P0 + c0, k1 = P0 + c0 + n;
                        p_kv_tiles(q, l.kv, k0 / 16, (k1 + 15) / 16, k1, pAT); p_q_nat(q, n, cQQ, cQS, pQN, pQSN); PKW("att tiles");
                        p_flash(q, n, k0, pQN, pQSN, pAT, cY1, cAO); PKW("attention");
                    } else
                    for (int ai = chunk_att[ci]; ai < chunk_att[ci + 1]; ai++) {
                        const Att & a = att[ai]; const size_t off = (size_t) (a.c0 - c0);
                        k_att(q, a.nc, pPRM + a.prm * 16, a.nch, cQQ + off * 4096, cQS + off * 256, l.kv, PART);
                        k_att_fin(q, a.nc, pPRM + a.prm * 16, a.nch, PART, cY1 + off * 9216, act_at(cAO, off, 4096));
                    }
                    PKW("attention");
                }
                p_act_tiles(q, n, 4096, cAO, pT); p_mm_q8<0>(q, l.w_out, 2048, 4096, pT, n, cY, 2048); PKW("mm out");
                k_norm_quant(q, n, Xc, cY, nullptr, false, l.par + POff::post_norm, eps, cXN, nullptr, ax);
                p_gemv_f32(q, l.router, 257, 2048, cXN, cRL, n / 8, 272); PKW("f32 ab/router");
                k_topk(q, n, cRL, pID + (size_t) c0 * 8, pWT + (size_t) c0 * 8, pSHW + c0, hid + (size_t) c0 * 8, h_wtw + (size_t) c0 * 8); PKW("topk");
            }
            { std::unique_lock<std::mutex> lk(pf_m); pf_cv.wait(lk, [&] { return pf_up_ready > il; }); }
            if (!pf_up_ev[il].empty()) q.ext_oneapi_submit_barrier(pf_up_ev[il]);
            PKW("wait: uploads");
            const uint64_t * tab = pTAB + (size_t) il * 256; const bool q6 = cache->layer_cls[il] != 0;
            for (int p0 = 0; p0 < W; p0 += PFM) { // MoE in pieces of PFM tokens
                const int m = std::min(PFM, W - p0), gmax = m / 2 + 256; const Act axp = act_at(pAX, p0, 2048); float * y2 = pY2 + (size_t) p0 * 2048;
                p_group(q, m * 8, pID + (size_t) p0 * 8, pCNT, pGX); PKW("group");
                p_act_tiles(q, m, 2048, axp, pT); p_mm_q8<0>(q, l.sh_gu, 1024, 2048, pT, m, pGU, 1024);
                p_swiglu_quant(q, m, pGU, pAS);
                p_act_tiles(q, m, 512, pAS, pT); p_mm_q8<1>(q, l.sh_down, 2048, 512, pT, m, y2, 2048, pSHW + p0); PKW("shared expert");
                p_grp_tiles(q, gmax, pGX, 2048, axp, 3, pTG); PKW("grp tiles");
                p_exp_glu_x(q, gmax, pGX, tab, pTG, pACTE); PKW("exp gate/up");
                k_act_quant(q, m, 8, 8, pACTE, pAE);
                p_grp_tiles(q, gmax, pGX, 512, pAE, 0, pTG); PKW("grp tiles");
                if (q6) p_exp_down6_x(q, gmax, pGX, tab, pTG, pWT + (size_t) p0 * 8, pYP);
                else    p_exp_down_x(q, gmax, pGX, tab, pTG, pWT + (size_t) p0 * 8, pYP);
                PKW("exp down");
                p_moe_reduce(q, m, pYP, y2); PKW("moe reduce");
            }
            { sycl::event ev = q.ext_oneapi_submit_barrier(); std::lock_guard<std::mutex> lk(pf_m); pf_moe_ev[il] = ev; pf_moe_sub = il + 1; } pf_cv.notify_all();
            if (il >= 2) { pf_moe_ev[il - 2].wait(); progress(il - 1, total); }
        }
        q.wait(); upl.join(); progress(n_layer, total);
        for (int il = 0; il < n_layer; il++) { const int * hid = h_idsw + (size_t) il * W * 8;
            for (size_t i = 0; i < (size_t) W * 8; i++) { const size_t key = (size_t) il * 256 + (hid[i] & 255); cache->score[key] += 1.0f; cache->total[key]++; } }
        // final norm: HTw row i = hidden state before token i (row 0 = the one that predicted the first token of the window)
        q.memcpy(HTw, Hp, 2048 * sizeof(float));
        for (int ci = 0; ci < nchunk; ci++) { const int c0 = ci * PFC, n = std::min(PFC, W - c0);
            k_norm_quant(q, n, Xw + (size_t) c0 * 2048, pY2 + (size_t) c0 * 2048, nullptr, false, gpar, eps, HTw + (size_t) (c0 + 1) * 2048, nullptr, act_at(pAX, c0, 2048)); PKW("k_norm_quant"); }
        if (mtp && has_mtp) { // K/V of the MTP attention for the window: pairs (token, hidden state before it)
            const LW & l = L[n_layer];
            for (int ci = 0; ci < nchunk; ci++) { const int c0 = ci * PFC, n = std::min(PFC, W - c0), ng = n / 8; const Act ax = act_at(pAX, c0, 2048);
                k_mtp_in(q, n, h_embw + (size_t) c0 * 2048, HTw + (size_t) c0 * 2048, l.par + POff::enorm, l.par + POff::hnorm, eps, cAO); PKW("k_mtp_in");
                p_act_tiles(q, n, 4096, cAO, pT); p_mm_q8<0>(q, eh_proj, 2048, 4096, pT, n, cY, 2048); PKW("mtp mm");
                k_norm_quant(q, n, cY, nullptr, nullptr, false, l.par + POff::attn_norm, eps, cXN, nullptr, ax); PKW("k_norm_quant");
                p_act_tiles(q, n, 2048, ax, pT); p_mm_q8<0>(q, l.w_in, 9216, 2048, pT, n, cY1, 9216); PKW("mtp mm");
                k_att_prep(q, n, pPRM + chunk_prm[ci] * 16, cY1, l.par + POff::q_norm, l.par + POff::k_norm, eps, pROPE + (size_t) c0 * 64, cQQ, cQS, l.kv); PKW("k_att_prep"); }
        }
        q.memcpy(Hp, HTw + (size_t) W * 2048, 2048 * sizeof(float)); q.wait();
        if (total > n_layer) progress(total, total);
        s_pos = P0 + W; n_past = P0 + W; step_P = n_past; pf_tokens += W; pf_time += now_s() - t0;
        cache->give_back(); pf_dirty = true;
    }
    // after the last window of a prompt: bring the expert cache close to the experts the prompt used most
    bool pf_dirty = false;
    void prefill_end() {
        if (!pf_dirty) return; pf_dirty = false;
        const double tr = now_s(); pf_swaps += cache->refill(1500); pf_refill += now_s() - tr;
        { double in = 0, all = 0; for (size_t k = 0; k < cache->score.size(); k++) { all += cache->score[k]; if (cache->slot_of[k] >= 0) in += cache->score[k]; } pf_cover = all > 0 ? in / all : 0; }
    }
    long pf_tokens = 0, pf_staged = 0, pf_swaps = 0; double pf_time = 0, pf_refill = 0, pf_stage_s = 0, pf_mix_s = 0, pf_moe_s = 0, pf_cover = 0;

    // ---- checkpoints of the recurrent (delta-net) state: positions the context can be rolled back to.
    // The attention K/V caches need no copy: entries below the checkpoint position stay valid.
    struct Ckpt { int n_past = 0, s_pos = 0; std::vector<std::vector<float>> bufs; std::vector<float> hp; };
    void save_ckpt(Ckpt & c) {
        c.n_past = n_past; c.s_pos = s_pos; c.bufs.clear(); c.hp.resize(2048);
        for (auto & l : L) if (l.recr) {
            const size_t sz[5] = { (size_t) 32 * 128 * 128, (size_t) RING * 12288, (size_t) RING * 64, (size_t) RING * 8192, (size_t) RING * 4096 }; float * src[5] = { l.S, l.YR, l.G, l.QKV, l.Dr };
            for (int i = 0; i < 5; i++) { c.bufs.emplace_back(sz[i]); q.memcpy(c.bufs.back().data(), src[i], sz[i] * 4); }
        }
        q.memcpy(c.hp.data(), Hp, 2048 * 4); q.wait();
    }
    void restore_ckpt(const Ckpt & c) {
        size_t k = 0;
        for (auto & l : L) if (l.recr) { float * dst[5] = { l.S, l.YR, l.G, l.QKV, l.Dr }; for (int i = 0; i < 5; i++, k++) q.memcpy(dst[i], c.bufs[k].data(), c.bufs[k].size() * 4); }
        q.memcpy(Hp, c.hp.data(), 2048 * 4); q.wait(); n_past = c.n_past; s_pos = c.s_pos;
    }

    // record the kernel graphs for the step sizes that will be used (otherwise the first step of each size pays for it)
    void warm_up(const std::vector<int> & ncs, bool mtp) {
        std::vector<float> lg((size_t) n_vocab * NCMAX); memset(h_emb, 0, sizeof(float) * NCMAX * 2048); memset(h_embm, 0, sizeof(float) * NCMAX * 2048);
        for (int nc : ncs) {
            if (nc < 1 || nc > NCMAX) continue;
            step(nc, true, lg.data());
            if (mtp && has_mtp) { q.fill(HIN, 0.0f, (size_t) NCMAX * 2048); mtp_step(nc, 0, true, nullptr); if (nc == 1) mtp_step(1, 0, false, lg.data()); }
        }
        q.wait(); reset(); cache->hits = cache->misses = 0; missed.clear();
        std::fill(cache->score.begin(), cache->score.end(), 0.0f); // warm-up routing must not count as usage
    }

    void dbg_x(const char * tag) { // residual checksum for comparisons with the reference
        std::vector<float> x(2048); q.memcpy(x.data(), X, 2048 * 4).wait(); double s = 0; for (float v : x) s += (double) v * v;
        printf("  dbg %s |x|=%.5f x0=%.5f x1=%.5f x2047=%.5f\n", tag, sqrt(s), x[0], x[1], x[2047]);
    }
};

} // namespace q36
