// q36 server: OpenAI-compatible HTTP API (chat completions with streaming, reasoning and tool calls) on top of the engine.
//   q36-server <model.q36> [--tok file.tok] [--port 9936] [--ctx 32768] [--vram-gb 10.5] [--profile file] [--mtp 2] [--threads 6]
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include "../src/engine.hpp"
#include "../src/tokenizer.hpp"
#include "../src/chat.hpp"
#include <random>
#include <ctime>
using namespace q36;

// ---------------------------------------------------------------- sampling
struct SParams { float temp = 1.0f, top_p = 0.95f, min_p = 0.0f; int top_k = 20; };
struct Sampler {
    std::mt19937_64 rng{ 1234 }; std::vector<std::pair<float, int>> cand;
    // returns the sampled token; candidates are the tokens within 30 nats of the best one (AVX scan), then top-k / top-p / min-p / temperature
    int sample(const float * l, int n, const SParams & sp) {
        __m256 vm = _mm256_set1_ps(-1e30f); int i = 0; for (; i + 8 <= n; i += 8) vm = _mm256_max_ps(vm, _mm256_loadu_ps(l + i));
        float mx[8]; _mm256_storeu_ps(mx, vm); float m = mx[0]; for (int k = 1; k < 8; k++) m = std::max(m, mx[k]); for (; i < n; i++) m = std::max(m, l[i]);
        if (sp.temp <= 0.0f) { // greedy: first index of the maximum
            const __m256 vq = _mm256_set1_ps(m);
            for (i = 0; i + 8 <= n; i += 8) { const int mask = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(l + i), vq, _CMP_EQ_OQ)); if (mask) { unsigned long b; _BitScanForward(&b, (unsigned long) mask); return i + (int) b; } }
            for (; i < n; i++) if (l[i] == m) return i;
            return 0;
        }
        // candidates within `width` nats of the best token (a token 16 nats down has 1e-7 of its probability); widened if top-k needs more
        float width = 16.0f;
        for (;;) {
            const __m256 vt = _mm256_set1_ps(m - width); cand.clear();
            for (i = 0; i + 8 <= n; i += 8) { int mask = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(l + i), vt, _CMP_GE_OQ)); while (mask) { unsigned long b; _BitScanForward(&b, (unsigned long) mask); mask &= mask - 1; cand.push_back({ l[i + b], i + (int) b }); } }
            for (; i < n; i++) if (l[i] >= m - width) cand.push_back({ l[i], i });
            if (sp.top_k <= 0 || (int) cand.size() >= sp.top_k || width > 60.0f) break;
            width *= 2.0f;
        }
        auto gt = [](const std::pair<float, int> & a, const std::pair<float, int> & b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };
        if (sp.top_k > 0 && (int) cand.size() > sp.top_k) { std::partial_sort(cand.begin(), cand.begin() + sp.top_k, cand.end(), gt); cand.resize(sp.top_k); }
        else std::sort(cand.begin(), cand.end(), gt);
        if (cand.size() == 1) return cand[0].second;
        std::vector<double> p(cand.size()); double sum = 0; for (size_t k = 0; k < cand.size(); k++) { p[k] = exp((double) cand[k].first - m); sum += p[k]; }
        size_t keep = cand.size();
        if (sp.top_p < 1.0f) { double c = 0; for (size_t k = 0; k < cand.size(); k++) { c += p[k] / sum; if (c >= sp.top_p) { keep = k + 1; break; } } }
        if (sp.min_p > 0.0f) { size_t k = 1; while (k < keep && p[k] >= sp.min_p * p[0]) k++; keep = k; }
        double s2 = 0; for (size_t k = 0; k < keep; k++) { p[k] = exp(((double) cand[k].first - m) / sp.temp); s2 += p[k]; }
        double r = std::uniform_real_distribution<double>(0.0, s2)(rng);
        for (size_t k = 0; k < keep; k++) { r -= p[k]; if (r <= 0) return cand[k].second; }
        return cand[keep - 1].second;
    }
};

// ---------------------------------------------------------------- model session: context tokens, prefix reuse, generation
struct Session {
    Engine & e; Tokenizer & tk; std::vector<int> ctx; std::vector<Engine::Ckpt> ckpts; std::vector<float> logits, dl; Sampler smp;
    int mtp_max = 0, pf_min = 512; double p_acc[q36g::NCMAX] = { 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7 };
    int tok_think_end, tok_tool, tok_tool_end, tok_im_end, tok_eos;
    long st_prompt_n = 0, st_gen_n = 0, st_drafted = 0, st_accepted = 0, st_lk_drafted = 0, st_lk_accepted = 0; int lookup = 0; bool lk_streak = false; double lk_S = 2, lk_F = 1, cost_F = 6e-3, cost_C = 2.5e-3, cost_D = 1.6e-3; double st_prompt_s = 0, st_gen_s = 0;
    // progress log: every log_interval seconds one line with the current speed (0 = off); log_level 2 adds the time split per token
    double log_interval = 1.0; int log_level = 1;
    struct Prog { double t_last = 0; long n_last = 0, hits = 0, misses = 0, drafted = 0, accepted = 0, steps = 0; double sub = 0, gpu = 0, cpu = 0, head = 0; } pg;
    void prog_start() { pg.t_last = now_s(); pg.n_last = 0; pg.hits = e.cache->hits; pg.misses = e.cache->misses; pg.drafted = st_drafted; pg.accepted = st_accepted; pg.steps = e.n_steps;
                        pg.sub = e.t_submit; pg.gpu = e.t_gpu_wait; pg.cpu = e.t_cpu; pg.head = e.t_head; }
    // phase: "prefill" (done of total) or "gen" (done tokens); t0 = start of the phase
    void prog(const char * phase, long done, long total, double t0, int layer = 0, int layers = 0) {
        if (log_interval <= 0) return;
        const double t = now_s(); if (t - pg.t_last < log_interval) return;
        const long dn = done - pg.n_last; const double dt = t - pg.t_last; const long dh = e.cache->hits - pg.hits, dm = e.cache->misses - pg.misses;
        char buf[512]; int k = 0;
        if (total > 0) k += snprintf(buf + k, sizeof buf - k, "  %s %ld/%ld tok (%.0f%%) | now %.1f t/s, avg %.1f t/s", phase, done, total, 100.0 * done / total, dn / dt, done / std::max(1e-9, t - t0));
        else           k += snprintf(buf + k, sizeof buf - k, "  %s %ld tok | now %.1f t/s, avg %.1f t/s", phase, done, dn / dt, done / std::max(1e-9, t - t0));
        if (layer > 0) k += snprintf(buf + k, sizeof buf - k, " | GPU window, layer %d/%d | ctx %d", layer, layers, e.n_past);
        else if (dh + dm) k += snprintf(buf + k, sizeof buf - k, " | cache hit %.3f | ctx %d", (double) dh / (dh + dm), e.n_past);
        else k += snprintf(buf + k, sizeof buf - k, " | ctx %d", e.n_past);
        if (total <= 0 && mtp_max > 0) k += snprintf(buf + k, sizeof buf - k, " | drafts %ld/%ld, depth %d", st_accepted - pg.accepted, st_drafted - pg.drafted, draft_depth());
        if (log_level >= 2 && dn > 0) k += snprintf(buf + k, sizeof buf - k, " | per tok ms: gpu %.2f cpu %.2f head %.2f submit %.2f | %.2f tok/step | uploads %ld",
            (e.t_gpu_wait - pg.gpu) / dn * 1e3, (e.t_cpu - pg.cpu) / dn * 1e3, (e.t_head - pg.head) / dn * 1e3, (e.t_submit - pg.sub) / dn * 1e3, (double) dn / std::max(1L, e.n_steps - pg.steps), e.cache->uploads);
        printf("%s\n", buf);
        pg.t_last = t; pg.n_last = done; pg.hits = e.cache->hits; pg.misses = e.cache->misses; pg.drafted = st_drafted; pg.accepted = st_accepted; pg.steps = e.n_steps;
        pg.sub = e.t_submit; pg.gpu = e.t_gpu_wait; pg.cpu = e.t_cpu; pg.head = e.t_head;
    }
    long feed_base = 0, feed_total = 0; double feed_t0 = 0;
    Session(Engine & en, Tokenizer & t, int mtp) : e(en), tk(t), mtp_max(std::min(mtp, q36g::NCMAX - 1)) {
        logits.resize((size_t) q36g::NCMAX * e.n_vocab); dl.resize(e.n_vocab);
        auto id = [&](const char * s) { auto it = tk.tok2id.find(s); return it == tk.tok2id.end() ? -1 : it->second; };
        tok_think_end = id("</think>"); tok_tool = id("<tool_call>"); tok_tool_end = id("</tool_call>"); tok_im_end = id("<|im_end|>"); tok_eos = tk.eos;
        if (!e.has_mtp) mtp_max = 0;
    }
    // decode tokens[from, to) into the context (no sampling); logits of the last one if asked
    void feed(const std::vector<int> & t, size_t from, size_t to, bool want_last_logits) {
        size_t i = from;
        while (i < to) {
            // long stretches go through the layer-major path (whole windows on the GPU); the tail and the last token step by step
            const size_t rest = to - i - (want_last_logits ? 1 : 0);
            if (e.pf_window > 0 && rest >= (size_t) pf_min) {
                const int W = (int) std::min<size_t>((size_t) e.pf_window, rest / 8 * 8); const long base = (long) ctx.size() - feed_base;
                e.prefill_window(t.data() + i, W, mtp_max > 0, [&](int done, int total) { if (feed_total > 0) prog("prefill", base + (long) W * done / total, feed_total, feed_t0, done < total ? done : 0, total); });
                for (int c = 0; c < W; c++) ctx.push_back(t[i + c]);
                i += W; continue;
            }
            e.prefill_end();
            size_t nc = std::min<size_t>(q36g::NCMAX, to - i);
            if (want_last_logits && i + nc == to && nc > 1) nc--; // the last token goes alone, with the LM head
            const bool lg = want_last_logits && i + nc == to;
            for (size_t c = 0; c < nc; c++) e.embed(t[i + c], e.h_emb + c * 2048);
            const int P = e.n_past;
            e.step((int) nc, lg, logits.data()); e.commit((int) nc);
            if (mtp_max > 0) { // MTP attention K/V for these positions: pairs (token, hidden state before it)
                memcpy(e.h_embm, e.h_emb, nc * 2048 * 4); e.q.memcpy(e.HIN, e.Hp, 2048 * 4); if (nc > 1) e.q.memcpy(e.HIN + 2048, e.HT, (nc - 1) * 2048 * 4);
                e.mtp_step((int) nc, P, true, nullptr); e.q.memcpy(e.Hp, e.HT + (nc - 1) * 2048, 2048 * 4);
            }
            for (size_t c = 0; c < nc; c++) ctx.push_back(t[i + c]);
            i += nc;
            if (feed_total > 0) prog("prefill", (long) ctx.size() - feed_base, feed_total, feed_t0);
        }
        e.prefill_end();
        if (want_last_logits) if (const char * dp = getenv("Q36_DUMP_LOGITS")) { static int nd = 0; FILE * f = fopen((std::string(dp) + "." + std::to_string(++nd)).c_str(), "wb"); if (f) { fwrite(logits.data(), 4, logits.size(), f); fclose(f); } } // debugging aid
    }
    // bring the context to `t` (the prompt), reusing the longest valid prefix; ckpt_at = position worth a checkpoint (0 = none)
    void set_prompt(const std::vector<int> & t, size_t ckpt_at) {
        if ((int) t.size() + 8 >= e.n_ctx) throw std::runtime_error("the prompt does not fit into the context (" + std::to_string(t.size()) + " tokens, context " + std::to_string(e.n_ctx) + ")");
        size_t lcp = 0; while (lcp < t.size() && lcp < ctx.size() && t[lcp] == ctx[lcp]) lcp++;
        while (!ckpts.empty() && (size_t) ckpts.back().n_past > lcp) ckpts.pop_back(); // checkpoints after the divergence are stale
        size_t start;
        if (lcp == ctx.size() && lcp < t.size() && (int) ctx.size() == e.n_past) start = lcp; // pure continuation
        else if (!ckpts.empty() && (size_t) ckpts.back().n_past < t.size()) { e.restore_ckpt(ckpts.back()); start = (size_t) ckpts.back().n_past; ctx.resize(start); }
        else { ckpts.clear(); e.reset(); ctx.clear(); start = 0; }
        const double t0 = now_s(); st_prompt_n = (long) (t.size() - start);
        feed_base = (long) start; feed_total = st_prompt_n; feed_t0 = t0; st_drafted = st_accepted = 0; prog_start();
        if (ckpt_at > start && ckpt_at < t.size()) {
            feed(t, start, ckpt_at, false);
            if (ckpts.size() >= 4) ckpts.erase(ckpts.begin());
            ckpts.emplace_back(); e.save_ckpt(ckpts.back()); start = ckpt_at;
        }
        feed(t, start, t.size(), true);
        st_prompt_s = now_s() - t0; feed_total = 0;
    }
    // prompt lookup: the longest earlier occurrence (>= 3 tokens) of the end of the context; drafts = the tokens that followed it
    int lookup_drafts(int * out, int maxn, int & mlen) const {
        const int n = (int) ctx.size(); mlen = 0; if (n < 4 || maxn <= 0) return 0;
        int best = -1, bl = 0; const int a = ctx[n - 1], b = ctx[n - 2], c = ctx[n - 3];
        for (int i = n - 2; i >= 2; i--) { // ctx[i] matches the last token, the continuation starts at i + 1
            if (ctx[i] != a || ctx[i - 1] != b || ctx[i - 2] != c) continue;
            int l = 3; while (l < 64 && i - l >= 0 && ctx[i - l] == ctx[n - 1 - l]) l++;
            if (l > bl) { bl = l; best = i; if (l >= 64) break; }
        }
        if (best < 0) return 0;
        int k = 0; while (k < maxn && best + 1 + k < n) { const int t = ctx[best + 1 + k]; if (t == tok_im_end || t == tok_eos) break; out[k] = t; k++; }
        mlen = bl; return k;
    }
    int draft_depth() const { int d = 0; double p = 1.0; while (d < mtp_max) { p *= p_acc[d]; if (d > 0 && p < 0.5) break; d++; } return d; }
    // generate up to max_new tokens; on_token(id) returns false to stop. Returns the finish reason.
    template <class F> const char * generate(int max_new, const SParams & sp, F && on_token) {
        const double t0 = now_s(); st_gen_n = 0; st_drafted = st_accepted = st_lk_drafted = st_lk_accepted = 0; prog_start();
        int next = smp.sample(logits.data(), e.n_vocab, sp); const char * reason = "length";
        while (st_gen_n < max_new) {
            if (next == tok_im_end || next == tok_eos) { reason = "stop"; break; }
            if (e.n_past + q36g::NCMAX + 1 >= e.n_ctx) { reason = "length"; break; }
            // choose the drafts by expected tokens per second: a step costs F + C * columns (+ D per MTP draft); measured below
            const int n = e.n_past; int drafts[q36g::NCMAX]; int nd = 0; int depth = 0; bool lk = false;
            const int room = (int) std::min<long>(q36g::NCMAX - 1, (long) max_new - st_gen_n - 1);
            depth = std::min(draft_depth(), room); // MTP depth by its acceptance (as before); prompt lookup only when it beats that by cost
            double best; { double tok = 1, p = 1; for (int d = 1; d <= depth; d++) { p *= p_acc[d - 1]; tok += p; } best = tok / (cost_F + cost_C * (1 + depth) + cost_D * depth); }
            if (lookup && room > 0) { // the context continues an earlier passage: its continuation as drafts
                int ld[q36g::NCMAX]; ctx.push_back(next); int mlen = 0; int nl = lookup_drafts(ld, room, mlen); ctx.pop_back();
                if (mlen < (lk_streak ? 3 : 8)) nl = 0; // short matches are noise unless we are copying a passage right now
                const double r = std::min(0.98, (lk_S + 1.0) / (lk_S + lk_F + 2.0) + 0.005 * mlen); double tok = 1, p = 1; int bn = 0;
                for (int k = 1; k <= nl; k++) { p *= r; tok += p; const double v = tok / (cost_F + cost_C * (1 + k)); if (v > best) { best = v; bn = k; } }
                if (bn > 0) { nd = bn; for (int k = 0; k < bn; k++) drafts[k] = ld[k]; lk = true; depth = 0; }
            }
            const double td0 = now_s();
            if (depth > 0) {
                int tok = next;
                for (int k = 0; k < depth; k++) {
                    e.embed(tok, e.h_embm); e.q.memcpy(e.HIN, k == 0 ? e.Hp : e.HM, 2048 * 4);
                    e.mtp_step(1, n + k, false, dl.data());
                    const int nr = e.draft_rows_d ? e.n_draft_rows : e.n_vocab; int best = (int) (std::max_element(dl.begin(), dl.begin() + nr) - dl.begin()); if (e.draft_rows_d) best = e.draft_rows_h[best];
                    if (best == tok_im_end || best == tok_eos) break; // do not draft past the end of the turn
                    drafts[nd++] = best; tok = best;
                }
            }
            const int nc = 1 + nd;
            e.embed(next, e.h_emb); for (int k = 0; k < nd; k++) e.embed(drafts[k], e.h_emb + (size_t) (k + 1) * 2048);
            const double ts0 = now_s(); if (nd > 0 && !lk) cost_D = 0.9 * cost_D + 0.1 * (ts0 - td0) / nd;
            const double cpu0 = e.t_cpu; e.step(nc, true, logits.data());
            { const double dt = now_s() - ts0, dc = e.t_cpu - cpu0; cost_C = 0.9 * cost_C + 0.1 * std::max(1e-4, dc / nc); cost_F = 0.9 * cost_F + 0.1 * std::max(1e-3, dt - dc); }
            ctx.push_back(next); st_gen_n++; bool go = on_token(next);
            int acc = 0; int samp = smp.sample(logits.data(), e.n_vocab, sp);
            while (go && acc < nd && samp == drafts[acc] && st_gen_n < max_new) {
                ctx.push_back(samp); st_gen_n++; go = on_token(samp); acc++;
                samp = smp.sample(logits.data() + (size_t) acc * e.n_vocab, e.n_vocab, sp);
            }
            if (lk) { lk_streak = acc == nd; lk_S = 0.9 * lk_S + acc; lk_F = 0.9 * lk_F + (acc < nd ? 1 : 0); st_lk_drafted += nd; st_lk_accepted += acc; }
            else { for (int k = 0; k < nd; k++) { if (k < acc) p_acc[k] = 0.9 * p_acc[k] + 0.1; else if (k == acc) p_acc[k] = 0.9 * p_acc[k]; }
                   // every draft accepted: evidence for the next depth too (otherwise a depth that once fell below the threshold is never tried again)
                   static const bool probe = getenv("Q36_NO_DEPTH_PROBE") == nullptr; if (probe && nd > 0 && acc == nd && nd < mtp_max) p_acc[nd] = 0.99 * p_acc[nd] + 0.01; }
            e.commit(1 + acc); st_drafted += nd; st_accepted += acc;
            if (mtp_max > 0) {
                if (depth == 0) { e.embed(next, e.h_embm); e.q.memcpy(e.HIN, e.Hp, 2048 * 4); e.mtp_step(1, n, true, nullptr); }
                if (acc > 0) { for (int k = 0; k < acc; k++) e.embed(drafts[k], e.h_embm + (size_t) k * 2048); e.q.memcpy(e.HIN, e.HT, (size_t) acc * 2048 * 4); e.mtp_step(acc, n + 1, true, nullptr); }
                e.q.memcpy(e.Hp, e.HT + (size_t) acc * 2048, 2048 * 4);
            }
            prog("gen", st_gen_n, 0, t0);
            if (!go) { reason = "abort"; break; }
            next = samp;
        }
        e.q.wait(); st_gen_s = now_s() - t0;
        return reason;
    }
};

// ---------------------------------------------------------------- HTTP
struct Req { std::string method, path, body; };
static bool send_all(SOCKET s, const std::string & d) { size_t o = 0; while (o < d.size()) { int n = send(s, d.data() + o, (int) std::min<size_t>(d.size() - o, 1 << 20), 0); if (n <= 0) return false; o += n; } return true; }
static bool read_req(SOCKET s, Req & r) {
    std::string buf; char tmp[65536]; size_t hdr_end = std::string::npos;
    while (hdr_end == std::string::npos) { int n = recv(s, tmp, sizeof tmp, 0); if (n <= 0) return false; buf.append(tmp, n); hdr_end = buf.find("\r\n\r\n"); if (buf.size() > (1 << 20) && hdr_end == std::string::npos) return false; }
    const std::string head = buf.substr(0, hdr_end); const size_t sp1 = head.find(' '), sp2 = head.find(' ', sp1 + 1); if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    r.method = head.substr(0, sp1); r.path = head.substr(sp1 + 1, sp2 - sp1 - 1); const size_t qm = r.path.find('?'); if (qm != std::string::npos) r.path.resize(qm);
    size_t clen = 0; { std::string low = head; for (auto & c : low) c = (char) tolower((unsigned char) c); const size_t p = low.find("content-length:"); if (p != std::string::npos) clen = (size_t) strtoull(low.c_str() + p + 15, nullptr, 10); }
    if (clen > (256u << 20)) return false;
    r.body = buf.substr(hdr_end + 4);
    while (r.body.size() < clen) { int n = recv(s, tmp, sizeof tmp, 0); if (n <= 0) return false; r.body.append(tmp, n); }
    return true;
}
static const char * CORS = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
static void respond(SOCKET s, int code, const std::string & body, const char * ctype = "application/json; charset=utf-8") {
    const char * msg = code == 200 ? "OK" : code == 204 ? "No Content" : code == 400 ? "Bad Request" : code == 404 ? "Not Found" : "Internal Server Error";
    send_all(s, "HTTP/1.1 " + std::to_string(code) + " " + msg + "\r\nContent-Type: " + ctype + "\r\nContent-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n" + CORS + "\r\n" + body);
}
static std::string err_json(const std::string & m, const char * type = "invalid_request_error") { J e = J::obj(); J & o = e.set("error", J::obj()); o.set("message", m); o.set("type", type); return e.dump(); }

// longest suffix of s that is an incomplete UTF-8 sequence
static size_t utf8_tail(const std::string & s) {
    size_t n = s.size(), k = 0;
    while (k < 4 && k < n && ((unsigned char) s[n - 1 - k] & 0xC0) == 0x80) k++;
    if (k >= n) return 0;
    const unsigned char c = (unsigned char) s[n - 1 - k]; const size_t need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    return need > k + 1 ? k + 1 : 0;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) { fprintf(stderr, "usage: q36-server model.q36 [--tok file.tok] [--port 9936] [--host 127.0.0.1] [--ctx 32768] [--vram-gb 10.5 | --cache-gb G] [--profile file] [--mtp 2] [--threads 6] [--draft-vocab 65536 --draft-vocab-extra file] [--alias name] [--log-interval 1] [--log-level 0|1|2] [--prefill-window 2048]\n"); return 1; }
    std::string pack_path = argv[1], tok_path, prof, dv_extra, host = "127.0.0.1", alias = "Qwen3.6-35B-A3B-q36"; int port = 9936, ctx = 32768, n_threads = 6, mtp = 2, dv_keep = 65536, log_level = 1, pf_window = 4096, pf_min = 512, lookup = 0; double cache_gb = 0, vram_gb = 10.5, log_interval = 1.0;
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string a = argv[i]; const char * v = argv[i + 1];
        if (a == "--tok") tok_path = v; else if (a == "--port") port = atoi(v); else if (a == "--ctx") ctx = atoi(v); else if (a == "--cache-gb") cache_gb = atof(v); else if (a == "--vram-gb") vram_gb = atof(v); else if (a == "--profile") prof = v;
        else if (a == "--mtp") mtp = atoi(v); else if (a == "--threads") n_threads = atoi(v); else if (a == "--draft-vocab") dv_keep = atoi(v); else if (a == "--draft-vocab-extra") dv_extra = v;
        else if (a == "--host") host = v; else if (a == "--log-interval") log_interval = atof(v); else if (a == "--log-level") log_level = atoi(v); else if (a == "--prefill-window") pf_window = atoi(v); else if (a == "--prefill-min") pf_min = atoi(v); else if (a == "--lookup") lookup = atoi(v); else if (a == "--alias") alias = v; else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }
    try {
        double t0 = now_s(); if (tok_path.empty()) { tok_path = pack_path; const size_t d = tok_path.rfind('.'); tok_path = tok_path.substr(0, d) + ".tok"; }
        Tokenizer tk; tk.load(tok_path); Pack pk; pk.open(pack_path); Pool pool(n_threads);
        { std::atomic<uint64_t> sum{0}; const uint64_t n = pk.mm.n; const uint8_t * p = pk.mm.p; // bring the pack into RAM
          pool.pfor((int64_t) (n / 4096), [&](int64_t a, int64_t b) { uint64_t s = 0; for (int64_t i = a; i < b; i++) s += p[(uint64_t) i * 4096]; sum += s; }); }
        printf("model in RAM: %.1f GB, %.1fs\n", pk.mm.n / 1e9, now_s() - t0);
        if (prof.empty()) prof = pack_path + ".prof";
        if (tok_path.empty()) { tok_path = pack_path; const size_t d = tok_path.rfind('.'); tok_path = tok_path.substr(0, d) + ".tok"; }
        Engine e(pk, pool, ctx, cache_gb > 0 ? cache_gb : -vram_gb, pf_window);
        const bool have_prof = e.cache->load_profile(prof);
        { std::vector<int> ncs = { 1, 8 }; for (int k = 2; k <= mtp + 1; k++) ncs.push_back(k); const std::vector<float> keep = e.cache->score; e.cache->rate = 0; e.warm_up(ncs, mtp > 0); e.cache->score = keep; e.cache->rate = 4; }
        e.cache->prefill();
        if (dv_keep > 0 && mtp > 0) {
            std::vector<int> cyr; // without a list file: every token past n_keep that contains Cyrillic letters (drafts in Ukrainian etc.)
            if (dv_extra.empty()) for (int id = dv_keep; id < (int) tk.tokens.size(); id++) {
                const std::string pc = tk.piece(id); for (size_t i = 0; i + 1 < pc.size(); i++) { const unsigned char a = (unsigned char) pc[i], b = (unsigned char) pc[i + 1];
                    if (a >= 0xD0 && a <= 0xD4 && (b & 0xC0) == 0x80) { const uint32_t cp = ((a & 0x1Fu) << 6) | (b & 0x3Fu); if (cp >= 0x400 && cp <= 0x52F) { cyr.push_back(id); break; } } } }
            e.set_draft_vocab(dv_keep, dv_extra, cyr);
            printf("draft vocabulary: %d rows (%d Cyrillic tokens added)\n", e.n_draft_rows, (int) cyr.size());
        }
        printf("device: %s | VRAM: %.2f GB model + state, %.2f GB expert cache (%d experts, %.1f%%) | context %d | profile %s | MTP depth up to %d | prefill window %d\n", e.q.get_device().get_info<sycl::info::device::name>().c_str(),
               e.vram / 1e9, e.cache->bytes / 1e9, e.cache->cls[0].cap + e.cache->cls[1].cap, 100.0 * (e.cache->cls[0].cap + e.cache->cls[1].cap) / (e.cache->nl * 256.0), ctx, have_prof ? "loaded" : "none", mtp, e.pf_window);
        Session ss(e, tk, mtp); ss.log_interval = log_level >= 1 ? log_interval : 0; ss.log_level = log_level; ss.pf_min = std::max(64, pf_min); ss.lookup = lookup;

        WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
        SOCKET ls = socket(AF_INET, SOCK_STREAM, 0); int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof one);
        sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons((u_short) port); inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
        if (bind(ls, (sockaddr *) &addr, sizeof addr) != 0 || listen(ls, 16) != 0) throw std::runtime_error("cannot listen on " + host + ":" + std::to_string(port));
        printf("listening on http://%s:%d  (POST /v1/chat/completions)\n", host.c_str(), port);
        long req_id = 0; time_t last_save = time(nullptr);
        for (;;) {
            SOCKET cs = accept(ls, nullptr, nullptr); if (cs == INVALID_SOCKET) continue;
            setsockopt(cs, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof one);
            Req r;
            if (read_req(cs, r)) {
                try {
                    if (r.method == "OPTIONS") respond(cs, 204, "");
                    else if (r.path == "/health" || r.path == "/v1/health") respond(cs, 200, "{\"status\":\"ok\"}");
                    else if (r.path == "/v1/models" || r.path == "/models") {
                        J o = J::obj(); o.set("object", "list"); J & d = o.set("data", J::arr()); J & m = d.push(J::obj()); m.set("id", alias); m.set("object", "model"); m.set("owned_by", "q36"); respond(cs, 200, o.dump());
                    }
                    else if (r.path == "/tokenize" && r.method == "POST") {
                        J in = J::parse(r.body); auto ids = tk.encode(in.at("content").str()); J o = J::obj(); J & a = o.set("tokens", J::arr()); for (int id : ids) a.push(J(id)); respond(cs, 200, o.dump());
                    }
                    else if ((r.path == "/v1/chat/completions" || r.path == "/chat/completions" || r.path == "/completion") && r.method == "POST") {
                        const J in = J::parse(r.body); const bool raw = r.path == "/completion"; const bool stream = in.at("stream").boolean(false);
                        SParams sp; sp.temp = (float) in.at("temperature").num(1.0); sp.top_p = (float) in.at("top_p").num(0.95); sp.top_k = (int) in.at("top_k").num(20); sp.min_p = (float) in.at("min_p").num(0.0);
                        if (in.find("seed") && in.at("seed").num(-1) >= 0) ss.smp.rng.seed((uint64_t) in.at("seed").num()); 
                        int max_new = (int) in.at("max_completion_tokens").num(in.at("max_tokens").num(in.at("n_predict").num(-1))); if (max_new < 0) max_new = 1 << 30;
                        const J & tools = in.at("tools"); bool thinking = true, preserve = false;
                        { const J & kw = in.at("chat_template_kwargs"); if (kw.find("enable_thinking")) thinking = kw.at("enable_thinking").boolean(true); if (kw.find("preserve_thinking")) preserve = kw.at("preserve_thinking").boolean(false);
                          if (in.find("enable_thinking")) thinking = in.at("enable_thinking").boolean(true);
                          const std::string eff = in.at("reasoning_effort").str(); if (eff == "none" || eff == "minimal") thinking = false; }
                        std::vector<std::string> stops; { const J & st = in.at("stop"); if (st.is_str()) stops.push_back(st.s); else if (st.is_arr()) for (auto & x : st.a) if (x.is_str() && !x.s.empty()) stops.push_back(x.s); }
                        std::vector<int> prompt; size_t ckpt_at = 0;
                        if (raw) { const J & p = in.at("prompt"); if (p.is_arr()) for (auto & x : p.a) prompt.push_back((int) x.num()); else prompt = tk.encode(p.str()); thinking = false; }
                        else {
                            prompt = tk.encode(render_chat(in.at("messages"), tools, thinking, preserve, true));
                            ckpt_at = tk.encode(render_chat(in.at("messages"), tools, thinking, preserve, false)).size();
                        }
                        if (prompt.empty()) throw std::runtime_error("empty prompt");
                        const std::string id = "chatcmpl-q36-" + std::to_string(++req_id); const long long created = (long long) time(nullptr);
                        if (log_level >= 1) printf("req %ld: %s, prompt %zu tok, max_tokens %s, temp %.2f, top_k %d, top_p %.2f%s%s\n", req_id, r.path.c_str(), prompt.size(), max_new == (1 << 30) ? "-" : std::to_string(max_new).c_str(), sp.temp, sp.top_k, sp.top_p, thinking ? ", thinking" : "", tools.is_arr() && !tools.a.empty() ? ", tools" : "");
                        ss.set_prompt(prompt, ckpt_at);
                        if (log_level >= 2 && e.pf_tokens > 0) printf("  GPU windows: %ld tok, %.2fs, %ld experts streamed, cache refill %ld swaps in %.2fs, cached set covers %.3f of the scored uses\n", e.pf_tokens, e.pf_time, e.pf_staged, e.pf_swaps, e.pf_refill, e.pf_cover);
                        if (e.pf_kprof && e.pf_tokens > 0) { e.print_prof(e.pf_tokens); e.kprof.clear(); } e.pf_tokens = e.pf_staged = e.pf_swaps = 0; e.pf_time = e.pf_refill = e.pf_stage_s = e.pf_mix_s = e.pf_moe_s = 0;
                        if (log_level >= 1) printf("  prefill done: %ld new tok in %.2fs (%.0f t/s), %zu reused\n", ss.st_prompt_n, ss.st_prompt_s, ss.st_prompt_n / std::max(1e-9, ss.st_prompt_s), prompt.size() - (size_t) ss.st_prompt_n);
                        // output state machine: reasoning until </think>, then content; <tool_call> blocks are parsed into tool calls
                        enum { THINK, CONTENT, TOOL } mode = (thinking && !raw) ? THINK : CONTENT;
                        std::string pend, reasoning, content, tool_buf, full; J tool_calls = J::arr(); bool client_ok = true, lead = true, stop_hit = false;
                        auto chunk = [&](const J & delta, const char * finish) {
                            J c = J::obj(); c.set("id", id); c.set("object", "chat.completion.chunk"); c.set("created", J(created)); c.set("model", alias);
                            J & ch = c.set("choices", J::arr()).push(J::obj()); ch.set("index", 0); ch.set("delta", delta); ch.set("finish_reason", finish ? J(finish) : J());
                            if (!send_all(cs, "data: " + c.dump() + "\n\n")) client_ok = false;
                        };
                        if (stream) { send_all(cs, std::string("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n") + CORS + "\r\n"); J d = J::obj(); d.set("role", "assistant"); d.set("content", ""); if (!raw) chunk(d, nullptr); }
                        auto emit = [&](const std::string & text) { // complete UTF-8 text for the current mode
                            if (text.empty()) return;
                            if (mode == THINK) { reasoning += text; if (stream) { J d = J::obj(); d.set("reasoning_content", text); chunk(d, nullptr); } }
                            else if (mode == CONTENT) { content += text; if (stream) { J d = J::obj(); d.set("content", text); chunk(d, nullptr); } }
                            else tool_buf += text;
                        };
                        auto flush = [&](bool all) { const size_t keep = all ? 0 : utf8_tail(pend); std::string t = pend.substr(0, pend.size() - keep); pend.erase(0, pend.size() - keep);
                            if (lead && mode != TOOL) { size_t k = 0; while (k < t.size() && (t[k] == '\n' || t[k] == ' ')) k++; t.erase(0, k); if (!t.empty()) lead = false; }
                            emit(t); };
                        const char * reason = ss.generate(max_new, sp, [&](int tok) {
                            if (!raw && tok == ss.tok_think_end && mode == THINK) { flush(true); mode = CONTENT; lead = true; return client_ok; }
                            if (!raw && tok == ss.tok_tool) { flush(true); mode = TOOL; tool_buf.clear(); return client_ok; }
                            if (!raw && tok == ss.tok_tool_end && mode == TOOL) {
                                std::string name, args;
                                if (parse_tool_call(tool_buf, tools, name, args)) {
                                    J tc = J::obj(); tc.set("index", (int) tool_calls.a.size()); tc.set("id", "call_" + std::to_string(req_id) + "_" + std::to_string(tool_calls.a.size())); tc.set("type", "function");
                                    J & fn = tc.set("function", J::obj()); fn.set("name", name); fn.set("arguments", args);
                                    if (stream) { J d = J::obj(); d.set("tool_calls", J::arr()).push(tc); chunk(d, nullptr); }
                                    tool_calls.push(tc);
                                }
                                mode = CONTENT; lead = true; return client_ok;
                            }
                            const std::string pc = tk.piece(tok, raw); pend += pc; full += pc;
                            if (!stops.empty() && mode == CONTENT) { for (auto & st : stops) { const size_t p = (content + pend).find(st); if (p != std::string::npos) { const std::string all = content + pend; pend = all.substr(content.size() > p ? p : content.size(), p > content.size() ? p - content.size() : 0); stop_hit = true; } }
                                if (stop_hit) { flush(true); return false; } }
                            flush(false); return client_ok;
                        });
                        flush(true);
                        if (stop_hit) reason = "stop"; if (!tool_calls.a.empty() && strcmp(reason, "stop") == 0) reason = "tool_calls"; if (!strcmp(reason, "abort")) reason = "stop";
                        J usage = J::obj(); usage.set("prompt_tokens", (int) prompt.size()); usage.set("completion_tokens", (int) ss.st_gen_n); usage.set("total_tokens", (int) (prompt.size() + ss.st_gen_n));
                        J tm = J::obj(); tm.set("prompt_n", (int) ss.st_prompt_n); tm.set("prompt_ms", ss.st_prompt_s * 1e3); tm.set("prompt_per_second", ss.st_prompt_n / std::max(1e-9, ss.st_prompt_s));
                        tm.set("predicted_n", (int) ss.st_gen_n); tm.set("predicted_ms", ss.st_gen_s * 1e3); tm.set("predicted_per_second", ss.st_gen_n / std::max(1e-9, ss.st_gen_s));
                        tm.set("draft_n", (int) ss.st_drafted); tm.set("draft_n_accepted", (int) ss.st_accepted); tm.set("expert_cache_hit_rate", e.cache->hit_rate()); tm.set("lookup_n", (int) ss.st_lk_drafted); tm.set("lookup_n_accepted", (int) ss.st_lk_accepted);
                        printf("req %ld done (%s): prompt %zu tok (%ld new, %.0f t/s), gen %ld tok %.1f t/s, drafts %ld/%ld (lookup %ld/%ld), cache hit %.3f, ctx %d\n", req_id, reason, prompt.size(), ss.st_prompt_n, ss.st_prompt_n / std::max(1e-9, ss.st_prompt_s),
                               ss.st_gen_n, ss.st_gen_n / std::max(1e-9, ss.st_gen_s), ss.st_accepted, ss.st_drafted, ss.st_lk_accepted, ss.st_lk_drafted, e.cache->hit_rate(), e.n_past);
                        e.cache->hits = e.cache->misses = 0;
                        if (raw) { J o = J::obj(); o.set("content", full); J & ta = o.set("tokens", J::arr()); for (size_t i = prompt.size(); i < ss.ctx.size(); i++) ta.push(J(ss.ctx[i])); o.set("timings", tm); o.set("stop_type", reason); respond(cs, 200, o.dump()); }
                        else if (stream) { chunk(J::obj(), reason); J u = J::obj(); u.set("id", id); u.set("object", "chat.completion.chunk"); u.set("created", J(created)); u.set("model", alias); u.set("choices", J::arr()); u.set("usage", usage); u.set("timings", tm);
                            send_all(cs, "data: " + u.dump() + "\n\ndata: [DONE]\n\n"); }
                        else {
                            J o = J::obj(); o.set("id", id); o.set("object", "chat.completion"); o.set("created", J(created)); o.set("model", alias);
                            J & ch = o.set("choices", J::arr()).push(J::obj()); ch.set("index", 0); J & msg = ch.set("message", J::obj()); msg.set("role", "assistant"); msg.set("content", trim(content));
                            if (!reasoning.empty()) msg.set("reasoning_content", trim(reasoning)); if (!tool_calls.a.empty()) msg.set("tool_calls", tool_calls);
                            ch.set("finish_reason", reason); o.set("usage", usage); o.set("timings", tm); respond(cs, 200, o.dump());
                        }
                        if (time(nullptr) - last_save > 120) { e.cache->save_profile(prof); last_save = time(nullptr); }
                    }
                    else respond(cs, 404, err_json("not found"));
                } catch (const sycl::exception & ex) { printf("sycl error: %s\n", ex.what()); respond(cs, 500, err_json(ex.what(), "server_error")); ss.ckpts.clear(); ss.ctx.clear(); e.reset();
                } catch (const std::exception & ex) { printf("request error: %s\n", ex.what()); respond(cs, 400, err_json(ex.what())); }
            }
            shutdown(cs, SD_SEND); closesocket(cs);
        }
    } catch (const sycl::exception & ex) { fprintf(stderr, "sycl error: %s\n", ex.what()); return 1;
    } catch (const std::exception & ex) { fprintf(stderr, "error: %s\n", ex.what()); return 1; }
    return 0;
}
