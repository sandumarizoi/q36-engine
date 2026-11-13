// q36 CLI (development / benchmarking): feeds token ids, generates greedily, prints top logprobs and timings.
//   q36 <model.q36> --tokens <ids file> [--gen N] [--ctx C] [--threads T] [--top K]
#include "../src/engine.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <iterator>
using namespace q36;

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: q36 model.q36 --tokens file [--gen N] [--ctx C] [--threads T] [--top K]\n"); return 1; }
    std::string pack_path = argv[1], tok_path; int n_gen = 0, n_threads = 6, top = 5, ctx = 8192, max_tokens = 1 << 30, rate = 4, mtp = 0; int pf_win = 0; bool quiet = false, prefault = false, kprof = false, nograph = false; double cache_gb = 0.0; std::string prof, dv_extra; int dv_keep = 0; bool adaptive = false, nosave = false; double prior = 60.0;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--tokens") tok_path = argv[++i]; else if (a == "--gen") n_gen = atoi(argv[++i]); else if (a == "--threads") n_threads = atoi(argv[++i]);
        else if (a == "--top") top = atoi(argv[++i]); else if (a == "--ctx") ctx = atoi(argv[++i]); else if (a == "--max-tokens") max_tokens = atoi(argv[++i]); else if (a == "--quiet") quiet = true; else if (a == "--prefault") prefault = true; else if (a == "--kprof") kprof = true; else if (a == "--no-graph") nograph = true; else if (a == "--prefill-window") pf_win = atoi(argv[++i]); else if (a == "--mtp") mtp = atoi(argv[++i]); else if (a == "--draft-vocab") dv_keep = atoi(argv[++i]); else if (a == "--draft-vocab-extra") dv_extra = argv[++i]; else if (a == "--adaptive") adaptive = true; else if (a == "--prior") prior = atof(argv[++i]); else if (a == "--no-save") nosave = true; else if (a == "--cache-gb") cache_gb = atof(argv[++i]); else if (a == "--rate") rate = atoi(argv[++i]); else if (a == "--profile") prof = argv[++i];
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        double t0 = now_s(); Pack pk; pk.open(pack_path); Pool pool(n_threads);
        if (prefault) { double tp = now_s(); std::atomic<uint64_t> sum{0}; const uint64_t n = pk.mm.n; const uint8_t * p = pk.mm.p;
            pool.pfor((int64_t) (n / 4096), [&](int64_t a, int64_t b) { uint64_t s = 0; for (int64_t i = a; i < b; i++) s += p[(uint64_t) i * 4096]; sum += s; });
            printf("prefault: %.1f GB in %.1fs\n", n / 1e9, now_s() - tp); fflush(stdout); }
        Engine e(pk, pool, ctx, cache_gb, pf_win);
        e.cache->rate = rate; e.use_graph = !nograph;
        if (dv_keep > 0 && mtp > 0) { e.set_draft_vocab(dv_keep, dv_extra); printf("draft vocabulary: %d of %d rows\n", e.n_draft_rows, e.n_vocab); }
        e.cache->prior_tokens = prior;
        const bool have_prof = !prof.empty() && e.cache->load_profile(prof);
        double tp = now_s(); e.cache->prefill();
        printf("expert cache: %.2f GB, %d + %d slots (%.1f%% of experts), profile %s, prefill %.1fs\n", e.cache->bytes / 1e9, e.cache->cls[0].cap, e.cache->cls[1].cap,
               100.0 * (e.cache->cls[0].cap + e.cache->cls[1].cap) / (e.cache->nl * 256.0), have_prof ? "loaded" : "none", now_s() - tp);
        printf("device: %s | VRAM used %.2f GB | load %.1fs\n", e.q.get_device().get_info<sycl::info::device::name>().c_str(), e.vram / 1e9, now_s() - t0);
        std::vector<int> toks; { std::ifstream f(tok_path); std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            for (auto & c : all) if (c < '0' || c > '9') c = ' ';
            std::istringstream is(all); int t; while (is >> t) { if ((int) toks.size() >= max_tokens) break; toks.push_back(t); } }
        if (toks.empty()) { fprintf(stderr, "no tokens\n"); return 1; }
        std::vector<float> logits((size_t) q36g::NCMAX * e.n_vocab);
        const size_t n_prompt = toks.size(); t0 = now_s(); double t_prompt = 0;
        auto argmax = [&](const float * l) { return (int) (std::max_element(l, l + e.n_vocab) - l); };
        auto show = [&](size_t i, int tok, const float * l) {
            int best = argmax(l); const float mx = l[best]; double sum = 0; for (int v = 0; v < e.n_vocab; v++) sum += exp((double) l[v] - mx);
            std::vector<int> ord(e.n_vocab); std::iota(ord.begin(), ord.end(), 0);
            std::partial_sort(ord.begin(), ord.begin() + top, ord.end(), [&](int a, int b) { return l[a] > l[b]; });
            printf("pos %zu in %d ->", i, tok); for (int k = 0; k < top; k++) printf(" %d:%.4f", ord[k], (float) (l[ord[k]] - mx - log(sum))); printf("\n");
        };
        long n_drafted = 0, n_accepted = 0, n_iters = 0; double t_draft = 0;
        const bool use_mtp = mtp > 0 && e.has_mtp;
        // prompt: one token per step (batched prefill is not implemented yet); the MTP block follows with K/V only
        size_t i0 = 0;
        if (e.pf_window > 0) { // long prompt: GPU windows (the last token goes through a normal step for its logits)
            const bool gp = e.gprof; e.gprof = false;
            while (n_prompt - 1 - i0 >= 64) { const int W = (int) std::min<size_t>((size_t) e.pf_window, (n_prompt - 1 - i0) / 8 * 8); e.prefill_window(toks.data() + i0, W, use_mtp, [](int, int) {}); i0 += W; }
            e.prefill_end(); e.gprof = gp; e.gev.clear(); e.gsum.clear(); e.g_busy = e.g_span = 0; e.g_steps = 0;
        }
        for (size_t i = i0; i < n_prompt; i++) {
            if (i % 500 == 499) printf("  .. %zu tokens, %.1f t/s, hit rate %.3f, uploads %ld\n", i + 1, (i + 1) / (now_s() - t0), e.cache->hit_rate(), e.cache->uploads);
            e.embed(toks[i], e.h_emb);
            const bool last = i + 1 == n_prompt;
            e.step(1, last, logits.data()); e.commit(1);
            if (use_mtp) {
                memcpy(e.h_embm, e.h_emb, 2048 * 4); e.q.memcpy(e.HIN, e.Hp, 2048 * 4);
                e.mtp_step(1, (int) i, true, nullptr); e.q.memcpy(e.Hp, e.HT, 2048 * 4);
            }
        }
        t_prompt = now_s() - t0; e.t_gpu_wait = e.t_cpu = e.t_submit = e.t_head = e.t_maint = 0; e.prof = kprof; e.cache->hits = e.cache->misses = 0;
        if (!quiet) show(n_prompt - 1, toks[n_prompt - 1], logits.data());
        // generation. Invariant at the top: tokens [0, n) are decoded, Hp = hidden of position n-1 (for MTP), `next` is the
        // token for position n. Without MTP: plain one-token steps.
        int next = argmax(logits.data()); const int dmax = std::min(mtp, q36g::NCMAX - 1); int depth = adaptive ? std::min(2, dmax) : dmax; long depth_sum = 0;
        std::vector<float> dl(e.n_vocab);
        while ((int) (toks.size() - n_prompt) < n_gen) {
            const int n = (int) toks.size(); n_iters++;
            int drafts[q36g::NCMAX]; int nd = 0;
            if (use_mtp) {
                const double td = now_s();
                int tok = next;
                for (int k = 0; k < depth; k++) {
                    e.embed(tok, e.h_embm);
                    e.q.memcpy(e.HIN, k == 0 ? e.Hp : e.HM, 2048 * 4);
                    e.mtp_step(1, n + k, false, dl.data());
                    int best = 0; { const int nr = e.draft_rows_d ? e.n_draft_rows : e.n_vocab; best = (int) (std::max_element(dl.begin(), dl.begin() + nr) - dl.begin()); if (e.draft_rows_d) best = e.draft_rows_h[best]; }
                    drafts[nd++] = best; tok = best;
                }
                t_draft += now_s() - td;
            }
            // verify: [next, drafts...] in one trunk step
            const int nc = 1 + nd;
            e.embed(next, e.h_emb); for (int k = 0; k < nd; k++) e.embed(drafts[k], e.h_emb + (size_t) (k + 1) * 2048);
            e.step(nc, true, logits.data());
            toks.push_back(next);
            int acc = 0; int samp = argmax(logits.data());
            if (!quiet) show(n, next, logits.data());
            while (acc < nd && samp == drafts[acc] && (int) (toks.size() - n_prompt) < n_gen) {
                toks.push_back(samp); acc++; samp = argmax(logits.data() + (size_t) acc * e.n_vocab);
                if (!quiet) show(n + acc, toks.back(), logits.data() + (size_t) acc * e.n_vocab);
            }
            e.commit(1 + acc); n_drafted += nd; n_accepted += acc; depth_sum += nd;
            if (adaptive && nd > 0) { if (acc == nd) depth = std::min(depth + 1, dmax); else if (2 * acc < nd) depth = std::max(1, depth - 1); }
            if (use_mtp) {
                // MTP K/V of the accepted draft positions must hold the true pairs (draft token, trunk hidden before it)
                if (acc > 0) { for (int k = 0; k < acc; k++) e.embed(drafts[k], e.h_embm + (size_t) k * 2048); e.q.memcpy(e.HIN, e.HT, (size_t) acc * 2048 * 4); e.mtp_step(acc, n + 1, true, nullptr); }
                e.q.memcpy(e.Hp, e.HT + (size_t) acc * 2048, 2048 * 4);
            }
            next = samp;
        }
        const int n_out = (int) (toks.size() - n_prompt);
        const double t_gen = now_s() - t0 - t_prompt;
        printf("generated:"); for (size_t i = n_prompt; i < toks.size(); i++) printf(" %d", toks[i]); printf("\n");
        printf("time: prompt %zu tok %.2fs (%.1f t/s), gen %d tok %.2fs (%.2f t/s) | per gen token: submit %.2f ms, wait gpu %.2f ms, cpu experts %.2f ms, head %.2f ms\n",
               n_prompt, t_prompt, n_prompt / std::max(1e-9, t_prompt), n_out, t_gen, n_out / std::max(1e-9, t_gen),
               e.t_submit / std::max(1, n_out) * 1e3, e.t_gpu_wait / std::max(1, n_out) * 1e3, e.t_cpu / std::max(1, n_out) * 1e3, e.t_head / std::max(1, n_out) * 1e3);
        if (use_mtp) printf("mtp: depth %s%d (avg %.2f), %ld steps, drafted %ld, accepted %ld (%.1f%%), %.2f tokens per step, draft time %.2f ms per step\n", adaptive ? "adaptive up to " : "", dmax, (double) depth_sum / std::max(1L, n_iters), n_iters, n_drafted, n_accepted,
                            100.0 * n_accepted / std::max(1L, n_drafted), (double) n_out / std::max(1L, n_iters), t_draft / std::max(1L, n_iters) * 1e3);
        printf("maintain %.2f ms/token\n", e.t_maint / std::max(1, n_out) * 1e3);
        if (kprof) e.print_prof(n_out);
        if (e.gprof) e.gprof_print(n_out);
        printf("cache: hit rate %.3f (gen phase), uploads %ld\n", e.cache->hit_rate(), e.cache->uploads);
        if (!prof.empty() && !nosave) e.cache->save_profile(prof);
    } catch (const sycl::exception & ex) { fprintf(stderr, "sycl error: %s\n", ex.what()); return 1;
    } catch (const std::exception & ex) { fprintf(stderr, "error: %s\n", ex.what()); return 1; }
    return 0;
}
