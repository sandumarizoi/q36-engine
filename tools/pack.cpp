// q36pack: GGUF (Qwen3.6-35B-A3B, qwen35moe) -> q36 pack (see src/pack.hpp). Lossless re-layout.
//   pack <model.gguf> <out.q36> [--threads T]
#include "../src/model.hpp"
#include "../src/pack.hpp"
using namespace q36;

struct Writer {
    FILE * f = nullptr; uint64_t pos = 0; std::ostringstream toc;
    void open(const std::string & p) { f = fopen(p.c_str(), "wb"); if (!f) throw std::runtime_error("cannot create " + p); std::vector<uint8_t> z(4096, 0); put(z.data(), z.size()); }
    void put(const void * d, size_t n) { if (fwrite(d, 1, n, f) != n) throw std::runtime_error("write failed (disk full?)"); pos += n; }
    void align(uint64_t a) { static const std::vector<uint8_t> z(4096, 0); uint64_t pad = (a - pos % a) % a; if (pad) put(z.data(), (size_t) pad); }
    void entry(const std::string & name, uint32_t type, uint64_t rows, uint32_t n_in, uint32_t row_bytes, const std::vector<uint8_t> & buf) {
        align(4096); toc << "t " << name << ' ' << type << ' ' << rows << ' ' << n_in << ' ' << row_bytes << ' ' << pos << ' ' << buf.size() << '\n';
        put(buf.data(), buf.size());
    }
    void close() {
        align(4096); const std::string t = toc.str(); const uint64_t off = pos, sz = t.size(); put(t.data(), t.size());
#ifdef _WIN32
        _fseeki64(f, 0, SEEK_SET);
#else
        fseeko(f, 0, SEEK_SET);
#endif
        fwrite("Q36PACK1", 1, 8, f); fwrite(&off, 8, 1, f); fwrite(&sz, 8, 1, f); fclose(f);
    }
};

// append rows [r0, r0+n) of matrix m (converted to row-SoA) at buf + off, stride = row stride in the output
static void conv_rows(Pool & pool, const Mat & m, int64_t r0, int64_t n, uint8_t * out, size_t stride) {
    pool.pfor(n, [&](int64_t a, int64_t b) { for (int64_t r = a; r < b; r++) soa_convert_row(m.type, m.n_in, m.data + (r0 + r) * m.row_bytes, out + (size_t) r * stride); });
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: pack model.gguf out.q36 [--threads T]\n"); return 1; }
    int nth = 6; for (int i = 3; i < argc; i++) if (std::string(argv[i]) == "--threads") nth = atoi(argv[++i]);
    try {
        Model m; m.load(argv[1]); const HP & hp = m.hp; Pool pool(nth); Writer w; w.open(argv[2]);
        const double t0 = Engine::now();
        w.toc << "hp n_layer " << hp.n_layer << "\nhp n_embd " << hp.n_embd << "\nhp n_vocab " << hp.n_vocab << "\nhp n_head " << hp.n_head << "\nhp n_head_kv " << hp.n_head_kv
              << "\nhp head_dim " << hp.head_dim << "\nhp n_rot " << hp.n_rot << "\nhp n_expert " << hp.n_expert << "\nhp n_used " << hp.n_used << "\nhp n_ff " << hp.n_ff
              << "\nhp gdn_kh " << hp.gdn_kh << "\nhp gdn_vh " << hp.gdn_vh << "\nhp gdn_d " << hp.gdn_d << "\nhp conv_k " << hp.conv_k << "\nhp eps " << hp.eps << "\nhp rope_base " << hp.rope_base
              << "\nhp has_mtp " << (m.has_mtp() ? 1 : 0) << "\n";
        if (hp.n_embd != 2048 || hp.qkv_dim() != 8192 || hp.conv_k != 4 || hp.head_dim != 256 || hp.n_ff != 512) throw std::runtime_error("unexpected model shape");
        std::vector<uint8_t> buf;
        auto fused = [&](const std::string & name, std::initializer_list<Mat> mats) { // same type and n_in, rows concatenated
            const Mat & f = *mats.begin(); const uint32_t rb = soa_row_bytes(f.type, f.n_in); uint64_t rows = 0; for (auto & x : mats) rows += x.n_out;
            buf.assign((size_t) rows * rb, 0); uint64_t r = 0;
            for (auto & x : mats) { if (x.type != f.type || x.n_in != f.n_in) throw std::runtime_error("fused: mismatch in " + name); conv_rows(pool, x, 0, x.n_out, buf.data() + (size_t) r * rb, rb); r += x.n_out; }
            w.entry(name, f.type, rows, f.n_in, rb, buf);
        };
        auto floats = [&](const std::string & name, const std::vector<float> & v, uint64_t rows, uint32_t n_in) {
            buf.assign((const uint8_t *) v.data(), (const uint8_t *) v.data() + v.size() * 4); w.entry(name, T_F32, rows, n_in, n_in * 4, buf);
        };
        auto frow = [&](const Mat & x, std::vector<float> & dst) { // append all rows as f32
            for (int64_t r = 0; r < x.n_out; r++) { size_t o = dst.size(); dst.resize(o + x.n_in); dequant_row(x, r, dst.data() + o); }
        };
        const int N = hp.n_embd;
        for (int il = 0; il < (int) m.layers.size(); il++) {
            const Layer & L = m.layers[il]; const std::string p = std::to_string(il) + "."; const bool mtp = il == hp.n_layer;
            std::vector<float> par; par.insert(par.end(), L.attn_norm, L.attn_norm + N); par.insert(par.end(), L.post_norm, L.post_norm + N);
            if (L.recr) {
                fused(p + "qkvz", { L.qkv, L.zgate }); fused(p + "out", { L.ssm_out });
                std::vector<float> ab; frow(L.alpha, ab); frow(L.beta, ab); floats(p + "ab", ab, 64, N);
                for (int t = 0; t < 4; t++) for (int ch = 0; ch < 8192; ch++) par.push_back(L.conv[t + ch * 4]);
                par.insert(par.end(), L.dt_bias, L.dt_bias + 32); par.insert(par.end(), L.ssm_a, L.ssm_a + 32); par.insert(par.end(), L.ssm_norm, L.ssm_norm + 128);
            } else {
                fused(p + "qkv", { L.wq, L.wk, L.wv }); fused(p + "out", { L.wo });
                par.insert(par.end(), L.q_norm, L.q_norm + 256); par.insert(par.end(), L.k_norm, L.k_norm + 256);
                if (mtp) { fused(p + "eh_proj", { L.eh_proj }); par.insert(par.end(), L.enorm, L.enorm + N); par.insert(par.end(), L.hnorm, L.hnorm + N); par.insert(par.end(), L.head_norm, L.head_norm + N); }
            }
            floats(p + "params", par, 1, (uint32_t) par.size());
            { std::vector<float> r; frow(L.router, r); frow(L.sh_gate_inp, r); floats(p + "router", r, 257, N); }
            { // shared expert: gate/up rows interleaved, then down
                const uint32_t rb = soa_row_bytes(L.sh_gate.type, N); buf.assign((size_t) 2 * hp.n_ff * rb, 0);
                conv_rows(pool, L.sh_gate, 0, hp.n_ff, buf.data(), 2 * (size_t) rb); conv_rows(pool, L.sh_up, 0, hp.n_ff, buf.data() + rb, 2 * (size_t) rb);
                w.entry(p + "sh_gu", L.sh_gate.type, 2 * hp.n_ff, N, rb, buf); fused(p + "sh_down", { L.sh_down });
            }
            { // routed experts: one record per expert
                if (L.e_gate.type != L.e_up.type) throw std::runtime_error("expert gate/up type mismatch");
                const uint32_t rg = soa_row_bytes(L.e_gate.type, N), rd = soa_row_bytes(L.e_down.type, hp.n_ff);
                const size_t rec = ((size_t) 2 * hp.n_ff * rg + (size_t) N * rd + 63) / 64 * 64;
                buf.assign(rec * hp.n_expert, 0);
                pool.pfor(hp.n_expert, [&](int64_t a, int64_t b) {
                    for (int64_t e = a; e < b; e++) {
                        uint8_t * o = buf.data() + (size_t) e * rec;
                        for (int i = 0; i < hp.n_ff; i++) {
                            soa_convert_row(L.e_gate.type, N, L.e_gate.data + (e * hp.n_ff + i) * L.e_gate.row_bytes, o + (size_t) (2 * i) * rg);
                            soa_convert_row(L.e_up.type,   N, L.e_up.data   + (e * hp.n_ff + i) * L.e_up.row_bytes,   o + (size_t) (2 * i + 1) * rg);
                        }
                        uint8_t * od = o + (size_t) 2 * hp.n_ff * rg;
                        for (int r = 0; r < N; r++) soa_convert_row(L.e_down.type, hp.n_ff, L.e_down.data + (e * N + r) * L.e_down.row_bytes, od + (size_t) r * rd);
                    }
                });
                // rows = experts, n_in = gate/up type << 8 | down type, row_bytes = record size
                w.entry(p + "exp", (L.e_gate.type << 8) | L.e_down.type, hp.n_expert, rg << 16 | rd, (uint32_t) rec, buf);
            }
            printf("layer %2d %s  %.1f GB written, %.0fs\n", il, L.recr ? "gdn " : (mtp ? "mtp " : "attn"), w.pos / 1e9, Engine::now() - t0); fflush(stdout);
        }
        fused("head", { m.output });
        fused("tok_embd", { m.tok_embd });
        { std::vector<float> g(m.output_norm, m.output_norm + N); floats("g.params", g, 1, N); }
        w.close();
        printf("done: %.2f GB in %.0fs\n", w.pos / 1e9, Engine::now() - t0);
    } catch (const std::exception & ex) { fprintf(stderr, "error: %s\n", ex.what()); return 1; }
    return 0;
}
