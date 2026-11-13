// mktok: extract the tokenizer of a GGUF into the q36 sidecar file.   mktok model.gguf out.tok
#include "../src/gguf.hpp"
using namespace q36;
int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: mktok model.gguf out.tok\n"); return 1; }
    try {
        GGUF g; g.open(argv[1], true); FILE * f = fopen(argv[2], "wb"); if (!f) throw std::runtime_error("cannot create output");
        auto w32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); }; auto ws = [&](const std::string & s) { w32((uint32_t) s.size()); fwrite(s.data(), 1, s.size(), f); };
        const auto & toks = g.kv.at("tokenizer.ggml.tokens").strs; const auto & types = g.kv.at("tokenizer.ggml.token_type").nums; const auto & merges = g.kv.at("tokenizer.ggml.merges").strs;
        w32(0x4B543633); w32((uint32_t) toks.size());
        for (size_t i = 0; i < toks.size(); i++) { ws(toks[i]); w32((uint32_t) types[i]); }
        w32((uint32_t) merges.size()); for (auto & m : merges) ws(m);
        ws(g.kv.at("tokenizer.chat_template").str); w32((uint32_t) g.num("tokenizer.ggml.eos_token_id")); fclose(f);
        printf("tokens %zu merges %zu\n", toks.size(), merges.size());
    } catch (const std::exception & e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
    return 0;
}
