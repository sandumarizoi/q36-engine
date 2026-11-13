// q36 tokenizer: byte-level BPE of Qwen3.6 (gpt2 vocab, "qwen35" pre-tokenizer), written to behave like llama.cpp.
// Data comes from a small sidecar file made from the GGUF (tools/mktok.cpp).
#pragma once
#include "unicode_tables.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <stdexcept>

namespace q36 {

struct Tokenizer {
    std::vector<std::string> tokens; std::vector<int> types; // 1 normal, 3 control, 4 user defined
    std::unordered_map<std::string, int> tok2id, rank;       // rank: "left right" -> merge priority
    std::vector<std::pair<std::string, int>> specials;       // longest first
    std::string chat_template; int eos = -1, eot = -1;

    static bool in_ranges(const uint32_t (*t)[2], size_t n, uint32_t cp) {
        size_t lo = 0, hi = n;
        while (lo < hi) { size_t m = (lo + hi) / 2; if (cp > t[m][1]) lo = m + 1; else hi = m; }
        return lo < n && cp >= t[lo][0];
    }
    static bool isL(uint32_t c) { return in_ranges(uni_L, sizeof(uni_L) / sizeof(uni_L[0]), c); }
    static bool isM(uint32_t c) { return in_ranges(uni_M, sizeof(uni_M) / sizeof(uni_M[0]), c); }
    static bool isN(uint32_t c) { return in_ranges(uni_N, sizeof(uni_N) / sizeof(uni_N[0]), c); }
    static bool isS(uint32_t c) { // llama.cpp unicode whitespace set
        return (c >= 0x9 && c <= 0xD) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
    }
    static void utf8_decode(const std::string & s, std::vector<uint32_t> & cps, std::vector<uint32_t> & offs) {
        size_t i = 0; const size_t n = s.size();
        while (i < n) {
            const unsigned char c = (unsigned char) s[i]; uint32_t cp = c; int len = 1;
            if (c >= 0xF0 && i + 3 < n) { cp = ((c & 7) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F); len = 4; }
            else if (c >= 0xE0 && c < 0xF0 && i + 2 < n) { cp = ((c & 15) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F); len = 3; }
            else if (c >= 0xC0 && c < 0xE0 && i + 1 < n) { cp = ((c & 31) << 6) | (s[i + 1] & 0x3F); len = 2; }
            cps.push_back(cp); offs.push_back((uint32_t) i); i += len;
        }
        offs.push_back((uint32_t) n);
    }
    static void put_utf8(std::string & o, uint32_t cp) {
        if (cp < 0x80) o += (char) cp;
        else if (cp < 0x800) { o += (char) (0xC0 | (cp >> 6)); o += (char) (0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { o += (char) (0xE0 | (cp >> 12)); o += (char) (0x80 | ((cp >> 6) & 0x3F)); o += (char) (0x80 | (cp & 0x3F)); }
        else { o += (char) (0xF0 | (cp >> 18)); o += (char) (0x80 | ((cp >> 12) & 0x3F)); o += (char) (0x80 | ((cp >> 6) & 0x3F)); o += (char) (0x80 | (cp & 0x3F)); }
    }
    // GPT-2 byte <-> unicode mapping
    uint32_t b2u[256]; std::unordered_map<uint32_t, uint8_t> u2b;
    void init_bytes() {
        int n = 0;
        for (int b = 0; b < 256; b++) {
            const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            b2u[b] = keep ? (uint32_t) b : (uint32_t) (256 + n++); u2b[b2u[b]] = (uint8_t) b;
        }
    }

    void load(const std::string & path) {
        FILE * f = fopen(path.c_str(), "rb"); if (!f) throw std::runtime_error("cannot open tokenizer file " + path);
        auto r32 = [&]() { uint32_t v = 0; if (fread(&v, 4, 1, f) != 1) throw std::runtime_error("tokenizer file truncated"); return v; };
        auto rs = [&]() { uint32_t n = r32(); std::string s(n, 0); if (n && fread(&s[0], 1, n, f) != n) throw std::runtime_error("tokenizer file truncated"); return s; };
        if (r32() != 0x4B543633) throw std::runtime_error("bad tokenizer file");
        const uint32_t nt = r32(); tokens.resize(nt); types.resize(nt);
        for (uint32_t i = 0; i < nt; i++) { tokens[i] = rs(); types[i] = (int) r32(); }
        const uint32_t nm = r32(); rank.reserve(nm * 2);
        for (uint32_t i = 0; i < nm; i++) rank.emplace(rs(), (int) i);
        chat_template = rs(); eos = (int) r32(); fclose(f);
        tok2id.reserve(nt * 2);
        for (uint32_t i = 0; i < nt; i++) { tok2id.emplace(tokens[i], (int) i); if (types[i] == 3 || types[i] == 4) specials.push_back({ tokens[i], (int) i }); }
        std::sort(specials.begin(), specials.end(), [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });
        auto it = tok2id.find("<|im_end|>"); eot = it == tok2id.end() ? -1 : it->second;
        init_bytes();
    }

    // qwen35 pre-tokenizer:
    // (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    static size_t match(const std::vector<uint32_t> & c, size_t i) {
        const size_t n = c.size(); auto lo = [&](size_t k) { return k < n ? (c[k] | 0x20) : 0u; };
        if (c[i] == '\'' && i + 1 < n) {
            const uint32_t a = lo(i + 1);
            if (c[i + 1] < 128 && (a == 's' || a == 't' || a == 'm' || a == 'd')) return 2;
            if (i + 2 < n && c[i + 1] < 128 && c[i + 2] < 128 && ((a == 'r' && lo(i + 2) == 'e') || (a == 'v' && lo(i + 2) == 'e') || (a == 'l' && lo(i + 2) == 'l'))) return 3;
        }
        auto LM = [&](size_t k) { return k < n && (isL(c[k]) || isM(c[k])); };
        { // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            const bool opt = c[i] != '\r' && c[i] != '\n' && !isL(c[i]) && !isN(c[i]);
            if (opt && LM(i + 1)) { size_t k = i + 1; while (LM(k)) k++; return k - i; }
            if (LM(i)) { size_t k = i; while (LM(k)) k++; return k - i; }
        }
        if (isN(c[i])) return 1;
        { // ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*'
            auto P = [&](size_t k) { return k < n && !isS(c[k]) && !isL(c[k]) && !isM(c[k]) && !isN(c[k]); };
            size_t k = i; if (c[k] == ' ' && P(k + 1)) k++;
            if (P(k)) { while (P(k)) k++; while (k < n && (c[k] == '\r' || c[k] == '\n')) k++; return k - i; }
        }
        if (isS(c[i])) {
            size_t e = i; while (e < n && isS(c[e])) e++;
            size_t last = 0; bool has = false; for (size_t k = i; k < e; k++) if (c[k] == '\r' || c[k] == '\n') { last = k; has = true; }
            if (has) return last + 1 - i;            // \s*[\r\n]+
            if (e == n) return e - i;                // \s+(?!\S) at the end of the text
            if (e - i >= 2) return e - i - 1;        // \s+(?!\S): leave one space for the next word
            return e - i;                            // \s+
        }
        return 1;
    }

    void bpe_word(const std::string & w, std::vector<int> & out) const {
        std::vector<std::string> sym; sym.reserve(w.size());
        for (unsigned char b : w) { std::string s; put_utf8(s, b2u[b]); sym.push_back(std::move(s)); }
        while (sym.size() > 1) {
            int best = 0x7fffffff; size_t bi = 0;
            for (size_t k = 0; k + 1 < sym.size(); k++) { auto it = rank.find(sym[k] + " " + sym[k + 1]); if (it != rank.end() && it->second < best) { best = it->second; bi = k; } }
            if (best == 0x7fffffff) break;
            const std::string l = sym[bi], r = sym[bi + 1]; std::vector<std::string> nw; nw.reserve(sym.size());
            for (size_t k = 0; k < sym.size();) { if (k + 1 < sym.size() && sym[k] == l && sym[k + 1] == r) { nw.push_back(l + r); k += 2; } else nw.push_back(sym[k++]); }
            sym.swap(nw);
        }
        for (auto & s : sym) {
            auto it = tok2id.find(s);
            if (it != tok2id.end()) out.push_back(it->second);
            else { std::vector<uint32_t> cps, offs; utf8_decode(s, cps, offs); for (uint32_t cp : cps) { std::string one; put_utf8(one, cp); auto j = tok2id.find(one); if (j != tok2id.end()) out.push_back(j->second); } }
        }
    }
    void encode_plain(const std::string & text, std::vector<int> & out) const {
        if (text.empty()) return;
        std::vector<uint32_t> cps, offs; utf8_decode(text, cps, offs);
        for (size_t i = 0; i < cps.size();) { const size_t len = match(cps, i); bpe_word(text.substr(offs[i], offs[i + len] - offs[i]), out); i += len; }
    }
    // special tokens in the text are mapped to their ids when parse_special
    std::vector<int> encode(const std::string & text, bool parse_special = true) const {
        std::vector<int> out; if (!parse_special) { encode_plain(text, out); return out; }
        size_t start = 0, i = 0;
        while (i < text.size()) {
            if (text[i] == '<') {
                bool hit = false;
                for (auto & sp : specials) if (text.compare(i, sp.first.size(), sp.first) == 0) {
                    encode_plain(text.substr(start, i - start), out); out.push_back(sp.second); i += sp.first.size(); start = i; hit = true; break;
                }
                if (hit) continue;
            }
            i++;
        }
        encode_plain(text.substr(start), out);
        return out;
    }
    // raw bytes of a token (may be a partial UTF-8 sequence)
    std::string piece(int id, bool special = false) const {
        if (id < 0 || id >= (int) tokens.size()) return "";
        if (types[id] == 3) return special ? tokens[id] : "";
        if (types[id] == 4) return tokens[id];
        std::vector<uint32_t> cps, offs; utf8_decode(tokens[id], cps, offs); std::string o;
        for (uint32_t cp : cps) { auto it = u2b.find(cp); if (it != u2b.end()) o += (char) it->second; else put_utf8(o, cp); }
        return o;
    }
};

} // namespace q36
