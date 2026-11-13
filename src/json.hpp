// q36: small JSON value (insertion-ordered objects), parser and writer.
#pragma once
#include <string>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace q36 {

struct J {
    enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
    bool b = false; double n = 0; std::string s; std::vector<J> a; std::vector<std::pair<std::string, J>> o;
    J() {}
    J(bool v) : t(Bool), b(v) {}
    J(double v) : t(Num), n(v) {}
    J(int v) : t(Num), n(v) {}
    J(long long v) : t(Num), n((double) v) {}
    J(const char * v) : t(Str), s(v) {}
    J(const std::string & v) : t(Str), s(v) {}
    static J arr() { J j; j.t = Arr; return j; }
    static J obj() { J j; j.t = Obj; return j; }
    bool is_null() const { return t == Null; } bool is_str() const { return t == Str; } bool is_obj() const { return t == Obj; } bool is_arr() const { return t == Arr; }
    const J * find(const std::string & k) const { if (t == Obj) for (auto & kv : o) if (kv.first == k) return &kv.second; return nullptr; }
    const J & at(const std::string & k) const { static const J nul; const J * p = find(k); return p ? *p : nul; }
    J & set(const std::string & k, J v) { t = Obj; for (auto & kv : o) if (kv.first == k) { kv.second = std::move(v); return kv.second; } o.emplace_back(k, std::move(v)); return o.back().second; }
    J & push(J v) { t = Arr; a.push_back(std::move(v)); return a.back(); }
    std::string str(const std::string & def = "") const { return t == Str ? s : def; }
    double num(double def = 0) const { return t == Num ? n : (t == Bool ? (b ? 1 : 0) : def); }
    bool boolean(bool def = false) const { return t == Bool ? b : (t == Num ? n != 0 : def); }

    static void esc(const std::string & s, std::string & o) {
        o += '"';
        for (unsigned char c : s) {
            switch (c) {
                case '"': o += "\\\""; break; case '\\': o += "\\\\"; break; case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
                case '\b': o += "\\b"; break; case '\f': o += "\\f"; break;
                default: if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char) c;
            }
        }
        o += '"';
    }
    // py = Python json.dumps separators (", " and ": "), as Jinja's tojson prints
    void dump(std::string & out, bool py = false) const {
        switch (t) {
            case Null: out += "null"; break;
            case Bool: out += b ? "true" : "false"; break;
            case Num: { char buf[40]; if (std::floor(n) == n && std::fabs(n) < 1e15) snprintf(buf, sizeof buf, "%lld", (long long) n); else snprintf(buf, sizeof buf, "%.17g", n); out += buf; } break;
            case Str: esc(s, out); break;
            case Arr: out += '['; for (size_t i = 0; i < a.size(); i++) { if (i) out += py ? ", " : ","; a[i].dump(out, py); } out += ']'; break;
            case Obj: out += '{'; for (size_t i = 0; i < o.size(); i++) { if (i) out += py ? ", " : ","; esc(o[i].first, out); out += py ? ": " : ":"; o[i].second.dump(out, py); } out += '}'; break;
        }
    }
    std::string dump(bool py = false) const { std::string o; dump(o, py); return o; }

    struct P {
        const char * p, * e;
        void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++; }
        [[noreturn]] void fail(const char * m) { throw std::runtime_error(std::string("JSON: ") + m); }
        static void utf8(std::string & o, unsigned cp) {
            if (cp < 0x80) o += (char) cp; else if (cp < 0x800) { o += (char) (0xC0 | (cp >> 6)); o += (char) (0x80 | (cp & 0x3F)); }
            else if (cp < 0x10000) { o += (char) (0xE0 | (cp >> 12)); o += (char) (0x80 | ((cp >> 6) & 0x3F)); o += (char) (0x80 | (cp & 0x3F)); }
            else { o += (char) (0xF0 | (cp >> 18)); o += (char) (0x80 | ((cp >> 12) & 0x3F)); o += (char) (0x80 | ((cp >> 6) & 0x3F)); o += (char) (0x80 | (cp & 0x3F)); }
        }
        unsigned hex4() { if (e - p < 4) fail("bad \\u"); unsigned v = 0; for (int i = 0; i < 4; i++) { char c = *p++; v = v * 16 + (c >= '0' && c <= '9' ? c - '0' : (c | 0x20) >= 'a' && (c | 0x20) <= 'f' ? (c | 0x20) - 'a' + 10 : (fail("bad hex"), 0)); } return v; }
        std::string string() {
            std::string o; p++;
            while (p < e && *p != '"') {
                if (*p == '\\') {
                    p++; if (p >= e) fail("bad escape");
                    char c = *p++;
                    switch (c) {
                        case 'n': o += '\n'; break; case 't': o += '\t'; break; case 'r': o += '\r'; break; case 'b': o += '\b'; break; case 'f': o += '\f'; break;
                        case 'u': { unsigned cp = hex4(); if (cp >= 0xD800 && cp < 0xDC00 && e - p >= 6 && p[0] == '\\' && p[1] == 'u') { p += 2; unsigned lo = hex4(); cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); } utf8(o, cp); } break;
                        default: o += c;
                    }
                } else o += *p++;
            }
            if (p >= e) fail("unterminated string");
            p++; return o;
        }
        J value(int depth = 0) {
            if (depth > 200) fail("too deep");
            ws(); if (p >= e) fail("unexpected end");
            J j;
            if (*p == '{') { p++; j.t = Obj; ws(); if (p < e && *p == '}') { p++; return j; }
                for (;;) { ws(); if (p >= e || *p != '"') fail("expected key"); std::string k = string(); ws(); if (p >= e || *p != ':') fail("expected :"); p++; j.o.emplace_back(std::move(k), value(depth + 1)); ws(); if (p < e && *p == ',') { p++; continue; } if (p < e && *p == '}') { p++; return j; } fail("expected , or }"); } }
            if (*p == '[') { p++; j.t = Arr; ws(); if (p < e && *p == ']') { p++; return j; }
                for (;;) { j.a.push_back(value(depth + 1)); ws(); if (p < e && *p == ',') { p++; continue; } if (p < e && *p == ']') { p++; return j; } fail("expected , or ]"); } }
            if (*p == '"') { j.t = Str; j.s = string(); return j; }
            if (e - p >= 4 && !strncmp(p, "true", 4)) { p += 4; return J(true); }
            if (e - p >= 5 && !strncmp(p, "false", 5)) { p += 5; return J(false); }
            if (e - p >= 4 && !strncmp(p, "null", 4)) { p += 4; return j; }
            char * end = nullptr; std::string tmp(p, (size_t) std::min<ptrdiff_t>(e - p, 64)); double v = strtod(tmp.c_str(), &end);
            if (end == tmp.c_str()) fail("bad value");
            p += end - tmp.c_str(); return J(v);
        }
    };
    static J parse(const std::string & s) { P p{ s.data(), s.data() + s.size() }; J j = p.value(); p.ws(); if (p.p != p.e) p.fail("trailing data"); return j; }
    static bool try_parse(const std::string & s, J & out) { try { out = parse(s); return true; } catch (...) { return false; } }
};

} // namespace q36
