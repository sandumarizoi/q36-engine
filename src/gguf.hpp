// q36: minimal GGUF reader (read-only, memory-mapped). No dependencies.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace q36 {

enum GType : uint32_t { T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_BF16 = 30 };

struct MMap {
    const uint8_t * p = nullptr; uint64_t n = 0;
#ifdef _WIN32
    HANDLE hf = INVALID_HANDLE_VALUE, hm = nullptr;
#else
    int fd = -1;
#endif
    void open(const std::string & path) {
#ifdef _WIN32
        hf = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hf == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
        LARGE_INTEGER sz; GetFileSizeEx(hf, &sz); n = (uint64_t) sz.QuadPart;
        hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hm) throw std::runtime_error("CreateFileMapping failed");
        p = (const uint8_t *) MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
        if (!p) throw std::runtime_error("MapViewOfFile failed");
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat st; fstat(fd, &st); n = (uint64_t) st.st_size;
        void * m = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m == MAP_FAILED) throw std::runtime_error("mmap failed");
        p = (const uint8_t *) m;
#endif
    }
};

struct GTensor {
    std::string name; uint32_t type = 0; std::vector<uint64_t> dims; uint64_t off = 0; uint64_t nbytes = 0;
    const uint8_t * data = nullptr;
    uint64_t nelem() const { uint64_t e = 1; for (auto d : dims) e *= d; return e; }
};

struct GValue {
    uint32_t type = 0;           // gguf type; 9 = array
    uint32_t atype = 0;
    double num = 0; std::string str;
    std::vector<double> nums; std::vector<std::string> strs;
};

inline uint64_t type_row_bytes(uint32_t t, uint64_t n) {
    switch (t) {
        case T_F32:  return n * 4;
        case T_F16: case T_BF16: return n * 2;
        case T_Q8_0: return n / 32 * 34;
        case T_Q4_K: return n / 256 * 144;
        case T_Q5_K: return n / 256 * 176;
        case T_Q6_K: return n / 256 * 210;
    }
    throw std::runtime_error("unsupported tensor type " + std::to_string(t));
}

struct GGUF {
    MMap mm;
    std::map<std::string, GValue> kv;
    std::vector<GTensor> tensors;
    std::map<std::string, size_t> index;
    uint64_t data_start = 0;

    void open(const std::string & path, bool load_vocab = true) {
        mm.open(path);
        const uint8_t * p = mm.p; uint64_t o = 0;
        auto rd = [&](void * dst, size_t n) { memcpy(dst, p + o, n); o += n; };
        auto r32 = [&]() { uint32_t v; rd(&v, 4); return v; };
        auto r64 = [&]() { uint64_t v; rd(&v, 8); return v; };
        auto rstr = [&]() { uint64_t n = r64(); std::string s((const char *) p + o, n); o += n; return s; };
        auto rnum = [&](uint32_t t) -> double {
            switch (t) {
                case 0: { uint8_t v; rd(&v, 1); return v; }   case 1: { int8_t v; rd(&v, 1); return v; }
                case 2: { uint16_t v; rd(&v, 2); return v; }  case 3: { int16_t v; rd(&v, 2); return v; }
                case 4: { uint32_t v; rd(&v, 4); return v; }  case 5: { int32_t v; rd(&v, 4); return v; }
                case 6: { float v; rd(&v, 4); return v; }     case 7: { uint8_t v; rd(&v, 1); return v; }
                case 10: { uint64_t v; rd(&v, 8); return (double) v; } case 11: { int64_t v; rd(&v, 8); return (double) v; }
                case 12: { double v; rd(&v, 8); return v; }
            }
            throw std::runtime_error("bad gguf value type");
        };
        if (memcmp(p, "GGUF", 4) != 0) throw std::runtime_error("not a GGUF file");
        o = 4; uint32_t ver = r32(); (void) ver; uint64_t nt = r64(), nkv = r64();
        uint64_t align = 32;
        for (uint64_t i = 0; i < nkv; i++) {
            std::string k = rstr(); GValue v; v.type = r32();
            if (v.type == 8) v.str = rstr();
            else if (v.type == 9) {
                v.atype = r32(); uint64_t n = r64();
                bool keep = load_vocab || k.rfind("tokenizer.ggml.", 0) != 0;
                if (v.atype == 8) { if (keep) v.strs.reserve(n); for (uint64_t j = 0; j < n; j++) { auto s = rstr(); if (keep) v.strs.push_back(std::move(s)); } }
                else { if (keep) v.nums.reserve(n); for (uint64_t j = 0; j < n; j++) { double d = rnum(v.atype); if (keep) v.nums.push_back(d); } }
            } else v.num = rnum(v.type);
            if (k == "general.alignment") align = (uint64_t) v.num;
            kv[k] = std::move(v);
        }
        tensors.resize(nt);
        for (uint64_t i = 0; i < nt; i++) {
            GTensor & t = tensors[i]; t.name = rstr(); uint32_t nd = r32(); t.dims.resize(nd);
            for (uint32_t d = 0; d < nd; d++) t.dims[d] = r64();
            t.type = r32(); t.off = r64();
        }
        data_start = (o + align - 1) / align * align;
        for (size_t i = 0; i < tensors.size(); i++) {
            GTensor & t = tensors[i];
            t.data = p + data_start + t.off;
            uint64_t rows = t.nelem() / t.dims[0];
            t.nbytes = type_row_bytes(t.type, t.dims[0]) * rows;
            index[t.name] = i;
        }
    }
    const GTensor * find(const std::string & n) const { auto it = index.find(n); return it == index.end() ? nullptr : &tensors[it->second]; }
    const GTensor & get(const std::string & n) const { auto t = find(n); if (!t) throw std::runtime_error("missing tensor " + n); return *t; }
    double num(const std::string & k, double def = 0) const { auto it = kv.find(k); return it == kv.end() ? def : it->second.num; }
};

} // namespace q36
