// llama-safetensors.cpp — minimal safetensors reader for the metal-exl3 fork.
#include "llama-safetensors.h"

#include <cstdio>
#include <cstring>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

bool st_dtype_size(const std::string & dtype, size_t & out) {
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") { out = 2; return true; }
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") { out = 4; return true; }
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") { out = 8; return true; }
    if (dtype == "I8" || dtype == "U8" || dtype == "BOOL") { out = 1; return true; }
    return false;
}

static bool read_exact(FILE * f, void * dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}

bool st_read_header(const std::string & path, st_file & out, std::string & err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    uint64_t hlen = 0;
    if (!read_exact(f, &hlen, 8)) { fclose(f); err = "short header length in " + path; return false; }
    // The header is JSON; guard against absurd lengths before allocating.
    if (hlen == 0 || hlen > (uint64_t) 1 << 30) {
        fclose(f); err = "implausible header length in " + path; return false;
    }
    std::string hjson(hlen, '\0');
    if (!read_exact(f, hjson.data(), (size_t) hlen)) {
        fclose(f); err = "short header in " + path; return false;
    }
    fclose(f);
    const uint64_t data_base = 8 + hlen;
    json j;
    try {
        j = json::parse(hjson);
    } catch (const std::exception & e) {
        err = std::string("bad header JSON in ") + path + ": " + e.what();
        return false;
    }
    out.path = path;
    out.tensors.clear();
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string & name = it.key();
        if (name == "__metadata__") continue;
        const json & v = it.value();
        if (!v.contains("dtype") || !v.contains("shape") || !v.contains("data_offsets")) {
            err = "tensor entry without dtype/shape/data_offsets: " + name;
            return false;
        }
        st_tensor_meta m;
        m.dtype = v["dtype"].get<std::string>();
        for (const auto & d : v["shape"]) m.shape.push_back(d.get<int64_t>());
        const auto & off = v["data_offsets"];
        if (off.size() != 2) { err = "bad data_offsets for " + name; return false; }
        m.data_off_begin = data_base + off[0].get<uint64_t>();
        m.data_off_end   = data_base + off[1].get<uint64_t>();
        size_t esz = 0;
        if (!st_dtype_size(m.dtype, esz)) { err = "unknown dtype " + m.dtype + " for " + name; return false; }
        int64_t nelem = 1;
        for (int64_t d : m.shape) nelem *= d;
        if (m.data_off_end - m.data_off_begin != (uint64_t) nelem * esz) {
            err = "size mismatch for " + name; return false;
        }
        out.tensors[name] = std::move(m);
    }
    return true;
}

bool st_read_index(const std::string & dir,
                   std::map<std::string, std::string> & weight_map,
                   std::string & err) {
    const std::string path = dir + "/model.safetensors.index.json";
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fclose(f); err = "empty index " + path; return false; }
    std::string text((size_t) n, '\0');
    if (!read_exact(f, text.data(), (size_t) n)) {
        fclose(f); err = "short read of " + path; return false;
    }
    fclose(f);
    json j;
    try {
        j = json::parse(text);
    } catch (const std::exception & e) {
        err = std::string("bad index JSON in ") + path + ": " + e.what();
        return false;
    }
    if (!j.contains("weight_map")) { err = "index without weight_map: " + path; return false; }
    weight_map.clear();
    for (auto it = j["weight_map"].begin(); it != j["weight_map"].end(); ++it) {
        weight_map[it.key()] = it.value().get<std::string>();
    }
    return true;
}
