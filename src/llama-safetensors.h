// llama-safetensors.h — minimal safetensors reader for the metal-exl3 fork.
//
// Stock llama.cpp (v1.0.0) has no safetensors reader; the fork adds one so
// EXL3 checkpoints (Hugging Face safetensors dirs) load natively, no GGUF
// conversion. Format: 8-byte LE header length + JSON header + raw tensor
// bytes. This file only READS headers and the shard index; the model loader
// owns mmap / tensor creation (see llama-model-loader).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct st_tensor_meta {
    std::string dtype;               // "F16", "I16", "F32", ...
    std::vector<int64_t> shape;      // e.g. [128, 15520, 96]
    uint64_t data_off_begin = 0;     // absolute byte offset in the shard file
    uint64_t data_off_end   = 0;
};

struct st_file {
    std::string path;
    std::map<std::string, st_tensor_meta> tensors; // name -> meta
};

// Read the header of one .safetensors shard. False + err on failure.
bool st_read_header(const std::string & path, st_file & out, std::string & err);

// Read <dir>/model.safetensors.index.json -> tensor name -> shard file name.
bool st_read_index(const std::string & dir,
                   std::map<std::string, std::string> & weight_map,
                   std::string & err);

// Element size in bytes for the dtypes we accept. False for unknown dtype.
bool st_dtype_size(const std::string & dtype, size_t & out);
