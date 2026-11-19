#pragma once

#include <c10/util/Exception.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

namespace tts_sampling_assets {

constexpr size_t kPredictorParamsBytes = 14 * 68 * sizeof(uint16_t);
constexpr size_t kPredictorScmBytes = 14 * 256 * sizeof(uint16_t);
constexpr size_t kTalkerParamsBytes = 20 * 68 * sizeof(uint16_t);
constexpr size_t kTalkerScmBytes = 20 * 256 * sizeof(uint16_t);

struct Header {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t predictor_params_bytes;
    uint32_t predictor_scm_bytes;
    uint32_t talker_params_bytes;
    uint32_t talker_scm_bytes;
};

static_assert(sizeof(Header) == 32);

enum class TableSet { Predictor, Talker };

inline void read(const std::string& directory, TableSet set,
                 void* params, size_t params_bytes, void* scm, size_t scm_bytes) {
    constexpr char kMagic[8] = {'R', 'F', 'T', 'T', 'S', '0', '0', '1'};
    constexpr size_t kFileBytes = sizeof(Header) + kPredictorParamsBytes +
        kPredictorScmBytes + kTalkerParamsBytes + kTalkerScmBytes;
    const std::string path = directory + "/tts_sampling_tables.bin";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    TORCH_CHECK(file && file.tellg() == std::streampos(kFileBytes),
                "TTS sampling table size: ", path);
    file.seekg(0);
    Header header{};
    TORCH_CHECK(bool(file.read(reinterpret_cast<char*>(&header), sizeof(header))),
                "TTS sampling table header read: ", path);
    TORCH_CHECK(std::memcmp(header.magic, kMagic, sizeof(kMagic)) == 0 &&
                    header.version == 1 && header.header_bytes == sizeof(Header) &&
                    header.predictor_params_bytes == kPredictorParamsBytes &&
                    header.predictor_scm_bytes == kPredictorScmBytes &&
                    header.talker_params_bytes == kTalkerParamsBytes &&
                    header.talker_scm_bytes == kTalkerScmBytes,
                "TTS sampling table schema: ", path);
    const bool talker = set == TableSet::Talker;
    const size_t expected_params = talker ? kTalkerParamsBytes : kPredictorParamsBytes;
    const size_t expected_scm = talker ? kTalkerScmBytes : kPredictorScmBytes;
    TORCH_CHECK(params_bytes == expected_params && scm_bytes == expected_scm,
                "TTS sampling table destination size");
    const size_t offset = sizeof(Header) + (talker ? kPredictorParamsBytes + kPredictorScmBytes : 0);
    file.seekg(offset);
    TORCH_CHECK(bool(file.read(static_cast<char*>(params), params_bytes)) &&
                    bool(file.read(static_cast<char*>(scm), scm_bytes)),
                "TTS sampling table payload read: ", path);
}

}  // namespace tts_sampling_assets
