#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>

namespace rpu_kernel_manifest {

inline bool valid_kernel_name(const std::string& name) {
    if (name.empty() || name.size() > 255) return false;
    const auto alpha = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
    };
    const auto alnum = [&](char c) {
        return alpha(c) || (c >= '0' && c <= '9');
    };
    if (!alpha(name.front())) return false;
    for (char c : name) {
        if (!alnum(c)) return false;
    }
    return true;
}

inline std::unordered_set<std::string> load(
    const std::filesystem::path& canonical_asset) {
    const std::filesystem::path manifest(canonical_asset.string() + ".kernels");
    std::error_code error;
    if (!std::filesystem::is_regular_file(manifest, error) || error) {
        throw std::runtime_error("kernel manifest is missing or is not a regular file: " +
                                 manifest.string());
    }
    const auto size = std::filesystem::file_size(manifest, error);
    if (error || size > 1024 * 1024) {
        throw std::runtime_error("kernel manifest is unreadable or exceeds 1 MiB: " +
                                 manifest.string());
    }

    std::ifstream input(manifest);
    std::string line;
    auto read_line = [&]() {
        if (!std::getline(input, line)) return false;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return true;
    };
    if (!read_line() || line != "rhinoforge-kernels-v1") {
        throw std::runtime_error("invalid kernel manifest header: " + manifest.string());
    }
    if (!read_line() || line.rfind("asset-size=", 0) != 0) {
        throw std::runtime_error("kernel manifest is missing asset-size: " +
                                 manifest.string());
    }
    const std::string size_text = line.substr(sizeof("asset-size=") - 1);
    std::uintmax_t declared_size = 0;
    const auto parsed = std::from_chars(
        size_text.data(), size_text.data() + size_text.size(), declared_size);
    const auto asset_size = std::filesystem::file_size(canonical_asset, error);
    if (size_text.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != size_text.data() + size_text.size() || error ||
        declared_size != asset_size) {
        throw std::runtime_error("kernel manifest asset-size mismatch: " +
                                 manifest.string());
    }

    std::unordered_set<std::string> names;
    std::string previous;
    size_t line_number = 2;
    while (read_line()) {
        ++line_number;
        if (!valid_kernel_name(line)) {
            throw std::runtime_error("invalid kernel name at line " +
                                     std::to_string(line_number) + " of " +
                                     manifest.string());
        }
        if (!previous.empty() && line <= previous) {
            throw std::runtime_error("kernel names must be unique and sorted at line " +
                                     std::to_string(line_number) + " of " +
                                     manifest.string());
        }
        previous = line;
        names.insert(line);
    }
    if (!input.eof() || names.empty()) {
        throw std::runtime_error("kernel manifest is empty or unreadable: " +
                                 manifest.string());
    }
    return names;
}

}  // namespace rpu_kernel_manifest
