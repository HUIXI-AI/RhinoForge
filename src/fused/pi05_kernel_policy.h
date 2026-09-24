#pragma once
#include <c10/util/Exception.h>
#include <cstdlib>
#include <cstring>

// Resolve only while constructing the owning model. The resulting immutable
// opt-in is bound to the owner's physical manifest; REPLAY never reads getenv.
inline bool pi05_kernel_opt_in(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) return false;
    TORCH_CHECK(std::strcmp(value, "1") == 0, name, " must be 0 or 1");
    return true;
}
