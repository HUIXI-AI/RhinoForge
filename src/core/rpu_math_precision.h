#pragma once
#include <cstdint>

// Explicit, cold-bound arithmetic policy; formula selection remains separate.
enum class RpuUnaryPrecision : uint8_t {
    BASE = 0,
    HIGH = 1,
};
