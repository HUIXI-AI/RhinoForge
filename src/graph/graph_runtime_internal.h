// graph_runtime_internal.h — private helpers shared by graph runtime .cpp files.
//
// These helpers are not part of the public op-facing graph API. They are kept
// out of graph_runtime.h to avoid exposing patch application internals to RPU
// operator implementations.
#pragma once

#include "graph/graph_runtime.h"

inline void apply_register_patch_to_kernel(RpuKernelGraph& graph,
                                           const RegisterPatch& p,
                                           ::rhino_lkn::Kernel_t* k) {
    uint32_t status = ::rhino_lkn::kKernelRegOk;
    switch (p.width) {
    case 1:
        status = k->set_regs(
            p.reg_idx, static_cast<uint8_t>(p.value & 0xFFULL));
        break;
    case 2:
        status = k->set_regs(
            p.reg_idx, static_cast<uint16_t>(p.value & 0xFFFFULL));
        break;
    case 4:
        status = k->set_regs(
            p.reg_idx, static_cast<uint32_t>(p.value & 0xFFFFFFFFULL));
        break;
    case 8:
#if RHINO_LAUNCH_HAS_DEV_PROGRAM_API
        status = k->set_regs(static_cast<size_t>(p.reg_idx),
                             static_cast<uint64_t>(p.value),
                             static_cast<uint32_t>(p.stride));
        break;
#else
        TORCH_CHECK(
            false,
            "64-bit raw-address RegisterPatch is unavailable with "
            "rhino-launch-kernel v1.0.0 Release");
#endif
    default:
        TORCH_CHECK(false,
                    "RegisterPatch.width must be 1/2/4/8; got ",
                    static_cast<int>(p.width),
                    " (sig=", graph.built_signature().to_string(), ")");
    }
    TORCH_CHECK(status == ::rhino_lkn::kKernelRegOk,
                "RegisterPatch register write failed: status=", status,
                " (sig=", graph.built_signature().to_string(), ")");
}
