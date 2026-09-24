#pragma once

#include "fused_model_base.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace v3::detail {

struct ForwardSpmResidencyPlan {
    bool retained = false;
    int64_t original_peak_bytes = 0;
    int64_t retained_peak_bytes = 0;
};

// A valid candidate proves lifetime legality, NOT available capacity. Growing
// candidates require cold admission against the complete owner's actual FMB
// budget and a descriptor-bound selection before runtime can use them.
struct ForwardSpmResidencyCandidate {
    bool valid = false;
    int64_t original_peak_bytes = 0;
    int64_t retained_peak_bytes = 0;
    std::vector<BufferDecl> declarations;
};

// The caller supplies COMPLETE declarations (including body hooks), proves
// that selected operands have no other writers, and binds initialization to
// its graph route. Changing contents and independently leased layouts require
// their own lifecycle proof. This function never reads allocator occupancy.
inline ForwardSpmResidencyCandidate make_forward_spm_residency_candidate(
        const std::vector<BufferDecl>& declarations,
        std::initializer_list<const char*> names) {
    ForwardSpmResidencyCandidate result;
    result.original_peak_bytes = estimate_temporary_total(declarations);
    result.retained_peak_bytes = result.original_peak_bytes;
    if (names.size() == 0) return result;

    std::vector<size_t> selected;
    for (const char* name : names) {
        if (!name || !*name) return result;
        size_t found = declarations.size();
        for (size_t i = 0; i < declarations.size(); ++i) {
            const auto& decl = declarations[i];
            if (decl.alias_of && std::strcmp(decl.alias_of, name) == 0)
                return result;
            if (decl.name && std::strcmp(decl.name, name) == 0) {
                if (found != declarations.size()) return result;
                found = i;
            }
        }
        if (found == declarations.size()
                || std::find(selected.begin(), selected.end(), found)
                    != selected.end()) return result;
        const auto& decl = declarations[found];
        if (decl.storage != StorageClass::Temp || decl.per_layer != 0
                || decl.size <= 0 || decl.alias_of) return result;
        selected.push_back(found);
    }

    int first = 0;
    int last = 0;
    for (const auto& decl : declarations) {
        if (decl.size <= 0 || decl.alias_of) continue;
        if (decl.storage == StorageClass::TempPerLayer) return result;
        if (decl.storage != StorageClass::Temp) continue;
        // The allocator deliberately aliases OutsideLayerLoop with all layer
        // scopes, regardless of phase numbers. LayerWide cannot retain over it.
        if (decl.scope == BufferScope::OutsideLayerLoop
                || decl.phase_start > decl.phase_end) return result;
        first = std::min(first, decl.phase_start);
        last = std::max(last, decl.phase_end);
    }
    auto candidate = declarations;
    for (size_t i : selected) {
        candidate[i].scope = BufferScope::LayerWide;
        candidate[i].phase_start = first;
        candidate[i].phase_end = last;
    }
    result.retained_peak_bytes = estimate_temporary_total(candidate);
    result.declarations.swap(candidate);
    result.valid = true;
    return result;
}

// Existing adopters retain the no-growth contract. Persistent ownership and
// the framework's global budget remain unchanged; rejected inputs are intact.
inline ForwardSpmResidencyPlan plan_forward_spm_residency(
        std::vector<BufferDecl>& declarations,
        std::initializer_list<const char*> names) {
    auto candidate = make_forward_spm_residency_candidate(declarations, names);
    ForwardSpmResidencyPlan result{false, candidate.original_peak_bytes,
                                  candidate.retained_peak_bytes};
    if (candidate.valid &&
        candidate.retained_peak_bytes <= candidate.original_peak_bytes) {
        declarations.swap(candidate.declarations);
        result.retained = true;
    }
    return result;
}

}  // namespace v3::detail
