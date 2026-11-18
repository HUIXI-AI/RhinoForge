"""Actual Qwen3.5 forward publication window; ATen storage, SDK/consumer doubles.

This executes the checked-in forward loop and its real flush macros. Only the
RPU device label is mapped to CPU. SDK allocation lookup, CPU/DDR coherence and
run_all_layers are host doubles: this is not an SDK or Graph lifecycle test.
"""
from pathlib import Path

import pytest


@pytest.mark.parametrize("model", ["dense", "moe"])
def test_actual_gdn_forward_publishes_sdk_allocation_tails_before_consumption(
    tmp_path, monkeypatch, model
):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    from test_siglip_attention_dry_live_contract import _function, _run_native

    root = Path(__file__).resolve().parents[1]
    filename, owner = {
        "dense": ("rpu_qwen3_5_model.cpp", "Qwen3_5Model"),
        "moe": ("rpu_qwen3_5_moe_model.cpp", "Qwen3_5MoeModel"),
    }[model]
    source = (root / "src/fused" / filename).read_text()
    forward = _function(source, f"at::Tensor {owner}::forward_impl(")
    # Include the actual consumer call, not a reconstructed publication loop.
    window = forward[forward.index("    // Fast replay skips build_gdn()"):
                     forward.index("    // PREFILL 的 resolved chunk_size")]
    window = window.replace("at::kPrivateUse1", "at::kCPU")
    profile = (root / "src/core/rpu_profile.h").read_text()
    macros = "\n".join(
        profile[profile.index(f"#define {name}("):].split("\n\n", 1)[0]
        for name in ("rpu_ddr_flush", "rpu_ddr_flush_force")
    )
    cpp = r'''
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/ScopeExit.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

bool g_rpu_ddr_flush_enabled = false;
bool g_rpu_ddr_flush_force_enabled = true;
struct Range {void* ptr; size_t bytes;};
struct Owner {
    // No Tensor owner is retained by the mock SDK. The actual input views own
    // their entire ATen backing storage throughout publication/consumption.
    void* actual_base;
    size_t actual_bytes;
    uintptr_t reported_base;
    size_t reported_bytes;
    bool missing = false;
    std::vector<uint8_t> ddr;
    void* get_cpu_ptr() const {return reinterpret_cast<void*>(reported_base);}
    size_t get_memory_size() const {return reported_bytes;}
};
std::vector<std::unique_ptr<Owner>> owners;
std::vector<Range> published;
std::vector<std::string> order;
int batches = 0, singles = 0, lookups = 0, consumers = 0;
size_t expected_live_bases = 0;
Owner& locate(void* ptr) {
    auto p = reinterpret_cast<uintptr_t>(ptr);
    for (auto& o : owners) {
        auto b = reinterpret_cast<uintptr_t>(o->actual_base);
        if (p >= b && p - b < o->actual_bytes) return *o;
    }
    throw std::runtime_error("unregistered fixture pointer");
}
size_t offset(Owner& o, void* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) -
           reinterpret_cast<uintptr_t>(o.actual_base);
}
void publish(void* ptr, size_t bytes) {
    auto& o = locate(ptr);
    const auto off = offset(o, ptr);
    assert(bytes > 0 && bytes <= o.actual_bytes - off);
    // Model SDK613 cache-line maintenance, including dirty backing tails.
    const size_t first = off & ~size_t(63);
    const size_t last = (off + bytes + 63) & ~size_t(63);
    assert(last <= o.actual_bytes);
    std::memcpy(o.ddr.data() + first,
                static_cast<uint8_t*>(o.actual_base) + first, last - first);
    published.push_back({ptr, bytes});
}
namespace rhino_lkn {
using HostDDR_t = Owner;
HostDDR_t* RpuGetHostddr(void* ptr) {
    ++lookups;
    auto& owner = locate(ptr);
    return owner.missing ? nullptr : &owner;
}
uint64_t RpuGetDevAddr(void* ptr) {
    order.push_back("live-base");
    return reinterpret_cast<uint64_t>(ptr);
}
void RpuDdrFlush(void* ptr) {
    ++singles;
    order.push_back("unsized");
    auto& owner = locate(ptr);
    // Original unsized lookup remains the oracle even when the new batch
    // admission's owner metadata is unavailable or malformed.
    publish(ptr, owner.actual_bytes - offset(owner, ptr));
}
void RpuDdrFlushBatch(void** ptrs, size_t* sizes, size_t count) {
    assert(count > 0);
    assert(std::count(order.begin(), order.end(), "live-base") ==
           static_cast<int64_t>(expected_live_bases));
    assert(consumers == 0);
    ++batches;
    order.push_back("batch");
    for (size_t i = 0; i < count; ++i) publish(ptrs[i], sizes[i]);
}
}
__MACROS__

at::Tensor state(at::IntArrayRef shape, int64_t start = 16) {
    auto backing = at::zeros({512}, at::TensorOptions().dtype(at::kHalf));
    auto owner = std::make_unique<Owner>();
    owner->actual_base = backing.data_ptr();
    owner->actual_bytes = backing.nbytes();
    owner->reported_base = reinterpret_cast<uintptr_t>(backing.data_ptr());
    owner->reported_bytes = backing.nbytes();
    owner->ddr.assign(backing.nbytes(), 0xa7);
    owners.push_back(std::move(owner));
    int64_t count = 1;
    for (auto n : shape) count *= n;
    return backing.narrow(0, start, count).view(shape);
}
int64_t version(const at::Tensor& t) {return t.is_inference() ? -1 : t._version();}
void dirty(const at::Tensor& t, uint8_t bits) {
    auto& owner = locate(t.data_ptr());
    // Raw CPU alias: no ATen operation/version bump, also dirty bytes outside
    // the Tensor's logical view. Every old unsized tail must still publish.
    std::memset(owner.actual_base, bits, owner.actual_bytes);
}
void reset_observation() {
    published.clear(); order.clear(); batches = singles = lookups = consumers = 0;
}
struct Model {
    bool fast_replay_active_ = true;
    // Initial-prefill state flush skip (build_gdn zero-fills the GDN carry).
    bool initial_prefill_state_flush_skip_ = false, action_mode_ = false;
    int64_t harness_seq_len = 1;
    // Checked decode replay and candidate head are inert in this fixture; the
    // dense window still has to compile around the real consumer call.
    bool checked_decode_replay_scope_active_ = false;
    at::Tensor decode_head_w_;
    template<class... Args> bool begin_checked_decode_replay(Args&&...) {return false;}
    void cancel_checked_decode_replay() {}
    void finish_checked_decode_replay() {}
    bool checked_decode_replay_skipped() const {return false;}
    void emit_decode_candidate_head(const at::Tensor&) {}
    int cores = 8;
    int64_t gdn_nvh_ = 2, gdn_dk_ = 8, gdn_dv_ = 8;
    int64_t gdn_kc_ = 2, gdn_conv_dim_ = 32;
    std::vector<bool> layer_is_full_{false, true, false};
    std::vector<at::Tensor> states{state({2,8,8}), {}, state({2,8,8}, 32)};
    std::vector<at::Tensor> convs{state({4,2,8}), {}, state({4,2,8}, 64)};
    std::vector<uint64_t> gdn_state_live_addr_{0,0,0}, conv_state_live_addr_{0,0,0};
    int64_t num_layers() const {return layer_is_full_.size();}
    int num_cores() const {return cores;}
    int gdn_tp() const {return cores == 8 ? 8 : 4;}
    at::Tensor* gdn_state_for_layer(int64_t i) {return &states.at(i);}
    at::Tensor* conv_state_for_layer(int64_t i) {return &convs.at(i);}
    std::vector<at::Tensor> live() const {
        std::vector<at::Tensor> out;
        for (size_t i = 0; i < states.size(); ++i)
            if (!layer_is_full_[i]) {out.push_back(states[i]); out.push_back(convs[i]);}
        return out;
    }
    template<class... Args> at::Tensor run_all_layers(Args&&...) {
        ++consumers;
        order.push_back("consumer");
        const bool must_publish = fast_replay_active_ ?
            g_rpu_ddr_flush_force_enabled : g_rpu_ddr_flush_enabled;
        for (size_t i = 0; i < states.size(); ++i) {
            if (layer_is_full_[i]) continue;
            assert(gdn_state_live_addr_[i] == reinterpret_cast<uint64_t>(states[i].data_ptr()));
            assert(conv_state_live_addr_[i] == reinterpret_cast<uint64_t>(convs[i].data_ptr()));
            if (must_publish) for (auto& t : {states[i], convs[i]}) {
                auto& o = locate(t.data_ptr()); const auto off = offset(o, t.data_ptr());
                assert(std::memcmp(o.ddr.data()+off, t.data_ptr(), o.actual_bytes-off) == 0);
            }
        }
        return {};
    }
    void forward() {
        expected_live_bases = live().size();
        at::Tensor hidden_states;
        std::vector<at::Tensor> k_caches, v_caches;
        std::optional<at::Tensor> attention_mask;
        int64_t position = 0, planned_chunk_size = 64, seq_len = harness_seq_len;
        bool is_causal = true;
        std::vector<int64_t> planned_stage_descriptor;
__WINDOW__
    }
};
void check_tails(const Model& m) {
    auto inputs = m.live();
    assert(published.size() == inputs.size());
    // Match as a multiset: an exceptional fallback may precede the batch.
    std::vector<bool> found(published.size(), false);
    for (const auto& t : inputs) {
        auto& o = locate(t.data_ptr()); const size_t bytes = o.actual_bytes-offset(o,t.data_ptr());
        bool match = false;
        for (size_t i=0; i<published.size(); ++i)
            if (!found[i] && published[i].ptr==t.data_ptr() && published[i].bytes==bytes) {
                found[i]=true;match=true;break;
            }
        assert(match && bytes > t.nbytes());
    }
    assert(consumers == 1 && order.back() == "consumer");
}
int main() {
    for (bool inference : {false,true}) {
        c10::InferenceMode mode(inference);
        for (bool fast : {false,true}) for (bool normal : {false,true})
        for (bool force : {false,true}) {
            owners.clear(); Model m; m.fast_replay_active_=fast;
            g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;
            const auto inputs=m.live();std::vector<int64_t> versions;
            for (const auto& t:inputs) versions.push_back(version(t));
            for (uint8_t bits : {uint8_t(0x31),uint8_t(0x62),uint8_t(0x31)}) {
                for (const auto& t:inputs) dirty(t,bits);
                reset_observation();m.forward();
                if (fast && force) {assert(batches==1 && singles==0 && lookups==4);check_tails(m);}
                else if (!fast && normal) {assert(batches==0 && singles==4 && lookups==0);check_tails(m);}
                else assert(batches==0 && singles==0 && lookups==0 && published.empty());
                assert(consumers==1 && order.back()=="consumer");
                for(size_t i=0;i<inputs.size();++i) assert(version(inputs[i])==versions[i]);
            }
        }
        g_rpu_ddr_flush_force_enabled=true;g_rpu_ddr_flush_enabled=false;
        // New owners on a repeated invocation must refresh mutable bases and
        // publish their current bytes; an old numerical address is no receipt.
        owners.clear();Model m;for(auto& t:m.live())dirty(t,0x22);
        reset_observation();m.forward();auto old=m.states[0];const auto oldbase=m.gdn_state_live_addr_[0];
        m.states[0]=state({2,8,8},48);dirty(m.states[0],0x74);
        reset_observation();m.forward();check_tails(m);
        assert(batches==1 && singles==0 && m.gdn_state_live_addr_[0]!=oldbase);
        assert(old.data_ptr()!=m.states[0].data_ptr());
        // The same SDK allocation may occur twice: retain both original ranges.
        m.states[2]=m.states[0];reset_observation();m.forward();check_tails(m);
        assert(batches==1 && published.size()==4);
        // Initial text prefill zero-fills the GDN carry in SPM, so its fast
        // replay publishes nothing; action and decode calls still publish.
        if (__DENSE__) {
        m.initial_prefill_state_flush_skip_=true;m.harness_seq_len=16;
        reset_observation();m.forward();
        assert(batches==0 && singles==0 && lookups==0 && published.empty());
        assert(consumers==1 && order.back()=="consumer");
        m.action_mode_=true;reset_observation();m.forward();check_tails(m);
        assert(batches==1 && singles==0);
        m.action_mode_=false;m.harness_seq_len=1;reset_observation();m.forward();check_tails(m);
        assert(batches==1 && singles==0);
        m.initial_prefill_state_flush_skip_=false;
        }
        // No GDN layers means no publication or lookup, even when force is on.
        m.layer_is_full_={true,true,true};reset_observation();m.forward();
        assert(batches==0 && singles==0 && lookups==0 && consumers==1);
    }
    // Missing/bad owner metadata and both arithmetic overflows must select
    // original unsized publication; the SDK batch cannot validate addresses.
    for(int fault=0;fault<6;++fault) {
        owners.clear();Model m;auto& o=locate(m.states[0].data_ptr());
        const uintptr_t p=reinterpret_cast<uintptr_t>(m.states[0].data_ptr());
        const uintptr_t max=std::numeric_limits<uintptr_t>::max();
        if(fault==0)o.missing=true;
        if(fault==1)o.reported_base=p+1;
        if(fault==2)o.reported_bytes=0;
        if(fault==3)o.reported_bytes=p-o.reported_base; // ptr == end
        if(fault==4){o.reported_base=p-64;o.reported_bytes=max;} // base+size overflow
        if(fault==5){o.reported_base=0;o.reported_bytes=max-31;} // +63 rounding overflow
        for(auto& t:m.live())dirty(t,0x42);
        reset_observation();m.forward();check_tails(m);
        assert(batches==1 && singles==1 && lookups==4);
    }
    // TP8 has historically admitted views. Do not silently change its public
    // guard: a noncontiguous view still publishes the original SDK tail.
    owners.clear();Model views;
    views.states[0]=views.states[0].transpose(1,2);
    assert(!views.states[0].is_contiguous());
    for(auto& t:views.live())dirty(t,0x51);
    reset_observation();views.forward();check_tails(views);
    // Reduced-core guards still validate dtype/device/layout and exact shapes
    // before any address, publication or consumer effect for the failing layer.
    for(int cores : {4,6}) {
        owners.clear();Model ok;ok.cores=cores;
        for(auto& t:ok.live())dirty(t,0x63);
        reset_observation();ok.forward();check_tails(ok);
        for(int fault=0;fault<6;++fault) {
            Model bad;bad.cores=cores;
            if(fault==0)bad.states[0]=at::zeros({2,8,8},at::kFloat);
            if(fault==1)bad.states[0]=bad.states[0].transpose(1,2);
            if(fault==2)bad.states[0]=at::empty({2,8,8},at::TensorOptions().dtype(at::kHalf).device(at::kMeta));
            if(fault==3)bad.states[0]=bad.states[0].view({4,4,8});
            if(fault==4)bad.convs[0]=bad.convs[0].view({8,2,4});
            if(fault==5)bad.states[0]=at::Tensor();
            reset_observation();bool rejected=false;
            try {bad.forward();} catch(const c10::Error&) {rejected=true;}
            assert(rejected && order.empty() && !consumers && !lookups && !batches && !singles);
        }
    }
}
'''
    _run_native(
        tmp_path, cpp.replace("__MACROS__", macros)
        .replace("__DENSE__", "true" if model == "dense" else "false")
        .replace("__WINDOW__", window),
        torch,
    )
