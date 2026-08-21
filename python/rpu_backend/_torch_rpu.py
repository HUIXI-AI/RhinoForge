"""Configure the dynamic ``torch.rpu`` device namespace."""
from __future__ import annotations

from typing import Any


def configure_torch_rpu(
    torch_module: Any,
    rpu_module: Any,
    cpp_ext: Any,
    cpp_loaded: bool,
) -> Any:
    """Bind device protocol, runtime controls, and native convenience ops.

    Returns the runtime control module so the composition root can register
    shutdown after GraphCache cleanup.
    """
    from rpu_backend.runtime import control
    from rpu_backend.runtime import debug
    from rpu_backend.runtime import log

    # FakeTensor/Dynamo probes follow the torch.cuda-shaped device protocol.
    rpu_module.is_initialized = control.is_available
    rpu_module.current_device = lambda: 0
    rpu_module.device_count = lambda: 1 if control.is_available() else 0

    # The RNG half of that protocol (torch/utils/backend_registration.py
    # "Note(random)"). See the two functions' docstrings for why both are
    # honest here rather than merely warning-silencing.
    rpu_module._is_in_bad_fork = _is_in_bad_fork
    rpu_module.manual_seed_all = manual_seed_all

    for name in (
        "set_debug set_profile set_debug_export set_spm_debug "
        "get_debug get_profile get_debug_export get_spm_debug "
        "get_debug_tensor clear_debug_tensors list_debug_tensors "
        "spm_alloc_dump reset_profile_accumulators "
        "set_hw_perf_trace get_hw_perf_trace hw_perf_trace"
    ).split():
        setattr(rpu_module, name, getattr(debug, name))

    for name in ("set_debug_level", "get_debug_level"):
        setattr(rpu_module, name, getattr(log, name))
    rpu_module.TRACE = log.TRACE

    for name in (
        "set_caching_allocator get_caching_allocator set_ddr_flush get_ddr_flush "
        "empty_cache get_memory_stats memory_stats reset_peak_memory_stats "
        "reset_accumulated_memory_stats "
        "set_ddr_flush_force get_ddr_flush_force "
        "get_spm_mode get_allreduce_twostage get_linear_acc32 "
        "set_cross_layer_batch_prefill "
        "set_cross_layer_batch_size "
        "is_available get_amp_supported_dtype is_autocast_available shutdown "
        "reset_temporary_spm reset_all_spm spm_alloc_reset_temporary"
    ).split():
        setattr(rpu_module, name, getattr(control, name))
    rpu_module.get_rpu_warmup = control._get_rpu_warmup

    _install_printable_rpu_tensors(torch_module)

    # Most convenience ops are pybind-only; SDPA uses dispatcher registration.
    if cpp_loaded:
        for name in (
            "rms_norm apply_rotary_pos_emb "
            "insert_vcache insert_kcache "
            "insert_vcache_arena insert_kcache_arena"
        ).split():
            if hasattr(cpp_ext, name):
                setattr(rpu_module, name, getattr(cpp_ext, name))
        rpu_module.scaled_dot_product_attention_unified = (
            torch_module.ops.rpu.scaled_dot_product_attention_unified
        )

    return control


def _is_in_bad_fork() -> bool:
    """Always ``False``: there is no per-device RNG state a fork could poison.

    PyTorch calls this in exactly one place — ``torch/random.py``
    ``_seed_custom_device`` — to decide whether it is safe to push a seed into
    the device module. For CUDA the answer can be True because a fork of a
    CUDA-initialised process leaves the device context unusable; for ``rpu``
    the question does not arise, because :func:`manual_seed_all` below touches
    no device state at all.

    ⚠️ Narrow claim. This is NOT an assertion that an RPU process is safe to
    fork — the SDK's device fd and DMA buffers are not fork-safe, and nothing
    here changes that. It says only that seeding is unaffected by a fork,
    which is the single question this hook is asked.
    """
    return False


def manual_seed_all(seed: int) -> None:
    """Correct no-op: ``rpu`` has no device generator to seed.

    ``torch.Generator(device='rpu')`` raises ("Please register
    PrivateUse1HooksInterface..."), so ``torch.randn(..., device='rpu')`` is
    served by the CPU fallback and draws from the CPU **global** RNG, which
    ``torch.manual_seed`` seeds itself.

    So the seed already takes effect, and this function has nothing left to do.
    Registering it (with :func:`_is_in_bad_fork`) also prevents PyTorch from
    emitting an inapplicable warning that the seed does not take effect.

    It must stay a no-op rather than, say, re-seeding the CPU generator:
    ``_manual_seed_impl`` calls this BEFORE ``default_generator.manual_seed``,
    so doing anything here would only race the real seeding that follows.

    ⚠️ ``torch.empty(..., device='rpu')`` is genuinely seedless — it returns
    uninitialised DDR. No seed governs it, and none can.
    """
    return None


_PRINT_PATCH_FLAG = "_rpu_prints_via_cpu"


def _install_printable_rpu_tensors(torch_module: Any) -> None:
    """Make ``repr()`` of an RPU tensor copy to host before reading its data.

    Printing a tensor is not a debug luxury; PyTorch itself does it in ordinary
    control flow. ``torch/_dynamo/guards.py`` ID_MATCH builds every guard's
    description with::

        try:
            type_repr = repr(val)
        except Exception:
            type_repr = f"<{type(val).__name__}>"

    With ``val`` an RPU ``nn.Parameter`` that ``repr`` reached straight into
    device memory from the host and took the process down with SIGSEGV — which
    is not an ``Exception``, so their fallback could never catch it. Result:
    ``torch.compile`` on any RPU module died while BUILDING ITS GUARDS, before
    running a single op. ``print(param)`` at a prompt or in a debugger did the
    same thing.

    ``torch/_tensor_str.py`` already solves this for other out-of-tree backends
    -- ``_str_intern`` does ``self = self.to("cpu")`` for xla/lazy/ipu/mtia --
    but that list is an inline literal with no hook, and "rpu" is not in it. So
    patch one level down: ``_tensor_str`` builds only the NUMERIC part, and it
    runs after ``_str_intern`` has already appended the ``device='rpu:0'``
    suffix, so copying here keeps the printed form identical to every other
    device's while the formatter's slicing, ``masked_select`` and per-element
    iteration all run on host memory.

    Idempotent (re-import must not stack wrappers) and best-effort: a PyTorch
    that no longer exposes ``_tensor_str`` leaves printing exactly as it was.
    """
    try:
        tensor_str_mod = torch_module._tensor_str
        original = tensor_str_mod._tensor_str
    except AttributeError:
        return
    if getattr(original, _PRINT_PATCH_FLAG, False):
        return

    def _tensor_str_via_cpu(self, indent):
        if self.device.type == "rpu":
            self = self.detach().to("cpu")
        return original(self, indent)

    setattr(_tensor_str_via_cpu, _PRINT_PATCH_FLAG, True)
    tensor_str_mod._tensor_str = _tensor_str_via_cpu


__all__ = ["configure_torch_rpu"]
