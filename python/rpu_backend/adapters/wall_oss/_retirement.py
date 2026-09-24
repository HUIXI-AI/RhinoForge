"""Graph-aware retirement for the three existing Wall-OSS component owners."""

from contextlib import nullcontext

from rpu_backend.api import _execution


_FAILED_RETIREMENTS = []


def _require_open(owner):
    _execution._require_execution_process_safe()
    parent = getattr(owner, "_retirement_owner", owner)
    if (getattr(owner, "_closed", False) or getattr(parent, "_closed", False)
            or getattr(owner, "_retirement_failed", None) is not None
            or getattr(parent, "_retirement_failed", None) is not None):
        raise RuntimeError("Wall-OSS owner is closed or failed; build a new instance")


def _retain_failed_retirement(owner, error):
    # Keep the actual object (or pending builder's handle/weight tuple), not a
    # second copy of its device state. A failed native destroy must never retry.
    if not any(value is owner for value in _FAILED_RETIREMENTS):
        _FAILED_RETIREMENTS.append(owner)
    state = getattr(owner, "__dict__", None)
    if state is not None:
        state["_retirement_failed"] = error
    session = getattr(owner, "_execution_session", None)
    if session is not None:
        session.poison()
    _execution._mark_execution_process_unsafe(f"Wall-OSS retirement failed: {error}")


def _clear_graphs(components):
    seen = set()
    for component in components:
        cache = getattr(component, "_graph_cache", None)
        if cache is not None and id(cache) not in seen:
            seen.add(id(cache))
            cache.clear()
            if not cache.cache_invariant_ok():
                raise RuntimeError("Wall-OSS GraphCache invariant failed during close")


def _retire_native(owner, destroy):
    handle = getattr(owner, "_handle", None)
    if handle is not None:
        # Older/foreign wrappers may still supply a tracked callback. Never
        # interpret an already consumed finalizer as a successful raw destroy.
        finalizer = getattr(owner, "_handle_finalizer", None)
        if finalizer is not None and not finalizer.alive:
            raise RuntimeError("Wall-OSS live handle has a consumed finalizer")
        destroy(handle)
        owner._handle = None
        if finalizer is not None:
            finalizer.detach()


def _close(owner, components):
    parent = getattr(owner, "_retirement_owner", None)
    if parent is not None and parent is not owner:
        return parent.close()
    if getattr(owner, "_retirement_failed", None) is not None:
        raise RuntimeError("Wall-OSS cleanup failed; restart the process")
    if getattr(owner, "_closed", False):
        return

    def retire():
        try:
            _execution._require_execution_process_safe()
            _clear_graphs(components)
            for component in components:
                if not getattr(component, "_closed", False):
                    retire_native = getattr(component, "_retire_native", component.close)
                    retire_native()
            for component in components:
                component._closed = True
            owner._closed = True
            from rpu_backend.api.causal_lm import _release_live_instance
            _release_live_instance(getattr(owner, "_live_owner", owner))
        except BaseException as error:
            _retain_failed_retirement(owner, error)
            raise

    session = getattr(owner, "_execution_session", None)
    try:
        if session is None:
            retire()
        else:
            session.shutdown(retire)
    except BaseException as error:
        # An active-forward rejection is safe and leaves the owner untouched.
        # An earlier process failure, however, also has to retain this owner.
        if _execution._UNSAFE_PROCESS_REASON is not None:
            _retain_failed_retirement(owner, error)
        raise
    if not getattr(owner, "_closed", False):
        error = RuntimeError("Wall-OSS closed Session still owns native resources")
        _retain_failed_retirement(owner, error)
        raise error


def _gc_close(owner):
    if (not getattr(owner, "_gc_retirement_enabled", False)
            or getattr(owner, "_closed", False)
            or getattr(owner, "_retirement_failed", None) is not None):
        return
    parent = getattr(owner, "_retirement_owner", None)
    if parent is not None and parent is not owner:
        owner = parent
    if (getattr(owner, "_closed", False)
            or getattr(owner, "_retirement_failed", None) is not None):
        return
    session = getattr(owner, "_execution_session", None)
    try:
        with session._lock if session is not None else nullcontext():
            if session is not None and session._active:
                raise RuntimeError("Wall-OSS GC during an active forward")
            from rpu_backend.api import causal_lm
            with causal_lm._LIVE_LOCK:
                live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                if live is not None and live is not getattr(owner, "_live_owner", owner):
                    raise RuntimeError("Wall-OSS GC after live-owner handoff")
                owner.close()
    except BaseException as error:
        _retain_failed_retirement(owner, error)


def _cleanup_build_failure(error, resource, cleanup):
    """Preserve the originating exception, including KeyboardInterrupt."""
    try:
        _execution._require_execution_process_safe()
        cleanup()
    except BaseException as cleanup_error:
        _retain_failed_retirement(resource, cleanup_error)
        error.add_note(f"Wall-OSS cleanup also failed: {cleanup_error}")
