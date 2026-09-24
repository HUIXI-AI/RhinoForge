"""Canonical runtime error base; safe for offline filespec validators."""


class RPUBackendError(Exception):
    """Base for backend errors, publicly re-exported by runtime and api.errors."""


# Preserve the existing public exception/pickle identity after leaf extraction.
RPUBackendError.__module__ = "rpu_backend.runtime"
