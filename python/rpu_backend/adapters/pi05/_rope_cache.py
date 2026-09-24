"""One installed Gemma handle's last position-indexed RoPE upload."""
import torch


class PrefillRopeUploadCache:
    def __init__(self, cos, sin, *, device="rpu"):
        self.cos = cos
        self.sin = sin
        self.device = device
        self._entry = None

    def tables(self, positions):
        # Table mutation/replacement invalidates the upload. Inference tensors
        # have no mutation counter, so those use the uncached path.
        try:
            identity = tuple((id(t), t.data_ptr(), t._version, tuple(t.shape),
                              tuple(t.stride()), t.dtype)
                             for t in (self.cos, self.sin))
        except RuntimeError:
            identity = None
        entry = self._entry
        if (identity is not None and entry is not None and entry[0] == identity
                and torch.equal(entry[1], positions)):
            return entry[2], entry[3]
        cos = self.cos[positions].to(self.device)
        sin = self.sin[positions].to(self.device)
        self._entry = ((identity, positions.clone(), cos, sin)
                       if identity is not None else None)
        return cos, sin
