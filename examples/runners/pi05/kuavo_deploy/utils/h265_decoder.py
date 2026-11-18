"""Hardware-accelerated H.265 / H.264 stream decoder for Kuavo deployments on Rhino R1.

Install into a LeTools-Learning checkout by copying both this module and
``r1_video_decoder.py`` into ``kuavo_deploy/utils/`` at the repository root
(or ``src/kuavo_deploy/utils/`` in older directory layouts). Public class and
method signatures (``RealtimeH265Decoder``, ``decode``, ``is_initialized``,
``reset``) match what ``kuavo_deploy`` expects:

- Compressed Annex-B frames are decoded on the R1 V4L2 video engine
  (``hx-vpu-dec``) rather than CPU ``libavcodec``.
- Decoded planar YUV buffers are converted to ``rgb24`` through PyAV's
  ``swscale`` bridge populated with the stream's exact pixel format, color
  matrix, and color range metadata, matching PyAV's ``to_ndarray("rgb24")``
  output byte-for-byte.
- Output frames are scaled to ``(width, height)`` via ``cv2.resize``.

Whenever sequence parameter sets change, PyAV probes one picture to capture
the stream's color tags. When the R1 V4L2 device is absent or a sequence uses
an unsupported layout (non-8-bit 4:2:0, interlaced/MBAFF H.264, or dimensions
above ``max_size``), decoding transparently switches to PyAV and logs the
fallback reason.

Set ``KUAVO_VIDEO_DECODER`` to ``auto`` (default: prefer R1 hardware), ``r1``
(require R1 hardware), or ``pyav`` (force software PyAV).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import av
import cv2
import numpy as np

try:
    from .r1_video_decoder import DecoderError, R1VideoDecoder, is_sequence_parameter_set, parameter_sets
except ImportError:  # loaded as a top-level module
    from r1_video_decoder import DecoderError, R1VideoDecoder, is_sequence_parameter_set, parameter_sets

logger = logging.getLogger(__name__)

BACKEND_ENV = "KUAVO_VIDEO_DECODER"
_CODEC_NAMES = {"hevc": "hevc", "h265": "hevc", "h264": "h264"}
_TIMEOUTS_BEFORE_FALLBACK = 3
_PROBE_UNITS = 32           # access units kept to find the first picture of a new sequence
_KEPT_HEADERS = 16          # parameter sets replayed in front of a probe
_KEPT_TEMPLATES = 8
_START_CODE = b"\x00\x00\x00\x01"
# PyAV 19 removed av.error.ValueError (EINVAL maps to ArgumentError), whereas
# every PyAV codec exception inherits from ValueError.
_DECODE_ERRORS = (av.error.InvalidDataError, ValueError)


class _Template:
    """PyAV's pixel format and colour tags for the current sequence."""

    def __init__(self, frame: av.VideoFrame):
        self.format = frame.format.name
        self.colorspace = int(frame.colorspace)
        self.color_range = int(frame.color_range)
        self.size = (frame.width, frame.height)

    def to_rgb(self, i420: np.ndarray) -> np.ndarray:
        frame = av.VideoFrame.from_ndarray(i420, format=self.format)
        frame.colorspace = self.colorspace
        frame.color_range = self.color_range
        return frame.to_ndarray(format="rgb24")


class RealtimeH265Decoder:
    """Per-camera Annex-B H.265 / H.264 stream decoder producing resized RGB frames.

    Each ``decode`` call ingests one Annex-B access unit (such as a ROS
    ``CompressedImage.data`` payload) while retaining parameter-set and
    reference-picture state across inter-predicted frames.

    Args:
        width: Target output frame width in pixels.
        height: Target output frame height in pixels.
        codec: Stream codec selector, ``"hevc"`` / ``"h265"`` or ``"h264"``.
        backend: Decoder backend (``"auto"``, ``"r1"``, or ``"pyav"``); defaults
            to ``$KUAVO_VIDEO_DECODER`` or ``"auto"``.
        max_size: Maximum ``(width, height)`` stream resolution allocated on R1.
    """

    def __init__(self, width: int, height: int, *, codec: str = "hevc", backend: Optional[str] = None,
                 max_size: tuple[int, int] = (1920, 1088)):
        self.width = width
        self.height = height
        if codec.lower() not in _CODEC_NAMES:
            raise ValueError(f"codec must be one of {sorted(_CODEC_NAMES)}")
        self.codec = _CODEC_NAMES[codec.lower()]
        requested = (backend or os.environ.get(BACKEND_ENV) or "auto").lower()
        if requested not in ("auto", "r1", "pyav"):
            raise ValueError(f"{BACKEND_ENV} / backend must be auto, r1 or pyav, not {requested!r}")
        self._requested = requested
        self._max_size = tuple(max_size)
        self.fallback_reason: Optional[str] = None
        self._last_frame: Optional[np.ndarray] = None
        self._initialized = False
        self._start()

    # ---------------------------------------------------------------- setup
    def _start(self) -> None:
        self._hardware: Optional[R1VideoDecoder] = None
        self._codec_ctx: Optional[av.CodecContext] = None
        self._headers: dict[bytes, None] = {}          # parameter sets seen, oldest first
        self._sps: Optional[bytes] = None
        self._templates: dict[bytes, _Template] = {}    # by SPS: the colour tags live in its VUI
        self._template: Optional[_Template] = None
        self._probe: Optional[av.CodecContext] = None   # PyAV context fed until the sequence's first picture
        self._probe_units = 0
        self.stats = {"probe_decodes": 0, "frames_before_template": 0}
        self._last_yuv: Optional[np.ndarray] = None
        self._timeouts = 0
        if self._requested != "pyav":
            try:
                self._hardware = R1VideoDecoder(self.codec, "i420", max_size=self._max_size, timeout=0.5)
            except (DecoderError, OSError) as error:
                if self._requested == "r1":
                    raise
                self.fallback_reason = f"R1 decoder unavailable: {error}"
                logger.info("%s; decoding %s with PyAV", self.fallback_reason, self.codec)
        if self._hardware is None:
            self._codec_ctx = av.CodecContext.create(self.codec, "r")

    @property
    def backend(self) -> str:
        return "r1" if self._hardware is not None else "pyav"

    def _fall_back(self, reason: str) -> None:
        if self._requested == "r1":
            raise DecoderError(reason)
        logger.warning("%s; switching this %s stream to PyAV decoding", reason, self.codec)
        self.fallback_reason = reason
        if self._hardware is not None:
            self._hardware.close()
            self._hardware = None
        self._codec_ctx = av.CodecContext.create(self.codec, "r")
        if self._headers:   # a stream that sent its parameter sets earlier still decodes from its next IDR
            self._decode_pyav(b"".join(_START_CODE + nal for nal in self._headers))

    # --------------------------------------------------------------- decode
    def decode(self, nal_data: bytes) -> Optional[np.ndarray]:
        """Feed one Annex-B access unit and return a contiguous ``(H, W, 3)`` uint8 RGB array.

        Yields ``None`` until the initial keyframe produces a decoded picture;
        once initialized, transient packet errors reuse the most recent valid
        RGB frame.
        """
        if not nal_data:    # an empty packet would put PyAV into draining: EOFError on every later call
            return self._last_frame
        if self._hardware is not None:
            try:
                rgb = self._decode_hardware(nal_data)
            except DecoderError as error:
                self._fall_back(str(error))
            else:
                return self._finish(rgb)
        return self._finish(self._decode_pyav(nal_data))

    def _decode_pyav(self, nal_data: bytes) -> Optional[np.ndarray]:
        try:
            packet = av.Packet(nal_data)
            frames = self._codec_ctx.decode(packet)
        except _DECODE_ERRORS:
            return None
        if not frames:
            return None
        return frames[-1].to_ndarray(format="rgb24")

    def _decode_hardware(self, nal_data: bytes) -> Optional[np.ndarray]:
        for nal in parameter_sets(nal_data, self.codec):
            if is_sequence_parameter_set(nal, self.codec) and nal != self._sps:
                self._new_sequence(nal)
            self._headers.pop(nal, None)
            self._headers[nal] = None
            while len(self._headers) > _KEPT_HEADERS:
                del self._headers[next(iter(self._headers))]
        if self._probe is not None:
            self._feed_probe(nal_data)
        hardware = self._hardware
        timeouts = hardware.stats["timeouts"]
        yuv = hardware.decode(nal_data)
        if hardware.stats["timeouts"] > timeouts:
            self._timeouts += 1
            if self._timeouts >= _TIMEOUTS_BEFORE_FALLBACK:
                raise DecoderError(f"R1 decoder timed out {self._timeouts} times in a row")
        else:
            self._timeouts = 0
        if yuv is None or yuv is self._last_yuv:
            return None
        self._last_yuv = yuv
        if self._template is None:
            self.stats["frames_before_template"] += 1
            return None
        visible = (yuv.shape[1], yuv.shape[0] * 2 // 3)
        if self._template.size != visible:
            raise DecoderError(f"R1 decoder picture {visible} differs from PyAV's {self._template.size}")
        return self._template.to_rgb(yuv)

    def _new_sequence(self, sps: bytes) -> None:
        """A new SPS: find PyAV's pixel format and colour tags for its pictures."""
        self._sps, self._template = sps, self._templates.get(sps)
        self._probe, self._probe_units = None, 0
        if self._template is None:
            self._probe = av.CodecContext.create(self.codec, "r")
            if self._headers:   # parameter sets sent before this access unit
                self._feed_probe(b"".join(_START_CODE + nal for nal in self._headers), count=False)

    def _feed_probe(self, data: bytes, *, count: bool = True) -> None:
        self.stats["probe_decodes"] += int(count)
        try:
            frames = self._probe.decode(av.Packet(data))
        except _DECODE_ERRORS:
            frames = []
        if frames:
            self._template = self._templates[self._sps] = _Template(frames[0])
            while len(self._templates) > _KEPT_TEMPLATES:
                del self._templates[next(iter(self._templates))]
            self._probe = None
            return
        self._probe_units += int(count)
        if self._probe_units >= _PROBE_UNITS:
            raise DecoderError(f"PyAV decoded none of the first {_PROBE_UNITS} access units of this "
                               f"{self.codec} sequence, so its RGB conversion is unknown")

    def _finish(self, rgb: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if rgb is None:
            return self._last_frame
        if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
            rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        self._last_frame = np.ascontiguousarray(rgb)
        self._initialized = True
        return self._last_frame

    # ---------------------------------------------------------------- state
    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def reset(self) -> None:
        """Reinitialize codec and hardware contexts for a fresh stream."""
        if self._hardware is not None:
            self._hardware.close()
        self._last_frame = None
        self._initialized = False
        self.fallback_reason = None     # a fresh stream may suit the R1 decoder again
        self._start()

    def close(self) -> None:
        if self._hardware is not None:
            self._hardware.close()
            self._hardware = None
