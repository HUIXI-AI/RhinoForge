"""Hardware H.264 / H.265 decoding on the Rhino R1 (``hx-vpu-dec``).

The R1 video decoder is a V4L2 memory-to-memory device (``hx-vpu-dec0``). It
takes Annex-B access units and writes 8-bit 4:2:0 frames as NV12; the CPU only
submits the bitstream and copies the visible picture out. The decoder has no
colour conversion (asked for RGB it still writes NV12), so RGB is the caller's
step: ``h265_decoder.RealtimeH265Decoder`` next to this file converts with the
same libswscale call PyAV uses, which keeps its RGB byte-identical to PyAV.

Only numpy and the standard library are needed (ioctls through ``fcntl``,
buffers through ``mmap``). The calling user needs read/write access to the
device node (group ``video`` on the R1 image) and to
``/dev/dma_heap/hx_mem_high_reserve`` (open to all users).

    decoder = R1VideoDecoder("h265", "i420", max_size=(848, 480))   # or "h264", "nv12"
    yuv = decoder.decode(access_unit)         # (H * 3 // 2, W) uint8, or None
    decoder.close()

One instance decodes one stream; open one per camera. Each ``decode`` call
takes one access unit (one picture, with VPS/SPS/PPS in front of IDR
pictures) and returns that picture; streams without B-frames (the Kuavo
cameras send IDR + P) come back in the same call.

Driver contract, as used by the vendor ``libv4l2codec`` (traced on the board)
and measured on the R1 with the hx_codec driver:

- ``struct v4l2_format`` carries the format union at byte 4, not 8;
- the frame format's ``reserved[0]`` byte is the row alignment (the vendor
  library sends 64); left at 0 the driver sizes every plane 0 and no picture
  ever finishes;
- bitstream and frame buffers are DMABUFs from the ``hx_mem_high_reserve``
  DMA heap; the driver keeps them across the stream's resolution event.
  Never use V4L2_MEMORY_MMAP on this device (an MMAP probe rebooted a
  board). The heap maps buffers uncached:
  reading a frame out costs about 1.15 us per KiB on the R1 CPU (0.7 ms for
  848x480) and is the main CPU cost of this path, so each frame byte is read
  once, and the cache maintenance around it scales with the buffer size, so
  frame buffers are sized to the stream (allocated at its first SPS);
- a frame is one NV12 plane laid out from the stream's coded size (the
  stream-header control, equal to the SPS coded size: H.264 rounds up to
  16-pixel macroblocks, so 1080p is coded 1088 rows): rows are
  ``align64(coded width)`` bytes apart and chroma starts after ``coded
  height`` luma rows. The visible picture is the SPS cropping window. The
  driver's ``bytesperline`` / ``sizeimage`` do not describe this: they echo the
  requested size, and for H.264 the driver sizes buffers without the 64-byte
  row alignment the hardware writes with, so a frame wider than an unaligned
  request overruns the buffer. This module therefore allocates
  ``align64(coded width) x align64(coded height)`` frames, computes the
  layout itself and refuses, before submitting it, any sequence that exceeds
  ``max_size``, is not 8-bit 4:2:0 or (H.264) is not frame-coded. The
  hardware writes nothing past the NV12 picture (checked on every test
  stream with oversized buffers);
- a finished picture wakes ``poll`` through the returned bitstream buffer
  (POLLOUT), not through POLLIN, so waits include POLLOUT;
- the one decoder (``hx-vpu-dec0``) serves every open instance: pictures
  submitted together are decoded one after another (an 848x480 H.265 IDR
  takes ~9 ms; three cameras whose IDRs coincide wait up to ~17 ms), and
  frame buffers come from CMA at the first SPS (~7 ms alone, ~50 ms when
  three cameras start at once).

Run ``python r1_video_decoder.py --help`` for the stream check / benchmark.
"""
from __future__ import annotations

import ctypes
import errno
import fcntl
import mmap
import os
import select
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["R1VideoDecoder", "DecoderError", "SequenceInfo", "find_decoder_device", "is_sequence_parameter_set",
           "parameter_sets", "parse_sequence_parameter_set", "split_access_units", "yuv_planes"]

# --------------------------------------------------------------------------
# V4L2 / DMA-heap ABI (64-bit Linux). Sizes are asserted below.
# --------------------------------------------------------------------------
_u8, _u16, _u32, _s32, _u64 = ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint64


class _Capability(ctypes.Structure):
    _fields_ = [("driver", _u8 * 16), ("card", _u8 * 32), ("bus_info", _u8 * 32), ("version", _u32),
                ("capabilities", _u32), ("device_caps", _u32), ("reserved", _u32 * 3)]


class _FmtDesc(ctypes.Structure):
    _fields_ = [("index", _u32), ("type", _u32), ("flags", _u32), ("description", _u8 * 32),
                ("pixelformat", _u32), ("mbus_code", _u32), ("reserved", _u32 * 3)]


_FORMAT_SIZE = 208      # sizeof(struct v4l2_format); part of the ioctl number


class _FormatBuffer:
    """``struct v4l2_format`` whose multi-planar format starts at byte ``base``.

    The header places the union at byte 8; the hx-vpu driver reads and writes
    it at byte 4 (the vendor library does the same). ``R1VideoDecoder`` finds
    the offset from the driver's own G_FMT reply.
    """

    def __init__(self, kind: int, base: int):
        self.raw = bytearray(_FORMAT_SIZE)
        self.base = base
        struct.pack_into("<I", self.raw, 0, kind)

    def _get(self, offset: int) -> int:
        return struct.unpack_from("<I", self.raw, self.base + offset)[0]

    def _set(self, offset: int, value: int) -> None:
        struct.pack_into("<I", self.raw, self.base + offset, value)

    width = property(lambda self: self._get(0), lambda self, v: self._set(0, v))
    height = property(lambda self: self._get(4), lambda self, v: self._set(4, v))
    pixelformat = property(lambda self: self._get(8), lambda self, v: self._set(8, v))
    num_planes = property(lambda self: self.raw[self.base + 180],
                          lambda self, v: self.raw.__setitem__(self.base + 180, v))
    # pix_mp.reserved[0]: hx-vpu reads the frame row alignment from it.
    alignment = property(lambda self: self.raw[self.base + 185],
                         lambda self, v: self.raw.__setitem__(self.base + 185, v))

    def sizeimage(self, plane: int = 0) -> int:
        return self._get(20 + 20 * plane)

    def set_sizeimage(self, plane: int, value: int) -> None:
        self._set(20 + 20 * plane, value)

    def bytesperline(self, plane: int = 0) -> int:
        return self._get(24 + 20 * plane)


class _RequestBuffers(ctypes.Structure):
    _fields_ = [("count", _u32), ("type", _u32), ("memory", _u32), ("capabilities", _u32), ("reserved", _u32)]


class _PlaneM(ctypes.Union):
    _fields_ = [("mem_offset", _u32), ("userptr", ctypes.c_ulong), ("fd", _s32)]


class _Plane(ctypes.Structure):
    _fields_ = [("bytesused", _u32), ("length", _u32), ("m", _PlaneM), ("data_offset", _u32), ("reserved", _u32 * 11)]


class _Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class _Timecode(ctypes.Structure):
    _fields_ = [("type", _u32), ("flags", _u32), ("frames", _u8), ("seconds", _u8), ("minutes", _u8),
                ("hours", _u8), ("userbits", _u8 * 4)]


class _BufferM(ctypes.Union):
    _fields_ = [("offset", _u32), ("userptr", ctypes.c_ulong), ("planes", ctypes.POINTER(_Plane)), ("fd", _s32)]


class _Buffer(ctypes.Structure):
    _fields_ = [("index", _u32), ("type", _u32), ("bytesused", _u32), ("flags", _u32), ("field", _u32),
                ("timestamp", _Timeval), ("timecode", _Timecode), ("sequence", _u32), ("memory", _u32),
                ("m", _BufferM), ("length", _u32), ("reserved2", _u32), ("request_fd", _s32)]


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class _Event(ctypes.Structure):
    _fields_ = [("type", _u32), ("u", ctypes.c_uint64 * 8), ("pending", _u32), ("sequence", _u32),
                ("timestamp", _Timespec), ("id", _u32), ("reserved", _u32 * 8)]


class _EventSubscription(ctypes.Structure):
    _fields_ = [("type", _u32), ("id", _u32), ("flags", _u32), ("reserved", _u32 * 5)]


class _Control(ctypes.Structure):
    _fields_ = [("id", _u32), ("value", _s32)]


class _ExtControl(ctypes.Structure):               # struct v4l2_ext_control (packed)
    _pack_ = 1
    _fields_ = [("id", _u32), ("size", _u32), ("reserved2", _u32), ("ptr", ctypes.c_void_p)]


class _ExtControls(ctypes.Structure):
    _fields_ = [("which", _u32), ("count", _u32), ("error_idx", _u32), ("request_fd", _s32), ("reserved", _u32),
                ("controls", ctypes.POINTER(_ExtControl))]


class _HeapAllocation(ctypes.Structure):          # struct dma_heap_allocation_data
    _fields_ = [("len", _u64), ("fd", _u32), ("fd_flags", _u32), ("heap_flags", _u64)]


_ABI_SIZES = {_Capability: 104, _FmtDesc: 64, _RequestBuffers: 20, _Plane: 64, _Buffer: 88, _Event: 136,
              _EventSubscription: 32, _Control: 8, _ExtControl: 20, _ExtControls: 32, _HeapAllocation: 24}


def _ioc(direction: int, kind: str, number: int, size) -> int:
    size = size if isinstance(size, int) else ctypes.sizeof(size)
    return (direction << 30) | (size << 16) | (ord(kind) << 8) | number


_R, _W, _RW = 2, 1, 3
VIDIOC_QUERYCAP = _ioc(_R, "V", 0, _Capability)
VIDIOC_ENUM_FMT = _ioc(_RW, "V", 2, _FmtDesc)
VIDIOC_G_FMT = _ioc(_RW, "V", 4, _FORMAT_SIZE)
VIDIOC_S_FMT = _ioc(_RW, "V", 5, _FORMAT_SIZE)
VIDIOC_REQBUFS = _ioc(_RW, "V", 8, _RequestBuffers)
VIDIOC_QBUF = _ioc(_RW, "V", 15, _Buffer)
VIDIOC_DQBUF = _ioc(_RW, "V", 17, _Buffer)
VIDIOC_STREAMON = _ioc(_W, "V", 18, 4)
VIDIOC_STREAMOFF = _ioc(_W, "V", 19, 4)
VIDIOC_S_CTRL = _ioc(_RW, "V", 28, _Control)
VIDIOC_G_EXT_CTRLS = _ioc(_RW, "V", 71, _ExtControls)
VIDIOC_DQEVENT = _ioc(_R, "V", 89, _Event)
VIDIOC_SUBSCRIBE_EVENT = _ioc(_W, "V", 90, _EventSubscription)
DMA_HEAP_IOCTL_ALLOC = _ioc(_RW, "H", 0, _HeapAllocation)
DMA_BUF_IOCTL_SYNC = _ioc(_W, "b", 0, 8)

_OUTPUT, _CAPTURE, _DMABUF = 10, 9, 4         # V4L2_BUF_TYPE_VIDEO_{OUTPUT,CAPTURE}_MPLANE, V4L2_MEMORY_DMABUF
_CAP_M2M_MPLANE = 0x00004000
_BUF_FLAG_ERROR = 0x40
_EVENT_EOS, _EVENT_SOURCE_CHANGE = 2, 5
_CID_HX_DISABLE_REORDER = 0x00981CE9         # hx-vpu "v4l2_cid_dis_reorder"
_CID_HX_STREAM_HEADER = 0x00981CE8           # hx-vpu "v4l2_cid_vdec_header": width, height, min bitstream / frame buffers
_SYNC_READ, _SYNC_WRITE, _SYNC_END = 1, 2, 4
_HEAP = "/dev/dma_heap/hx_mem_high_reserve"
_HEAP_FLAGS = 1                              # what libv4l2codec passes
_ALIGN = 64                                  # frame row alignment: requested from the driver and used by the hardware
_POLL_SLICE_S = 0.01                         # re-check the queues at least this often while waiting
_INFLIGHT_LIMIT = 64                         # submitted pictures whose geometry is remembered


def _fourcc(text: str) -> int:
    return int.from_bytes(text.encode("ascii"), "little")


def _fourcc_text(value: int) -> str:
    return value.to_bytes(4, "little").decode("ascii", "replace")


def _align(value: int, alignment: int = _ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


# hx-vpu lists frame-based Annex-B H.264 under the 'S264' code.
_CODEC_FOURCCS = {"h264": ("H264", "S264"), "h265": ("HEVC",), "hevc": ("HEVC",)}
_FRAME_FOURCC = "NV12"
OUTPUTS = ("nv12", "i420")


class DecoderError(RuntimeError):
    """The hardware decoder refused a request, cannot take the stream, or reported a terminal error."""


def find_decoder_device(name_prefix: str = "hx-vpu-dec") -> str:
    """Path of the first V4L2 node whose name starts with ``name_prefix``."""
    for entry in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            name = (entry / "name").read_text().strip()
        except OSError:
            continue
        if name.startswith(name_prefix) and Path("/dev", entry.name).exists():
            return str(Path("/dev", entry.name))
    raise DecoderError(f"no V4L2 device named {name_prefix}* (is this a Rhino R1 with the hx-vpu driver loaded?)")


def _check_abi() -> None:
    if ctypes.sizeof(ctypes.c_long) != 8:
        raise DecoderError("r1_video_decoder supports 64-bit Linux only")
    for kind, size in _ABI_SIZES.items():
        if ctypes.sizeof(kind) != size:
            raise DecoderError(f"V4L2 ABI mismatch: sizeof({kind.__name__}) = {ctypes.sizeof(kind)}, expected {size}")


# --------------------------------------------------------------------------
# Elementary-stream syntax: NAL units, access units, sequence parameter sets
# --------------------------------------------------------------------------
def _canonical_codec(codec: str) -> str:
    codec = codec.lower()
    if codec not in _CODEC_FOURCCS:
        raise ValueError(f"codec must be one of {sorted(_CODEC_FOURCCS)}")
    return "h265" if codec == "hevc" else codec


def _nal_units(data: bytes):
    """(start, payload_start, end) of each Annex-B NAL unit."""
    starts = []
    i, n = 0, len(data)
    while True:
        i = data.find(b"\x00\x00\x01", i)
        if i < 0:
            break
        begin = i - 1 if i > 0 and data[i - 1] == 0 else i
        starts.append((begin, i + 3))
        i += 3
    for k, (begin, payload) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else n
        yield begin, payload, end


def _nal_type(data: bytes, payload: int, codec: str) -> int:
    return data[payload] & 0x1F if codec == "h264" else (data[payload] >> 1) & 0x3F


_SPS_TYPE = {"h264": 7, "h265": 33}
_PARAMETER_SET_TYPES = {"h264": (7, 8), "h265": (32, 33, 34)}


def is_sequence_parameter_set(nal: bytes, codec: str) -> bool:
    """True if ``nal`` (a NAL unit without start code) is an SPS."""
    codec = _canonical_codec(codec)
    return bool(nal) and _nal_type(nal, 0, codec) == _SPS_TYPE[codec]


def parameter_sets(data: bytes, codec: str) -> list[bytes]:
    """VPS / SPS / PPS NAL units (without start codes) in one access unit, in stream order."""
    codec = _canonical_codec(codec)
    kinds = _PARAMETER_SET_TYPES[codec]
    found = []
    for _, payload, end in _nal_units(data):
        if payload < end and _nal_type(data, payload, codec) in kinds:
            found.append(data[payload:end].rstrip(b"\x00"))
    return found


def split_access_units(data: bytes, codec: str) -> list[bytes]:
    """Split an Annex-B elementary stream into access units (one picture each)."""
    codec = _canonical_codec(codec)
    units, current, has_picture = [], [], False
    for begin, payload, end in _nal_units(data):
        if payload >= end:
            continue
        kind = _nal_type(data, payload, codec)
        if codec == "h264":
            is_slice = kind in (1, 5)
            first_slice = is_slice and payload + 1 < end and data[payload + 1] & 0x80  # first_mb_in_slice == 0
            starts_unit = kind in (6, 7, 8, 9)
        else:
            is_slice = kind < 32
            first_slice = is_slice and payload + 2 < end and data[payload + 2] & 0x80  # first_slice_segment_in_pic
            starts_unit = kind in (32, 33, 34, 35, 39)
        if has_picture and (starts_unit or first_slice):
            units.append(b"".join(current))
            current, has_picture = [], False
        current.append(data[begin:end])
        has_picture = has_picture or bool(is_slice)
    if current and has_picture:
        units.append(b"".join(current))
    return units


class _Bits:
    """Exp-Golomb reader over an RBSP (emulation-prevention bytes removed)."""

    def __init__(self, nal: bytes, skip_bytes: int):
        rbsp, zeros = bytearray(), 0
        for byte in nal[skip_bytes:]:
            if zeros >= 2 and byte == 3:
                zeros = 0
                continue
            rbsp.append(byte)
            zeros = zeros + 1 if byte == 0 else 0
        self.value = int.from_bytes(rbsp, "big") if rbsp else 0
        self.size, self.pos = len(rbsp) * 8, 0

    def u(self, count: int) -> int:
        if count == 0:
            return 0
        if self.pos + count > self.size:
            raise ValueError("sequence parameter set is truncated")
        self.pos += count
        return (self.value >> (self.size - self.pos)) & ((1 << count) - 1)

    def ue(self) -> int:
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
            if zeros > 31:
                raise ValueError("malformed Exp-Golomb code in the sequence parameter set")
        return (1 << zeros) - 1 + self.u(zeros)

    def se(self) -> int:
        code = self.ue()
        return (code + 1) // 2 if code & 1 else -(code // 2)


@dataclass(frozen=True)
class SequenceInfo:
    """What a sequence parameter set says about the decoded pictures."""

    codec: str
    sps_id: int
    profile_idc: int
    chroma_format_idc: int          # 1 = 4:2:0
    bit_depth_luma: int
    bit_depth_chroma: int
    frame_coded: bool               # H.264 frame_mbs_only_flag (no field / MBAFF pictures); always True for H.265
    coded_width: int
    coded_height: int
    crop: tuple[int, int, int, int]  # visible picture: x, y, width, height inside the coded picture

    @property
    def width(self) -> int:
        return self.crop[2]

    @property
    def height(self) -> int:
        return self.crop[3]

    def unsupported_reason(self) -> str | None:
        """Why the R1 path cannot reproduce this stream's pictures exactly (None if it can)."""
        if self.chroma_format_idc != 1:
            return f"chroma_format_idc {self.chroma_format_idc} (only 4:2:0 is supported)"
        if (self.bit_depth_luma, self.bit_depth_chroma) != (8, 8):
            return f"bit depth {self.bit_depth_luma}/{self.bit_depth_chroma} (only 8-bit is supported)"
        if not self.frame_coded:
            return "field or MBAFF coded H.264 (frame_mbs_only_flag = 0) is not supported"
        x, y, width, height = self.crop
        if width <= 0 or height <= 0 or x + width > self.coded_width or y + height > self.coded_height:
            return f"cropping window {self.crop} lies outside the {self.coded_width}x{self.coded_height} coded picture"
        return None


def _crop(coded_width: int, coded_height: int, unit_x: int, unit_y: int, left: int, right: int, top: int,
          bottom: int) -> tuple[int, int, int, int]:
    return (unit_x * left, unit_y * top, coded_width - unit_x * (left + right), coded_height - unit_y * (top + bottom))


def _skip_scaling_list(bits: _Bits, size: int) -> None:
    last = following = 8
    for _ in range(size):
        if following != 0:
            following = (last + bits.se() + 256) % 256
        last = last if following == 0 else following


_H264_HIGH_PROFILES = (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135)


def _parse_h264_sps(nal: bytes) -> SequenceInfo:
    bits = _Bits(nal, 1)
    profile = bits.u(8)
    bits.u(16)                                         # constraint flags, level_idc
    sps_id = bits.ue()
    chroma, separate_planes, depth_luma, depth_chroma = 1, 0, 8, 8
    if profile in _H264_HIGH_PROFILES:
        chroma = bits.ue()
        if chroma == 3:
            separate_planes = bits.u(1)
        depth_luma, depth_chroma = bits.ue() + 8, bits.ue() + 8
        bits.u(1)                                      # qpprime_y_zero_transform_bypass_flag
        if bits.u(1):                                  # seq_scaling_matrix_present_flag
            for index in range(12 if chroma == 3 else 8):
                if bits.u(1):
                    _skip_scaling_list(bits, 16 if index < 6 else 64)
    bits.ue()                                          # log2_max_frame_num_minus4
    poc_type = bits.ue()
    if poc_type == 0:
        bits.ue()                                      # log2_max_pic_order_cnt_lsb_minus4
    elif poc_type == 1:
        bits.u(1)
        bits.se()
        bits.se()
        cycle = bits.ue()
        if cycle > 255:
            raise ValueError("num_ref_frames_in_pic_order_cnt_cycle > 255")
        for _ in range(cycle):
            bits.se()
    bits.ue()                                          # max_num_ref_frames
    bits.u(1)                                          # gaps_in_frame_num_value_allowed_flag
    width_mbs, height_units = bits.ue() + 1, bits.ue() + 1
    frame_mbs_only = bits.u(1)
    if not frame_mbs_only:
        bits.u(1)                                      # mb_adaptive_frame_field_flag
    bits.u(1)                                          # direct_8x8_inference_flag
    offsets = (bits.ue(), bits.ue(), bits.ue(), bits.ue()) if bits.u(1) else (0, 0, 0, 0)
    coded_width, coded_height = width_mbs * 16, (2 - frame_mbs_only) * height_units * 16
    array_type = 0 if separate_planes else chroma
    if array_type == 0:
        unit_x, unit_y = 1, 2 - frame_mbs_only
    else:
        unit_x = 1 if array_type == 3 else 2
        unit_y = (2 if array_type == 1 else 1) * (2 - frame_mbs_only)
    return SequenceInfo("h264", sps_id, profile, chroma, depth_luma, depth_chroma, bool(frame_mbs_only),
                        coded_width, coded_height, _crop(coded_width, coded_height, unit_x, unit_y, *offsets))


def _parse_h265_sps(nal: bytes) -> SequenceInfo:
    bits = _Bits(nal, 2)
    bits.u(4)                                          # sps_video_parameter_set_id
    sub_layers = bits.u(3)                             # sps_max_sub_layers_minus1
    bits.u(1)                                          # sps_temporal_id_nesting_flag
    bits.u(3)                                          # general_profile_space, general_tier_flag
    profile = bits.u(5)
    bits.u(32)                                         # general_profile_compatibility_flags
    bits.u(48)                                         # source / constraint flags
    bits.u(8)                                          # general_level_idc
    present = [(bits.u(1), bits.u(1)) for _ in range(sub_layers)]
    if sub_layers:
        bits.u(2 * (8 - sub_layers))                   # reserved_zero_2bits
    for profile_present, level_present in present:
        bits.u(88 if profile_present else 0)
        bits.u(8 if level_present else 0)
    sps_id = bits.ue()
    chroma = bits.ue()
    separate_planes = bits.u(1) if chroma == 3 else 0
    coded_width, coded_height = bits.ue(), bits.ue()
    offsets = (bits.ue(), bits.ue(), bits.ue(), bits.ue()) if bits.u(1) else (0, 0, 0, 0)
    depth_luma, depth_chroma = bits.ue() + 8, bits.ue() + 8
    array_type = 0 if separate_planes else chroma
    unit_x = 2 if array_type in (1, 2) else 1
    unit_y = 2 if array_type == 1 else 1
    return SequenceInfo("h265", sps_id, profile, chroma, depth_luma, depth_chroma, True,
                        coded_width, coded_height, _crop(coded_width, coded_height, unit_x, unit_y, *offsets))


def parse_sequence_parameter_set(nal: bytes, codec: str) -> SequenceInfo:
    """Parse an SPS NAL unit (starting at its NAL header, no start code)."""
    codec = _canonical_codec(codec)
    if not is_sequence_parameter_set(nal, codec):
        raise ValueError(f"not a {codec} sequence parameter set")
    return _parse_h264_sps(nal) if codec == "h264" else _parse_h265_sps(nal)


# --------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------
class _DmaBuffer:
    """One DMA-heap buffer, mapped for CPU access."""

    def __init__(self, heap_fd: int, size: int):
        request = _HeapAllocation(len=size, fd_flags=os.O_RDWR | os.O_CLOEXEC, heap_flags=_HEAP_FLAGS)
        fcntl.ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, request, True)
        self.fd, self.size = int(request.fd), int(size)
        try:
            self.map = mmap.mmap(self.fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        except BaseException:
            os.close(self.fd)
            raise

    def sync(self, flags: int) -> None:
        fcntl.ioctl(self.fd, DMA_BUF_IOCTL_SYNC, struct.pack("<Q", flags))

    def close(self) -> None:
        try:
            self.map.close()
        except BufferError:     # a caller still holds a view; the mapping goes when that view does
            pass
        os.close(self.fd)


class R1VideoDecoder:
    """One H.264 / H.265 stream decoded by the R1 hardware decoder.

    Args:
        codec: ``"h265"`` (alias ``"hevc"``) or ``"h264"``.
        output: ``"i420"`` (planar Y, U, V: the ``yuv420p`` layout PyAV and
            OpenCV ``COLOR_YUV2RGB_I420`` take) or ``"nv12"`` (Y, then
            interleaved UV: OpenCV ``COLOR_YUV2RGB_NV12``). Both are returned
            as one (H * 3 // 2, W) uint8 array of the visible picture.
        max_size: largest (width, height) the stream may have; a sequence
            whose coded size exceeds it is refused before it is submitted.
            Frame buffers are allocated when the first sequence parameter set
            arrives, at its coded size rounded up to 64 (the buffers are
            mapped uncached and their cache maintenance costs per byte, so
            they are kept to the stream's size); a later, larger sequence
            restarts the decoder at its first access unit.
        device: V4L2 node; default: the first ``hx-vpu-dec*`` node.
        low_delay: turn off the display-reorder queue. Only for streams
            without B-frames (decode order is then display order); the
            default keeps display order, which IDR + P streams already leave
            in the same call.
        timeout: seconds ``decode`` waits for the submitted picture.
        input_buffers / input_buffer_bytes: bitstream buffers (one access unit
            each; a larger access unit is refused).
        capture_buffers: frame buffers (the decoder asks for up to 12 for
            H.265 reference pictures; the vendor library uses 14).
        heap: DMA heap the buffers come from.

    ``decode`` raises ``DecoderError`` (without submitting anything) when a
    sequence parameter set describes pictures this path cannot reproduce: not
    8-bit 4:2:0, field / MBAFF H.264, or larger than ``max_size``. The decoder
    stays usable for a following supported sequence; callers that must show
    every stream fall back to software decoding.
    """

    def __init__(self, codec: str = "h265", output: str = "i420", *, max_size: tuple[int, int] = (1920, 1088),
                 device: str | None = None, low_delay: bool = False, timeout: float = 1.0,
                 input_buffers: int = 4, input_buffer_bytes: int = 2 << 20, capture_buffers: int = 14,
                 heap: str = _HEAP):
        _check_abi()
        codec = _canonical_codec(codec)
        if output not in OUTPUTS:
            raise ValueError(f"output must be one of {OUTPUTS} (the R1 decoder has no RGB output)")
        if not 1 <= input_buffers <= 16 or not 2 <= capture_buffers <= 32:
            raise ValueError("input_buffers must be in 1..16 and capture_buffers in 2..32")
        width, height = (int(v) for v in max_size)
        if not (16 <= width <= 8192 and 16 <= height <= 8192):
            raise ValueError("max_size must be within 16..8192 per side")
        self.codec, self.output, self.low_delay, self.timeout = codec, output, bool(low_delay), float(timeout)
        self._options = dict(max_size=(width, height), input_buffers=int(input_buffers),
                             input_buffer_bytes=int(input_buffer_bytes), capture_buffers=int(capture_buffers),
                             heap=heap)
        self.device = device or find_decoder_device()
        self.stats = {"submitted": 0, "frames": 0, "superseded": 0, "timeouts": 0, "decode_errors": 0,
                      "source_changes": 0, "skipped_without_headers": 0, "restarts": 0}
        self._fd = -1
        self._last: np.ndarray | None = None
        self._open()

    # ----------------------------------------------------------------- setup
    def _ioctl(self, request: int, arg, *, allow=()) -> bool:
        try:
            fcntl.ioctl(self._fd, request, arg, True)
            return True
        except OSError as error:
            if error.errno in allow:
                return False
            raise DecoderError(f"{self.device}: ioctl 0x{request:08x} failed: {os.strerror(error.errno)}") from error

    def _open(self) -> None:
        """Open the device and check it; buffers wait for the stream's first SPS (``_start``)."""
        self._inputs: list[_DmaBuffer] = []
        self._frames: list[_DmaBuffer] = []
        self._free_inputs: list[int] = []
        self._streaming = False
        self._allocated: dict | None = None
        self._layout: dict | None = None
        self._sequence = 0
        self._sps: SequenceInfo | None = None
        self._inflight: dict[int, SequenceInfo] = {}
        try:
            self._fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
        except OSError as error:
            raise DecoderError(f"cannot open {self.device}: {error.strerror}") from error
        try:
            cap = _Capability()
            self._ioctl(VIDIOC_QUERYCAP, cap)
            if not (cap.device_caps or cap.capabilities) & _CAP_M2M_MPLANE:
                raise DecoderError(f"{self.device} is not a multi-planar memory-to-memory device")
            self._input_fourcc = self._pick_input_format()
            self._format_base = self._detect_format_base()
        except BaseException:
            self._close_fd()
            raise

    def _start(self, width: int, height: int) -> None:
        """Set the formats, allocate ``width`` x ``height`` (64-aligned) frame buffers and start streaming."""
        heap_fd = -1
        try:
            stream = _FormatBuffer(_OUTPUT, self._format_base)
            stream.width, stream.height = width, height
            stream.pixelformat = self._input_fourcc
            stream.num_planes = 1
            stream.set_sizeimage(0, self._options["input_buffer_bytes"])
            self._ioctl(VIDIOC_S_FMT, stream.raw)
            if stream.pixelformat != self._input_fourcc:
                raise DecoderError(f"{self.device} did not accept {_fourcc_text(self._input_fourcc)} bitstreams")
            input_bytes = max(int(stream.sizeimage(0)), self._options["input_buffer_bytes"])

            frame = _FormatBuffer(_CAPTURE, self._format_base)
            frame.width, frame.height = width, height
            frame.pixelformat = _fourcc(_FRAME_FOURCC)
            frame.num_planes = 1
            frame.alignment = _ALIGN
            self._ioctl(VIDIOC_S_FMT, frame.raw)
            if _fourcc_text(frame.pixelformat) != _FRAME_FOURCC or frame.num_planes != 1:
                raise DecoderError(f"{self.device} refused single-plane {_FRAME_FOURCC} frames "
                                   f"(offered {_fourcc_text(frame.pixelformat)}, {frame.num_planes} planes)")
            # The hardware's own layout, not the driver's sizeimage, decides how much it writes.
            frame_bytes = max(int(frame.sizeimage(0)), width * height * 3 // 2)
            self._allocated = {"width": width, "height": height, "frame_bytes": frame_bytes,
                               "driver_bytesperline": int(frame.bytesperline(0)),
                               "driver_sizeimage": int(frame.sizeimage(0))}

            inputs = _RequestBuffers(count=self._options["input_buffers"], type=_OUTPUT, memory=_DMABUF)
            self._ioctl(VIDIOC_REQBUFS, inputs)
            frames = _RequestBuffers(count=self._options["capture_buffers"], type=_CAPTURE, memory=_DMABUF)
            self._ioctl(VIDIOC_REQBUFS, frames)
            if inputs.count < 1 or frames.count < 2:
                raise DecoderError(f"{self.device} granted {inputs.count} bitstream / {frames.count} frame buffers")
            heap_fd = os.open(self._options["heap"], os.O_RDWR | os.O_CLOEXEC)
            self._inputs = [_DmaBuffer(heap_fd, input_bytes) for _ in range(inputs.count)]
            self._frames = [_DmaBuffer(heap_fd, frame_bytes) for _ in range(frames.count)]
            self._free_inputs = list(range(len(self._inputs)))
            self._input_capacity = input_bytes

            self._ioctl(VIDIOC_S_CTRL, _Control(id=_CID_HX_DISABLE_REORDER, value=int(self.low_delay)),
                        allow=(errno.EINVAL, errno.ENOTTY))
            for event in (_EVENT_SOURCE_CHANGE, _EVENT_EOS):
                self._ioctl(VIDIOC_SUBSCRIBE_EVENT, _EventSubscription(type=event))
            self._ioctl(VIDIOC_STREAMON, ctypes.c_int(_OUTPUT))
            self._ioctl(VIDIOC_STREAMON, ctypes.c_int(_CAPTURE))
            self._streaming = True
            for index in range(len(self._frames)):
                self._queue_frame(index)
        except OSError as error:     # the DMA heap: open, allocation, mapping
            self._close_fd()
            raise DecoderError(f"{self.device}: cannot set up {width}x{height} frame buffers: {error}") from error
        except BaseException:
            self._close_fd()
            raise
        finally:
            if heap_fd >= 0:
                os.close(heap_fd)

    def _pick_input_format(self) -> int:
        self._input_formats = []
        for index in range(64):
            desc = _FmtDesc(index=index, type=_OUTPUT)
            if not self._ioctl(VIDIOC_ENUM_FMT, desc, allow=(errno.EINVAL,)):
                break
            self._input_formats.append(int(desc.pixelformat))
        available = {_fourcc_text(code) for code in self._input_formats}
        for code in _CODEC_FOURCCS[self.codec]:
            if code in available:
                return _fourcc(code)
        raise DecoderError(f"{self.device} does not decode {self.codec} (bitstream formats: {sorted(available)})")

    def _detect_format_base(self) -> int:
        """Byte offset of the format union in this driver's ``struct v4l2_format``."""
        reply = _FormatBuffer(_OUTPUT, 8)
        self._ioctl(VIDIOC_G_FMT, reply.raw)
        for base in (8, 4):
            if struct.unpack_from("<I", reply.raw, base + 8)[0] in self._input_formats:
                return base
        raise DecoderError(f"{self.device}: unrecognised G_FMT reply: {bytes(reply.raw[:24]).hex()}")

    def _read_layout(self) -> None:
        """Frame layout of the new sequence, read when the decoder reports it.

        Reading the stream-header control is how the vendor library answers
        the resolution event; the decoder resumes only after it.
        """
        header = (_u32 * 4)()
        control = _ExtControl(id=_CID_HX_STREAM_HEADER, size=ctypes.sizeof(header),
                              ptr=ctypes.cast(header, ctypes.c_void_p))
        controls = _ExtControls(which=0, count=1, controls=ctypes.pointer(control))
        self._ioctl(VIDIOC_G_EXT_CTRLS, controls)
        width, height, _, min_frames = (int(v) for v in header)
        if min_frames > len(self._frames):
            raise DecoderError(f"{self.device}: the stream needs {min_frames} frame buffers; "
                               f"raise capture_buffers (now {len(self._frames)})")
        reply = _FormatBuffer(_CAPTURE, self._format_base)
        self._ioctl(VIDIOC_G_FMT, reply.raw)
        if _fourcc_text(reply.pixelformat) != _FRAME_FOURCC:
            raise DecoderError(f"{self.device} switched the frames to {_fourcc_text(reply.pixelformat)}")
        stride = _align(width)
        if stride > self._allocated["width"] or height > self._allocated["height"]:
            raise DecoderError(f"{self.device}: {width}x{height} coded pictures do not fit the "
                               f"{self._allocated['width']}x{self._allocated['height']} frame buffers")
        self._layout = {"coded_width": width, "coded_height": height, "stride": stride,
                        "chroma_offset": stride * height, "header": [int(v) for v in header],
                        "driver": {"width": int(reply.width), "height": int(reply.height),
                                   "bytesperline": int(reply.bytesperline(0)), "sizeimage": int(reply.sizeimage(0))},
                        "allocated": dict(self._allocated)}

    @property
    def frame_layout(self) -> dict | None:
        """Frame layout of the current sequence (None before the first picture)."""
        return None if self._layout is None else dict(self._layout)

    @property
    def sequence_info(self) -> SequenceInfo | None:
        """The most recent sequence parameter set submitted to the decoder."""
        return self._sps

    # ------------------------------------------------------------ bitstream
    def _admit(self, data: bytes) -> bool:
        """Check the access unit's SPS before it reaches the hardware; False if it cannot be decoded yet."""
        for _, payload, end in _nal_units(data):
            if payload < end and _nal_type(data, payload, self.codec) == _SPS_TYPE[self.codec]:
                try:
                    info = parse_sequence_parameter_set(data[payload:end], self.codec)
                except ValueError as error:
                    raise DecoderError(f"unreadable {self.codec} sequence parameter set: {error}") from error
                reason = info.unsupported_reason()
                width, height = _align(info.coded_width), _align(info.coded_height)
                if reason is None and (width > _align(self._options["max_size"][0])
                                       or height > _align(self._options["max_size"][1])):
                    reason = (f"{info.coded_width}x{info.coded_height} coded pictures exceed "
                              f"max_size {self._options['max_size']}")
                if reason is not None:
                    raise DecoderError(f"{self.device} cannot decode this {self.codec} sequence exactly: {reason}")
                if self._allocated is None:
                    self._start(width, height)
                elif width > self._allocated["width"] or height > self._allocated["height"]:
                    self._close_fd()        # larger pictures: a fresh decoder with larger buffers
                    self._open()
                    self._start(width, height)
                    self.stats["restarts"] += 1
                self._sps = info
        return self._sps is not None

    # --------------------------------------------------------------- queues
    def _queue_frame(self, index: int) -> None:
        planes = (_Plane * 1)()
        planes[0].length = self._frames[index].size
        planes[0].m.fd = self._frames[index].fd
        buf = _Buffer(index=index, type=_CAPTURE, memory=_DMABUF, length=1)
        buf.m.planes = planes
        self._ioctl(VIDIOC_QBUF, buf)

    def _reclaim_inputs(self) -> None:
        while True:
            planes = (_Plane * 1)()
            buf = _Buffer(type=_OUTPUT, memory=_DMABUF, length=1)
            buf.m.planes = planes
            if not self._ioctl(VIDIOC_DQBUF, buf, allow=(errno.EAGAIN, errno.EPIPE)):
                return
            self._free_inputs.append(int(buf.index))

    def _submit(self, data: bytes, deadline: float) -> int:
        size = len(data)
        if size > self._input_capacity:
            raise DecoderError(f"access unit of {size} bytes exceeds the {self._input_capacity}-byte bitstream "
                               "buffer; raise input_buffer_bytes")
        self._reclaim_inputs()
        while not self._free_inputs:
            self._wait(select.POLLOUT | select.POLLPRI, deadline, "a free bitstream buffer")
            self._reclaim_inputs()
        index = self._free_inputs.pop(0)
        target = self._inputs[index]
        target.sync(_SYNC_WRITE)
        target.map[:size] = data
        target.sync(_SYNC_WRITE | _SYNC_END)
        sequence = self._sequence = self._sequence + 1
        planes = (_Plane * 1)()
        planes[0].bytesused = size
        planes[0].length = target.size
        planes[0].m.fd = target.fd
        buf = _Buffer(index=index, type=_OUTPUT, memory=_DMABUF, length=1)
        buf.m.planes = planes
        buf.timestamp.tv_sec, buf.timestamp.tv_usec = divmod(sequence, 1_000_000)
        self._ioctl(VIDIOC_QBUF, buf)
        self._inflight[sequence] = self._sps
        while len(self._inflight) > _INFLIGHT_LIMIT:
            del self._inflight[next(iter(self._inflight))]
        self.stats["submitted"] += 1
        return sequence

    def _wait(self, events: int, deadline: float, what: str) -> None:
        """Wait for ``events`` (at most one poll slice) and handle pending V4L2 events."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{self.device}: timed out waiting for {what}")
        poller = select.poll()
        poller.register(self._fd, events)
        ready = poller.poll(max(1, int(min(remaining, _POLL_SLICE_S) * 1000)))
        if ready and ready[0][1] & select.POLLPRI:
            self._handle_events()

    def _handle_events(self) -> None:
        while True:
            event = _Event()
            if not self._ioctl(VIDIOC_DQEVENT, event, allow=(errno.ENOENT, errno.EAGAIN)):
                return
            if event.type == _EVENT_SOURCE_CHANGE:
                self.stats["source_changes"] += 1
                self._read_layout()
            if not event.pending:
                return

    def _dequeue_frame(self):
        """(sequence, frame or None) for one finished frame buffer, or None if none is ready."""
        planes = (_Plane * 1)()
        buf = _Buffer(type=_CAPTURE, memory=_DMABUF, length=1)
        buf.m.planes = planes
        if not self._ioctl(VIDIOC_DQBUF, buf, allow=(errno.EAGAIN, errno.EPIPE)):
            return None
        index = int(buf.index)
        sequence = int(buf.timestamp.tv_sec) * 1_000_000 + int(buf.timestamp.tv_usec)
        try:
            info = self._inflight.pop(sequence, None) or self._sps
            for older in [key for key in self._inflight if key < sequence]:
                del self._inflight[older]
            if buf.flags & _BUF_FLAG_ERROR:
                self.stats["decode_errors"] += 1
                return sequence, None
            if self._layout is None:
                self._handle_events()
            layout = self._layout
            if layout is None:
                raise DecoderError(f"{self.device} returned a frame before reporting its format")
            if info is None or (info.coded_width, info.coded_height) != (layout["coded_width"], layout["coded_height"]):
                raise DecoderError(f"{self.device}: decoded {layout['coded_width']}x{layout['coded_height']} "
                                   f"pictures disagree with the stream's sequence parameter set {info}")
            return sequence, self._copy_frame(self._frames[index], layout, info.crop)
        finally:
            self._queue_frame(index)

    def _copy_frame(self, source: _DmaBuffer, layout: dict, crop: tuple[int, int, int, int]) -> np.ndarray:
        stride, coded_height = layout["stride"], layout["coded_height"]
        x, y, width, height = crop
        # Rows read from the uncached mapping cost ~3x more unless they are
        # whole 16-byte vectors (846 columns: 1.67 ms, 848: 0.48 ms on the R1), so
        # such a crop reads whole vectors and is cut from the cached copy.
        left, right = x - x % 16, min(stride, _align(x + width, 16))
        offset, exact = x - left, (left, right) == (x, x + width)
        source.sync(_SYNC_READ)
        try:
            # NV12 in one plane: coded_height luma rows, then coded_height // 2 interleaved UV rows.
            rows = np.frombuffer(source.map, np.uint8, count=stride * (coded_height * 3 // 2))
            rows = rows.reshape(coded_height * 3 // 2, stride)
            luma = rows[y:y + height, left:right]
            chroma = rows[coded_height + y // 2:coded_height + (y + height) // 2, left:right]
            out = np.empty((height * 3 // 2, width), np.uint8)
            out[:height] = luma if exact else luma.copy()[:, offset:offset + width]
            if self.output == "nv12" and exact:
                out[height:] = chroma
                return out
            chroma = chroma.copy()[:, offset:offset + width]    # one read pass, then crop / split U and V
            if self.output == "nv12":
                out[height:] = chroma
                return out
            flat = out.reshape(-1)
            quarter = (height // 2) * (width // 2)
            start = height * width
            flat[start:start + quarter].reshape(height // 2, width // 2)[:] = chroma[:, 0::2]
            flat[start + quarter:].reshape(height // 2, width // 2)[:] = chroma[:, 1::2]
            return out
        finally:
            source.sync(_SYNC_READ | _SYNC_END)

    # --------------------------------------------------------------- public
    def decode(self, access_unit, *, timeout: float | None = None) -> np.ndarray | None:
        """Decode one access unit and return its picture.

        Returns the newest finished picture. If the submitted picture does not
        finish within ``timeout`` (or the decoder flags it as corrupt), the
        previous picture is returned instead (None before the first one) and
        ``stats`` counts the event, matching kuavo_deploy's PyAV decoder.
        Access units before the first sequence parameter set are not
        submitted (nothing could decode them) and return the previous picture.
        Raises ``DecoderError`` for a sequence this path cannot decode exactly.
        """
        if self._fd < 0:
            raise DecoderError("decoder is closed")
        data = access_unit if isinstance(access_unit, bytes) else bytes(access_unit)
        if not data:
            raise ValueError("empty access unit")
        if not self._admit(data):
            self.stats["skipped_without_headers"] += 1
            return self._last
        deadline = time.monotonic() + (self.timeout if timeout is None else float(timeout))
        sequence = self._submit(data, deadline)
        newest = None
        while True:
            item = self._dequeue_frame()
            while item is not None:
                done, frame = item
                if frame is not None:
                    if newest is not None:
                        self.stats["superseded"] += 1
                    newest = frame
                    self.stats["frames"] += 1
                if done >= sequence:
                    if newest is not None:
                        self._last = newest
                    return self._last
                item = self._dequeue_frame()
            try:
                # hx-vpu signals a finished picture through the returned bitstream buffer.
                self._wait(select.POLLIN | select.POLLOUT | select.POLLPRI, deadline, "the decoded picture")
                self._reclaim_inputs()
            except TimeoutError:
                self.stats["timeouts"] += 1
                if newest is not None:
                    self._last = newest
                return self._last

    def reset(self) -> None:
        """Forget the stream (new decoder context); the next picture must be an IDR with its headers."""
        self._close_fd()
        self._last = None
        self._open()

    @property
    def is_initialized(self) -> bool:
        return self._last is not None

    def close(self) -> None:
        self._close_fd()

    def _close_fd(self) -> None:
        if self._fd < 0:
            return
        try:
            if self._streaming:
                for kind in (_OUTPUT, _CAPTURE):
                    self._ioctl(VIDIOC_STREAMOFF, ctypes.c_int(kind), allow=(errno.EINVAL,))
                self._streaming = False
            for kind in (_OUTPUT, _CAPTURE):
                self._ioctl(VIDIOC_REQBUFS, _RequestBuffers(count=0, type=kind, memory=_DMABUF),
                            allow=(errno.EINVAL, errno.EBUSY))
        finally:
            for item in self._inputs + self._frames:
                item.close()
            self._inputs, self._frames = [], []
            os.close(self._fd)
            self._fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def yuv_planes(frame: np.ndarray, output: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Y, U, V) views of a decoded ``nv12`` / ``i420`` frame."""
    height, width = frame.shape[0] * 2 // 3, frame.shape[1]
    if output == "nv12":
        chroma = frame[height:]
        return frame[:height], chroma[:, 0::2], chroma[:, 1::2]
    flat, quarter = frame.reshape(-1), (height // 2) * (width // 2)
    rest = flat[height * width:]
    return (frame[:height], rest[:quarter].reshape(height // 2, width // 2),
            rest[quarter:].reshape(height // 2, width // 2))


def _main(argv=None) -> int:
    import argparse
    import json
    import resource

    parser = argparse.ArgumentParser(description="Decode an Annex-B H.264/H.265 file on the R1 hardware decoder "
                                                 "and report speed, CPU time and (optionally) agreement with PyAV.")
    parser.add_argument("stream", help="raw Annex-B elementary stream (.h264 / .h265)")
    parser.add_argument("--codec", choices=("h264", "h265"), required=True)
    parser.add_argument("--output", choices=OUTPUTS, default="i420")
    parser.add_argument("--max-size", default="1920x1088", help="largest WIDTHxHEIGHT the stream may have")
    parser.add_argument("--low-delay", action="store_true", help="turn off the display-reorder queue (no B-frames)")
    parser.add_argument("--rate", type=float, default=0.0, help="submit at this many access units per second (0 = as fast as possible)")
    parser.add_argument("--compare-pyav", action="store_true", help="also decode with PyAV and compare Y/U/V (needs `av`)")
    parser.add_argument("--json", help="write the report here")
    args = parser.parse_args(argv)
    units = split_access_units(Path(args.stream).read_bytes(), args.codec)
    reference = None
    if args.compare_pyav:
        import av  # noqa: PLC0415
        ctx = av.CodecContext.create("hevc" if args.codec == "h265" else "h264", "r")
        reference = []
        for unit in units:
            frames = ctx.decode(av.Packet(unit))
            if not frames:
                reference.append(None)
                continue
            frame = frames[-1]
            planes = []
            for index, (rows, cols) in enumerate(((frame.height, frame.width),) + ((frame.height // 2, frame.width // 2),) * 2):
                plane = frame.planes[index]
                planes.append(np.frombuffer(plane, np.uint8).reshape(-1, plane.line_size)[:rows, :cols])
            reference.append(planes)
    report = {"stream": args.stream, "codec": args.codec, "output": args.output, "access_units": len(units)}
    latencies, compared, mismatched, max_abs = [], 0, 0, 0
    max_size = tuple(int(v) for v in args.max_size.lower().split("x"))
    with R1VideoDecoder(args.codec, args.output, max_size=max_size, low_delay=args.low_delay) as decoder:
        cpu0, wall0 = time.process_time(), time.perf_counter()
        previous = None
        for k, unit in enumerate(units):
            if args.rate > 0:
                delay = wall0 + k / args.rate - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            started = time.perf_counter()
            frame = decoder.decode(unit)
            latencies.append((time.perf_counter() - started) * 1000)
            if reference is not None and reference[k] is not None:
                compared += 1
                if frame is None or frame is previous:
                    mismatched += 1
                else:
                    got = yuv_planes(frame, args.output)
                    diffs = [int(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()) if a.shape == b.shape else 255
                             for a, b in zip(got, reference[k])]
                    mismatched += int(any(diffs))
                    max_abs = max(max_abs, *diffs)
            previous = frame
        wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
        report.update(frames=decoder.stats["frames"], stats=decoder.stats, layout=decoder.frame_layout,
                      sequence=None if decoder.sequence_info is None else vars(decoder.sequence_info))
    ordered = sorted(latencies)
    report.update({
        "wall_s": round(wall, 3), "frames_per_s": round(len(units) / wall, 1),
        "cpu_ms_per_frame": round(cpu * 1000 / max(1, len(units)), 3),
        "latency_ms": {"p50": round(ordered[len(ordered) // 2], 3),
                       "p99": round(ordered[max(0, int(len(ordered) * 0.99) - 1)], 3),
                       "max": round(ordered[-1], 3)} if ordered else None,
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)})
    if reference is not None:
        report["pyav_yuv"] = {"frames_compared": compared, "frames_differing": mismatched, "max_abs": max_abs}
    text = json.dumps(report, indent=1, default=str)
    if args.json:
        Path(args.json).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
