"""kuavo_deploy camera decoding on the R1 video decoder (host doubles, no decoder device).

The hardware itself is checked on the board (examples/configs/pi05/README.md,
Kuavo camera decoding); these tests cover the parts that decide what reaches
it and what the drop-in returns.
"""
import importlib
import importlib.util
import shutil
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "examples/runners/pi05/kuavo_deploy/utils"
START = b"\x00\x00\x00\x01"


def _load_decoder_module():
    name = "kuavo_r1_video_decoder_under_test"
    spec = importlib.util.spec_from_file_location(name, UTILS / "r1_video_decoder.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


vdec = _load_decoder_module()
_REAL_START = vdec.R1VideoDecoder._start

# ---------------------------------------------------------------- kernel ABI


def test_ioctl_numbers_and_struct_sizes_match_the_64bit_linux_abi():
    vdec._check_abi()
    expected = {"VIDIOC_QUERYCAP": 0x80685600, "VIDIOC_ENUM_FMT": 0xC0405602, "VIDIOC_G_FMT": 0xC0D05604,
                "VIDIOC_S_FMT": 0xC0D05605, "VIDIOC_REQBUFS": 0xC0145608, "VIDIOC_QBUF": 0xC058560F,
                "VIDIOC_DQBUF": 0xC0585611, "VIDIOC_STREAMON": 0x40045612, "VIDIOC_STREAMOFF": 0x40045613,
                "VIDIOC_S_CTRL": 0xC008561C, "VIDIOC_G_EXT_CTRLS": 0xC0205647, "VIDIOC_DQEVENT": 0x80885659,
                "VIDIOC_SUBSCRIBE_EVENT": 0x4020565A, "DMA_HEAP_IOCTL_ALLOC": 0xC0184800,
                "DMA_BUF_IOCTL_SYNC": 0x40086200}
    assert {name: getattr(vdec, name) for name in expected} == expected


# ---------------------------------------------------------------- sequence parameter sets

H264_848 = "6764001facb606a1ed08000003000800000301e478c197"
H264_640 = "6764001eacb60501ed08000003000800000301e478b177"
H264_1080P = "67640028acb603c0113f2c20000003002000000791e3065c"
H264_10BIT = "676e000aa6cb20c5fcb080000003008000001e07891324"
H265_848 = "42010101600000030090000003000003005aa006a201e16592a4932b9a020000030002000003003c10"

# x264 / x265 encodes; the scaling-list and POC-type-1 SPSs were written by hand and
# decoded by FFmpeg spliced into an 848x480 stream (it agrees on every coded size).
SPS_CASES = [
    # codec, SPS (NAL header first), coded (w, h), visible (x, y, w, h), refusal
    ("h264", H264_848, (848, 480), (0, 0, 848, 480), None),
    ("h264", H264_1080P, (1920, 1088), (0, 0, 1920, 1080), None),
    ("h264", "6764001facb606a1ef5508000003000800000301e478c197", (848, 480), (0, 0, 846, 478), None),
    ("h264", H264_640, (640, 480), (0, 0, 640, 480), None),
    ("h264", "6764000aacb20c5f5342000003000200000300781e244c90", (96, 80), (0, 0, 94, 70), None),
    ("h264", "6764000aacb20c5fcb08000003000800000301e078913240", (96, 80), (0, 0, 96, 72), None),  # cqm=jvt
    ("h264", "6764001fadd90e228a61364388c22d90e228a6136438893ccc85479990a8f332151e6642a3ccc85479990a8f332151e6"
             "642a3ccc8547846d01a87b20", (848, 480), (0, 0, 848, 480), None),  # explicit scaling lists
    ("h264", "6764001fad364388a2984d90e22279990a8f332151e6642a3ccc85479990a8f332151e6642a3ccc85479990a8eb406a1"
             "ee9c88", (848, 480), (2, 0, 842, 474), None),  # scaling lists + left crop
    ("h264", "6764001faca9c84143e80d43d9", (848, 480), (0, 0, 848, 480), None),  # pic_order_cnt_type 1
    ("h264", "67640015ace419bf3c20000003002000000783e28549", (96, 96), (0, 0, 96, 72), "field or MBAFF"),
    ("h264", "677a000abcb20c5fc4c2000003000200000300781e244c90", None, None, "chroma_format_idc 2"),
    ("h264", "67f4000a9196418bf89840000003004000000f03c48992", None, None, "chroma_format_idc 3"),
    ("h264", "6764000af2c8317f1308000003000800000301e078913240", None, None, "chroma_format_idc 0"),
    ("h264", H264_10BIT, None, None, "bit depth 10/10"),
    ("h265", H265_848, (848, 480), (0, 0, 848, 480), None),
    ("h265", "42010101600000030090000003000003005aa006a201e1d56592a4932b9a020000030002000003003c10",
     (848, 480), (0, 0, 846, 478), None),
    ("h265", "42010101600000030090000003000003001ea030812755964a924cae6808000003000800000300f040",
     (96, 72), (0, 0, 94, 70), None),
    ("h265", "42010401600000030090000003000003001e0000a0308125965652b295964932b9a020000003002000000303c1",
     (96, 72), (0, 0, 96, 72), None),  # sps_max_sub_layers_minus1 = 2
    ("h265", "42010601600000030090000003000003001e0000a0308125965624ac495893104c9265734040000003004000000782",
     (96, 72), (0, 0, 96, 72), None),  # sps_max_sub_layers_minus1 = 3
    ("h265", "42010102200000030090000003000003001ea0308124d964a924cae68080000003008000000f04", None, None,
     "bit depth 10/10"),
    ("h265", "4201010408000003009d0800000300001eb0308125964a924cae6808000003000800000300f040", None, None,
     "chroma_format_idc 2"),
    ("h265", "4201010408000003009e0800000300001e90061024b2c9524995cd010000030001000003001e08", None, None,
     "chroma_format_idc 3"),
]


@pytest.mark.parametrize("codec,sps,coded,visible,refusal", SPS_CASES)
def test_sequence_parameter_sets_give_the_decoded_picture_geometry(codec, sps, coded, visible, refusal):
    info = vdec.parse_sequence_parameter_set(bytes.fromhex(sps), codec)
    if coded is not None:
        assert (info.coded_width, info.coded_height, info.crop) == (*coded, visible)
    reason = info.unsupported_reason()
    if refusal is None:
        assert reason is None
    else:
        assert refusal in reason


def test_unreadable_or_foreign_nal_units_are_rejected():
    with pytest.raises(ValueError):
        vdec.parse_sequence_parameter_set(bytes.fromhex(H264_848)[:5], "h264")
    with pytest.raises(ValueError, match="not a h264 sequence parameter set"):
        vdec.parse_sequence_parameter_set(b"\x68\xee\x3c\x80", "h264")
    assert not vdec.is_sequence_parameter_set(bytes.fromhex(H265_848), "h264")


# ---------------------------------------------------------------- access units


def test_annex_b_streams_split_into_pictures():
    sps, pps = bytes.fromhex(H264_848), b"\x68\xee\x3c\x80"
    idr, p, p_second_slice, sei = b"\x65\x88\x84", b"\x41\x9a\x02", b"\x41\x20\x11", b"\x06\x05\x01"
    stream = START + sps + START + pps + b"\x00\x00\x01" + idr + START + p + START + p_second_slice + START + sei + START + p
    units = vdec.split_access_units(stream, "h264")
    assert units == [START + sps + START + pps + b"\x00\x00\x01" + idr, START + p + START + p_second_slice,
                     START + sei + START + p]
    assert vdec.parameter_sets(units[0], "h264") == [sps, pps]
    assert vdec.parameter_sets(units[1], "h264") == []

    vps, sps5, pps5 = b"\x40\x01\x0c\x01", bytes.fromhex(H265_848), b"\x44\x01\xc1\x72"
    idr5, trail, dependent, aud = b"\x26\x01\xaf\x10", b"\x02\x01\xd0\x08", b"\x02\x01\x40\x22", b"\x46\x01\x50"
    stream = START + vps + START + sps5 + START + pps5 + START + idr5 + START + trail + START + dependent + START + aud + START + trail
    units = vdec.split_access_units(stream, "hevc")
    assert len(units) == 3 and units[1] == START + trail + START + dependent
    found = vdec.parameter_sets(units[0], "h265")
    assert found == [vps, sps5, pps5] and [vdec.is_sequence_parameter_set(n, "h265") for n in found] == [False, True, False]


# ---------------------------------------------------------------- admission (no device)


def _au(sps_hex, slice_nal=b"\x65\x88\x84"):
    return START + bytes.fromhex(sps_hex) + START + slice_nal


@pytest.fixture
def offline(monkeypatch):
    """R1VideoDecoder whose device calls are recorded instead of made."""
    calls = []

    def fake_open(self):
        self._fd, self._allocated, self._layout, self._sps = 1 << 20, None, None, None
        self._inflight, self._sequence, self._inputs, self._frames, self._streaming = {}, 0, [], [], False
        calls.append("open")

    def fake_start(self, width, height):
        self._allocated = {"width": width, "height": height}
        calls.append(("start", width, height))

    def fake_close(self):
        self._fd = -1
        calls.append("close")

    monkeypatch.setattr(vdec.R1VideoDecoder, "_open", fake_open)
    monkeypatch.setattr(vdec.R1VideoDecoder, "_start", fake_start)
    monkeypatch.setattr(vdec.R1VideoDecoder, "_close_fd", fake_close)
    made = []

    def make(codec="h264", **options):
        decoder = vdec.R1VideoDecoder(codec, device="/dev/test-vpu", **options)
        made.append(decoder)
        return decoder, calls

    yield make
    for decoder in made:
        decoder._fd = -1  # never let the real close() see the made-up descriptor


def test_frame_buffers_follow_the_stream_and_grow_by_restarting(offline):
    decoder, calls = offline("h264")
    assert calls == ["open"]
    assert decoder._admit(_au(H264_848)) is True
    assert calls[-1] == ("start", 896, 512)  # coded size rounded up to 64, not max_size
    assert decoder.sequence_info.crop == (0, 0, 848, 480)
    decoder._admit(_au(H264_640))  # fits the 896x512 buffers
    decoder._admit(_au(H264_848))
    assert calls == ["open", ("start", 896, 512)] and decoder.stats["restarts"] == 0
    decoder._admit(_au(H264_1080P))
    assert calls[-3:] == ["close", "open", ("start", 1920, 1088)] and decoder.stats["restarts"] == 1
    assert decoder.sequence_info.crop == (0, 0, 1920, 1080)


def test_sequences_the_hardware_cannot_reproduce_are_refused_before_submission(offline):
    decoder, calls = offline("h264", max_size=(1280, 720))
    decoder._admit(_au(H264_848))
    with pytest.raises(vdec.DecoderError, match="bit depth 10/10"):
        decoder._admit(_au(H264_10BIT))
    with pytest.raises(vdec.DecoderError, match="exceed max_size"):
        decoder._admit(_au(H264_1080P))
    with pytest.raises(vdec.DecoderError, match="unreadable"):
        decoder._admit(START + bytes.fromhex(H264_848)[:5] + START + b"\x65\x88\x84")
    assert calls == ["open", ("start", 896, 512)]
    assert decoder.sequence_info.crop == (0, 0, 848, 480)  # still decoding the earlier sequence


def test_access_units_before_the_first_sps_are_not_submitted(offline, monkeypatch):
    decoder, calls = offline("h265")
    monkeypatch.setattr(decoder, "_submit", lambda *args: pytest.fail("submitted without a sequence"))
    assert decoder.decode(START + b"\x02\x01\xd0\x08") is None
    assert decoder.stats["skipped_without_headers"] == 1 and calls == ["open"]


def test_a_missing_device_node_is_a_decoder_error(tmp_path):
    with pytest.raises(vdec.DecoderError, match="cannot open"):
        vdec.R1VideoDecoder("h264", device=str(tmp_path / "video99"))


def test_frame_buffer_failures_are_decoder_errors(offline, monkeypatch, tmp_path):
    decoder, calls = offline("h264", heap=str(tmp_path / "no-heap"))
    monkeypatch.setattr(vdec.R1VideoDecoder, "_start", _REAL_START)  # the real setup, ioctls accepted as sent
    monkeypatch.setattr(decoder, "_ioctl", lambda request, arg, allow=(): True)
    decoder._format_base, decoder._input_fourcc = 4, vdec._fourcc("H264")
    with pytest.raises(vdec.DecoderError, match="cannot set up 896x512 frame buffers"):
        decoder._admit(_au(H264_848))
    assert calls[-1] == "close"


# ---------------------------------------------------------------- frame copy


class _MappedFrame:
    def __init__(self, data):
        self.map = bytearray(data)
        self.syncs = []

    def sync(self, flags):
        self.syncs.append(flags)


@pytest.mark.parametrize("output", vdec.OUTPUTS)
@pytest.mark.parametrize("crop", [(0, 0, 96, 80), (0, 0, 94, 70), (2, 4, 90, 72), (16, 2, 64, 76), (18, 0, 78, 80)])
def test_pictures_are_cut_from_the_hardware_frame_layout(output, crop):
    stride, coded_height = 128, 80  # 96x80 coded, rows 64-byte aligned, chroma after coded_height rows
    nv12 = np.random.default_rng(0).integers(0, 256, (coded_height * 3 // 2, stride), dtype=np.uint8)
    decoder = vdec.R1VideoDecoder.__new__(vdec.R1VideoDecoder)
    decoder._fd, decoder.output = -1, output
    source = _MappedFrame(nv12.tobytes())
    frame = decoder._copy_frame(source, {"stride": stride, "coded_height": coded_height}, crop)
    x, y, width, height = crop
    chroma = nv12[coded_height + y // 2:coded_height + (y + height) // 2, x:x + width]
    planes = vdec.yuv_planes(frame, output)
    assert frame.shape == (height * 3 // 2, width) and frame.flags.c_contiguous
    assert np.array_equal(planes[0], nv12[y:y + height, x:x + width])
    assert np.array_equal(planes[1], chroma[:, 0::2]) and np.array_equal(planes[2], chroma[:, 1::2])
    assert source.syncs == [vdec._SYNC_READ, vdec._SYNC_READ | vdec._SYNC_END]


# ---------------------------------------------------------------- kuavo_deploy drop-in

ENCODERS = {"h264": ("libx264", "x264-params", ""), "hevc": ("libx265", "x265-params", "repeat-headers=1:log-level=error")}
FULL_RANGE_BT709 = {"h264": "colorprim=bt709:transfer=bt709:colormatrix=bt709:fullrange=on",
                    "hevc": "colorprim=bt709:transfer=bt709:colormatrix=bt709:range=full"}


def _encode(av, codec, *, size=(96, 80), frames=8, gop=4, extra=""):
    """Annex-B access units of a small moving test pattern (IDR every ``gop`` pictures, no B-frames)."""
    name, key, base = ENCODERS[codec]
    if name not in av.codecs_available:
        pytest.skip(f"PyAV lacks {name}")
    width, height = size
    context = av.CodecContext.create(name, "w")
    context.width, context.height, context.pix_fmt = width, height, "yuv420p"
    context.time_base, context.framerate = Fraction(1, 30), Fraction(30, 1)
    params = ":".join(p for p in (f"keyint={gop}:min-keyint={gop}:scenecut=0:bframes=0", base, extra) if p)
    context.options = {"preset": "ultrafast", "tune": "zerolatency", key: params}
    rows, cols = np.mgrid[0:height, 0:width]
    units = []
    for index in range(frames):
        rgb = np.stack([(cols * 2 + index * 9) % 256, (rows * 3) % 256, (rows + cols + index * 17) % 256], -1)
        frame = av.VideoFrame.from_ndarray(rgb.astype(np.uint8), format="rgb24").reformat(format="yuv420p")
        frame.pts = index
        units += [bytes(packet) for packet in context.encode(frame)]
    units += [bytes(packet) for packet in context.encode(None)]
    return units


def _without_parameter_sets(unit, codec):
    kept = vdec.parameter_sets(unit, codec)
    return b"".join(unit[begin:end] for begin, payload, end in vdec._nal_units(unit)
                    if unit[payload:end].rstrip(b"\x00") not in kept)


def _customer_frames(av, codec, units):
    """LeTools-Learning RealtimeH265Decoder.decode at the stream size: PyAV, then rgb24."""
    context, last, frames = av.CodecContext.create(codec, "r"), None, []
    for unit in units:
        try:
            decoded = context.decode(av.Packet(unit))
        except (av.error.InvalidDataError, ValueError):  # av.error.ValueError: PyAV < 19
            decoded = []
        if decoded:
            last = np.ascontiguousarray(decoded[-1].to_ndarray(format="rgb24"))
        frames.append(last)
    return frames


class _HardwareDouble:
    """R1VideoDecoder's contract without the device: exact SPS crop as I420, previous picture if none is new."""

    made = []
    av = None  # PyAV, set by the dropin fixture

    def __init__(self, codec, output, *, max_size, timeout):
        assert output == "i420"
        self.context = self.av.CodecContext.create(codec, "r")
        self.context.options = {"flags": "unaligned"}  # FFmpeg otherwise rounds a left crop down
        self.stats = {"timeouts": 0}
        self.last, self.stall, self.refuse, self.closed = None, False, None, False
        _HardwareDouble.made.append(self)

    def decode(self, unit):
        if self.refuse is not None:
            raise self.refuse
        if self.stall:
            self.stats["timeouts"] += 1
            return self.last
        for frame in self.context.decode(self.av.Packet(unit)):
            height, width = frame.height, frame.width
            planes = [np.frombuffer(frame.planes[i], np.uint8).reshape(-1, frame.planes[i].line_size)[:h, :w]
                      for i, (h, w) in enumerate([(height, width)] + [(height // 2, width // 2)] * 2)]
            self.last = np.concatenate([p.reshape(-1) for p in planes]).reshape(height * 3 // 2, width)
        return self.last

    def close(self):
        self.closed = True


@pytest.fixture
def av():
    """PyAV, the customer's decoder and the reference here. Optional: the CI wheelhouse does not carry it."""
    return pytest.importorskip("av")


@pytest.fixture
def dropin(av, tmp_path, monkeypatch):
    """kuavo_deploy.utils.h265_decoder as deployed (both files copied in), on the hardware double."""
    utils = tmp_path / "kuavo_deploy" / "utils"  # a namespace package, as in LeTools-Learning
    utils.mkdir(parents=True)
    for name in ("h265_decoder.py", "r1_video_decoder.py"):
        shutil.copy(UTILS / name, utils / name)
    if importlib.util.find_spec("cv2") is None:  # the customer's resize; optional here, like PyAV
        stub = ModuleType("cv2")
        stub.INTER_LINEAR = 1
        stub.resize = lambda image, size, interpolation: np.zeros((size[1], size[0], 3), np.uint8)
        monkeypatch.setitem(sys.modules, "cv2", stub)
    monkeypatch.delenv("KUAVO_VIDEO_DECODER", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [n for n in sys.modules if n.split(".")[0] == "kuavo_deploy"]:
        monkeypatch.delitem(sys.modules, name)
    importlib.invalidate_caches()
    module = importlib.import_module("kuavo_deploy.utils.h265_decoder")
    monkeypatch.setattr(module, "R1VideoDecoder", _HardwareDouble)
    monkeypatch.setattr(_HardwareDouble, "av", av)
    _HardwareDouble.made = []
    yield module
    for name in [n for n in sys.modules if n.split(".")[0] == "kuavo_deploy"]:
        del sys.modules[name]


def _run(decoder, units):
    return [decoder.decode(unit) for unit in units]


def _same(got, want):
    return len(got) == len(want) and all(
        (a is None and b is None) or (a is not None and b is not None and np.array_equal(a, b)) for a, b in zip(got, want))


@pytest.mark.parametrize("codec", ["h264", "hevc"])
@pytest.mark.parametrize("signalled", [False, True])
def test_dropin_rgb_is_byte_identical_to_the_customer_pyav_path(dropin, av, codec, signalled):
    units = _encode(av, codec, extra=FULL_RANGE_BT709[codec] if signalled else "")
    decoder = dropin.RealtimeH265Decoder(96, 80, codec=codec)
    got = _run(decoder, units)
    assert decoder.backend == "r1" and decoder.fallback_reason is None
    assert _same(got, _customer_frames(av, codec, units)) and got[-1] is not None
    # One PyAV probe per sequence, although the parameter sets repeat at every IDR.
    assert decoder.stats["probe_decodes"] == 1 and decoder.is_initialized
    assert all(frame.flags.c_contiguous and frame.dtype == np.uint8 for frame in got)


@pytest.mark.parametrize("backend", ["r1", "pyav"])
def test_dropin_skips_empty_messages_instead_of_draining_pyav(dropin, av, backend):
    units = _encode(av, "hevc", frames=4)
    decoder = dropin.RealtimeH265Decoder(96, 80, backend=backend)
    assert decoder.decode(b"") is None
    got = _run(decoder, units[:2]) + [decoder.decode(b"")] + _run(decoder, units[2:])
    want = _customer_frames(av, "hevc", units)
    assert _same(got, want[:2] + [want[1]] + want[2:]) and decoder.backend == backend


def test_dropin_falls_back_to_pyav_without_the_device_unless_r1_is_required(dropin, av, monkeypatch):
    def missing(*args, **kwargs):
        raise dropin.DecoderError("no V4L2 device named hx-vpu-dec*")

    monkeypatch.setattr(dropin, "R1VideoDecoder", missing)
    units = _encode(av, "hevc", frames=4)
    decoder = dropin.RealtimeH265Decoder(96, 80)
    assert decoder.backend == "pyav" and "no V4L2 device" in decoder.fallback_reason
    assert _same(_run(decoder, units), _customer_frames(av, "hevc", units))
    with pytest.raises(dropin.DecoderError, match="no V4L2 device"):
        dropin.RealtimeH265Decoder(96, 80, backend="r1")
    monkeypatch.setenv("KUAVO_VIDEO_DECODER", "pyav")
    assert dropin.RealtimeH265Decoder(96, 80).backend == "pyav"
    with pytest.raises(ValueError, match="auto, r1 or pyav"):
        dropin.RealtimeH265Decoder(96, 80, backend="gpu")


def test_dropin_switches_a_refused_stream_to_pyav_from_the_same_access_unit(dropin, av):
    units = _encode(av, "h264", frames=8, gop=4)
    decoder = dropin.RealtimeH265Decoder(96, 80, codec="h264")
    want = _customer_frames(av, "h264", units)
    got = _run(decoder, units[:4])
    _HardwareDouble.made[-1].refuse = dropin.DecoderError("bit depth 10/10 (only 8-bit is supported)")
    got += _run(decoder, units[4:])  # unit 4 is the next IDR, with its parameter sets
    assert decoder.backend == "pyav" and "bit depth" in decoder.fallback_reason
    assert _HardwareDouble.made[-1].closed and _same(got, want)
    decoder.reset()  # a new stream may suit the hardware again
    assert decoder.backend == "r1" and decoder.fallback_reason is None and not decoder.is_initialized


def test_dropin_falls_back_after_three_timeouts_and_resumes_at_the_next_idr(dropin, av):
    units = _encode(av, "h264", frames=12, gop=4)
    units[1:] = [_without_parameter_sets(unit, "h264") for unit in units[1:]]  # sent once, at the start
    want = _customer_frames(av, "h264", units)
    decoder = dropin.RealtimeH265Decoder(96, 80, codec="h264")
    got = _run(decoder, units[:5])
    _HardwareDouble.made[-1].stall = True
    got += _run(decoder, units[5:])
    assert decoder.backend == "pyav" and "timed out 3 times" in decoder.fallback_reason
    assert all(np.array_equal(frame, want[4]) for frame in got[5:8])  # the last good picture meanwhile
    assert _same(got[8:], want[8:])  # PyAV restarts at the IDR, with the parameter sets kept from unit 0


def test_dropin_leaves_pictures_pyav_would_crop_differently_to_pyav(dropin, av):
    # A left crop FFmpeg rounds down for alignment: PyAV shows 92 columns, the SPS 90.
    units = _encode(av, "h264", frames=4, extra="crop-rect=2,0,4,6")
    decoder = dropin.RealtimeH265Decoder(92, 74, codec="h264")
    got = _run(decoder, units)
    assert decoder.backend == "pyav" and "(90, 74) differs from PyAV's (92, 74)" in decoder.fallback_reason
    assert _same(got, _customer_frames(av, "h264", units))


def test_dropin_resizes_with_the_customer_opencv_call(dropin, av, monkeypatch):
    calls = []

    def resize(image, size, interpolation):
        calls.append((image.shape, size, interpolation))
        return np.zeros((size[1], size[0], 3), np.uint8)

    monkeypatch.setattr(dropin.cv2, "resize", resize)
    units = _encode(av, "hevc", frames=2)
    frame = dropin.RealtimeH265Decoder(64, 48).decode(units[0])
    assert frame.shape == (48, 64, 3) and calls == [((80, 96, 3), (64, 48), dropin.cv2.INTER_LINEAR)]
