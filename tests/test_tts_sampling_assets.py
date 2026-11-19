from pathlib import Path
import struct
import subprocess

import torch


ROOT = Path(__file__).resolve().parents[1]


def test_unified_tts_sampling_tables_validate_and_select_payload(tmp_path):
    sizes = (14 * 68 * 2, 14 * 256 * 2, 20 * 68 * 2, 20 * 256 * 2)
    payloads = [bytes([index + 1]) * size for index, size in enumerate(sizes)]
    header = struct.pack("<8s6I", b"RFTTS001", 1, 32, *sizes)
    table = tmp_path / "tts_sampling_tables.bin"
    table.write_bytes(header + b"".join(payloads))
    source = tmp_path / "tables.cpp"
    source.write_text(r'''
#include "tts_sampling_assets.h"
#include <cassert>
#include <vector>
int main(int argc, char** argv) {
    std::vector<unsigned char> pp(1904), ps(7168), tp(2720), ts(10240);
    tts_sampling_assets::read(argv[1], tts_sampling_assets::TableSet::Predictor,
                              pp.data(), pp.size(), ps.data(), ps.size());
    tts_sampling_assets::read(argv[1], tts_sampling_assets::TableSet::Talker,
                              tp.data(), tp.size(), ts.data(), ts.size());
    assert(pp.front() == 1 && pp.back() == 1 && ps.front() == 2 && ps.back() == 2);
    assert(tp.front() == 3 && tp.back() == 3 && ts.front() == 4 && ts.back() == 4);
}
''')
    root = Path(torch.__file__).resolve().parent
    executable = tmp_path / "tables"
    subprocess.run([
        "g++", "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(ROOT / "src/ops"), "-I", str(root / "include"), str(source),
        "-L", str(root / "lib"), "-lc10", f"-Wl,-rpath,{root / 'lib'}", "-o", str(executable),
    ], check=True)
    subprocess.run([str(executable), str(tmp_path)], check=True)
    table.write_bytes(b"BADMAGIC" + table.read_bytes()[8:])
    failed = subprocess.run([str(executable), str(tmp_path)], capture_output=True, text=True)
    assert failed.returncode != 0 and "table schema" in failed.stderr
