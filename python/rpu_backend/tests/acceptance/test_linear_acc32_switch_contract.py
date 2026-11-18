from pathlib import Path
import re
ROOT = Path(__file__).resolve().parents[4]
LINEAR_CPP = ROOT / 'src' / 'ops' / 'rpu_linear.cpp'
WALL_OSS_FUSED_SOURCES = (ROOT / 'src' / 'core' / 'fused_model_base.cpp', ROOT / 'src' / 'fused' / 'rpu_qwen25vl_vision_model.cpp', ROOT / 'src' / 'fused' / 'rpu_qwen3_model.h', ROOT / 'src' / 'fused' / 'rpu_wall_oss_action_step_model.cpp')

def _acc16_launcher_body() -> str:
    source = LINEAR_CPP.read_text()
    start = source.index('void rpu_launch_linear_spm_to_spm_acc16_kernel(')
    return source[start:]

def test_per_handle_acc32_precedes_quantized_autotile_dispatch():
    source = LINEAR_CPP.read_text()
    body = _acc16_launcher_body()
    assert 'std::getenv("RPU_LINEAR_ACC32")' not in source
    switch = body.index('const bool use_acc32 = force_acc32;')
    quant = body.index('if (weight.scalar_type() == at::kChar || fp8_e4m3_weight)')
    assert switch < quant
    override = body[switch:quant]
    assert 'weight.scalar_type() == at::kHalf' in override
    assert 'if (force_acc32)' in override
    assert 'rpu_launch_linear_spm_to_spm_kernel(' in override
    assert 'force_gemm' not in body
    assert '(!scale.defined() || scale.numel() == 0)' in override
    gemv = override.index('if (M == 1 && prefer_gemv && !use_acc32')
    acc32 = override.index('if (use_acc32 && weight.scalar_type() == at::kHalf)')
    assert override.index('if (force_acc32)') < gemv < acc32
    assert 'rpu_launch_gemv_spm_to_spm_kernel(' in override[gemv:acc32]

def test_wall_oss_fused_linears_share_the_typed_launcher():
    call = re.compile('\\brpu_launch_linear_spm_to_spm_acc16_kernel\\s*\\(')
    raw_acc32 = re.compile('\\brpu_launch_linear_spm_to_spm_kernel\\s*\\(')
    eager_ddr = re.compile('\\brpu_launch_linear_ddr_kernel\\s*\\(')
    total = 0
    for path in WALL_OSS_FUSED_SOURCES:
        source = path.read_text()
        calls = len(call.findall(source))
        total += calls
        assert calls > 0, f'{path.relative_to(ROOT)} no longer uses the shared fused launcher'
        assert not raw_acc32.search(source), f'{path.relative_to(ROOT)} bypasses the typed accumulation policy'
        assert not eager_ddr.search(source), f'{path.relative_to(ROOT)} bypasses the fused Linear precision policy'
    assert total > 0

def test_process_global_acc32_fallback_and_getter_are_deleted():
    source = LINEAR_CPP.read_text()
    assert 'linear_acc32_enabled' not in source
    assert 'rpu_get_linear_acc32_enabled' not in source
    assert 'rpu_get_linear_acc32_enabled' not in (ROOT / 'src' / 'core' / 'rpu_kernel_decls.h').read_text()
    assert 'm.def("get_linear_acc32"' not in (ROOT / 'src' / 'core' / 'rpu_pybind.inc').read_text()
    assert 'def get_linear_acc32() -> bool:' not in (ROOT / 'python' / 'rpu_backend' / 'runtime' / 'control.py').read_text()
    assert '"get_spm_mode get_linear_acc32 "' not in (ROOT / 'python' / 'rpu_backend' / '_torch_rpu.py').read_text()

def test_force_gemm_parameter_is_deleted_repo_wide_from_native_sources():
    for path in (ROOT / 'src').rglob('*'):
        if path.suffix in {'.cpp', '.h', '.inc'}:
            assert 'force_gemm' not in path.read_text(), path

def test_qwen3_vl_uses_cold_per_handle_acc32_without_touching_shared_installer():
    model = (ROOT / 'src' / 'fused' / 'rpu_qwen3_model.h').read_text()
    vision = (ROOT / 'src' / 'fused' / 'rpu_qwen3vl_vision_model.cpp').read_text()
    adapter = (ROOT / 'python' / 'rpu_backend' / 'adapters' / 'qwen3_vl' / '__init__.py').read_text()
    shared = (ROOT / 'python' / 'rpu_backend' / 'adapters' / 'qwen3_vl' / 'text.py').read_text()
    registrations = (ROOT / 'src' / 'core' / 'rpu_dispatch_registrations.inc').read_text()
    setter = model[model.index('void set_linear_acc32(bool enabled)'):]
    setter = setter[:setter.index('CausalDecoderModel() = default')]
    assert 'get_last_resolved_chunk_size() == 0' in setter
    assert 'linear_acc32_ = enabled;' in setter
    assert 'invalidate_model_state();' in setter
    assert 'bool linear_acc32_ = false;' in model
    assert model.count('/*force_acc32=*/linear_acc32_') == 10
    assert model.count('/*acc32=*/linear_acc32_') == 2
    vision_setter = vision[vision.index('void set_linear_acc32(bool enabled)'):]
    vision_setter = vision_setter[:vision_setter.index('int64_t resolved_chunk_size()')]
    assert 'get_last_resolved_chunk_size() == 0' in vision_setter
    assert 'linear_acc32_ = enabled;' in vision_setter
    assert 'invalidate_model_state();' in vision_setter
    assert 'bool linear_acc32_ = false;' in vision
    assert vision.count('/*force_acc32=*/linear_acc32_') == 8
    assert 'qwen3vl_vision_set_linear_acc32' in adapter
    assert re.search('causal_decoder_set_linear_acc32\\(\\s*handle,', adapter)
    assert 'RPU_LINEAR_ACC32' not in adapter
    assert 'causal_decoder_set_linear_acc32' not in shared
    assert 'm.def("causal_decoder_set_linear_acc32"' in registrations

def test_qwen35_vision_binds_g05_acc32_before_weights():
    model = (ROOT / 'src' / 'fused' / 'rpu_qwen3_5_vision_model.cpp').read_text()
    header = (ROOT / 'src' / 'fused' / 'rpu_qwen3_5_vision_model.h').read_text()
    adapter = (ROOT / 'python' / 'rpu_backend' / 'adapters' / 'qwen3_5' / 'vision.py').read_text()
    g05 = (ROOT / 'python' / 'rpu_backend' / 'adapters' / 'g05' / 'runtime.py').read_text()
    assert 'void set_linear_acc32(bool enabled);' in header
    assert 'bool linear_acc32_ = false;' in header
    setter = model[model.index('void Qwen3_5VisionModel::set_linear_acc32('):]
    setter = setter[:setter.index('void Qwen3_5VisionModel::set_weights(')]
    assert 'layer_weights_.empty() && num_layers() == 0' in setter
    assert 'linear_acc32_ = enabled;' in setter
    assert 'invalidate_model_state();' in setter
    assert model.count('/*force_acc32=*/linear_acc32_') == 18
    assert 'manifest.linear_accumulation = linear_acc32_' in model
    install = adapter[adapter.index('def _install_qwen3_5_vision_for_rpu_impl('):]
    create = install.index('torch.ops.rpu.qwen3_5_vision_create()')
    configure = install.index('set_linear_acc32(handle, linear_acc32)')
    weights = install.index('torch.ops.rpu.qwen3_5_vision_set_weights(')
    assert create < configure < weights
    assert 'linear_acc32, ultra_erf = _qwen35_vision_precision(' in install
    assert 'certified_g05=_allow_certified_g05_k1' in install
    assert '_rpu_vision_linear_acc32 = linear_acc32' in install
    assert '_allow_certified_g05_k1=rtc_release_fm_only' in g05
