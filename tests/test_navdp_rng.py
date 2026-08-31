from __future__ import annotations

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from rpu_backend.adapters.navdp.runtime import (
    ACTION_DIM,
    DDPM_STEPS,
    NavdpRuntime,
    PREDICT_SIZE,
    _initial_action_noise,
)


def _draw(generator: torch.Generator, sample_num: int = 2) -> torch.Tensor:
    return torch.randn(
        sample_num, PREDICT_SIZE, ACTION_DIM, generator=generator
    )


def test_navdp_default_init_and_steps_share_one_continuous_rng_stream() -> None:
    initial, generator = _initial_action_noise(2, 17, None)
    first_step_noise = _draw(generator)

    with torch.random.fork_rng():
        torch.manual_seed(17)
        assert torch.equal(initial, torch.randn_like(initial))
        assert torch.equal(first_step_noise, torch.randn_like(first_step_noise))


def test_navdp_explicit_init_keeps_step_stream_at_seed_start() -> None:
    supplied = torch.full((2, PREDICT_SIZE, ACTION_DIM), 0.25)
    initial, generator = _initial_action_noise(2, 17, supplied)

    assert torch.equal(initial, supplied)
    assert initial.data_ptr() != supplied.data_ptr()
    reference = torch.Generator().manual_seed(17)
    assert torch.equal(_draw(generator), _draw(reference))


def test_navdp_host_loop_continues_the_init_rng_stream(monkeypatch) -> None:
    monkeypatch.delenv("RPU_NAVDP_DENOISE_UNROLL", raising=False)
    runtime = NavdpRuntime.__new__(NavdpRuntime)
    runtime.predict_noise = lambda action, *_: torch.zeros_like(action)
    goal = torch.zeros(1, 1, 384)
    rgbd = torch.zeros(1, 32, 384)

    actual = runtime.predict_action(goal, rgbd, sample_num=2, seed=17)

    scheduler = DDPMScheduler(
        num_train_timesteps=DDPM_STEPS,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(DDPM_STEPS)
    generator = torch.Generator().manual_seed(17)
    expected = _draw(generator)
    for timestep in scheduler.timesteps:
        expected = scheduler.step(
            model_output=torch.zeros_like(expected),
            timestep=timestep,
            sample=expected,
            generator=generator,
        ).prev_sample

    assert torch.equal(actual, expected)
