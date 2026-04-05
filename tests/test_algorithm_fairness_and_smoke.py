from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
gym = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")
DummyVecEnv = pytest.importorskip("stable_baselines3.common.vec_env").DummyVecEnv

from src.algos.common import TrainingJob, load_run_config
from src.algos.ppo_kl_runner import NON_OPERATIVE_CLIP_RANGE, build_model as build_ppo_kl_model
from src.algos.trpo_runner import _set_trpo_critic_lr, build_model as build_trpo_model
from src.metrics.stability import StabilityInputs, evaluate_unstable_update


def _job_for_algo(
    algorithm: str,
    tmp_path: Path,
    seed: int = 0,
    *,
    overrides: dict[str, float | int] | None = None,
) -> TrainingJob:
    default_overrides: dict[str, float | int] = {
        "n_envs": 1,
        "steps_per_env": 32,
        "eval_episodes": 1,
        "eval_every": 32,
        "total_timesteps": 64,
    }
    if overrides:
        default_overrides.update(overrides)
    config = load_run_config(algorithm, "pendulum_v1", seed, overrides=default_overrides)
    return TrainingJob(
        algorithm=algorithm,
        env_key="pendulum_v1",
        seed=seed,
        config=config,
        output_dir=tmp_path / f"{algorithm}_{seed}",
        device="cpu",
    )


def _make_env() -> DummyVecEnv:
    return DummyVecEnv([lambda: gym.make("Pendulum-v1")])


@pytest.mark.parametrize(
    ("algorithm", "builder"),
    [
        ("a2c", pytest.importorskip("src.algos.a2c_runner").build_model),
        ("ppo_clip", pytest.importorskip("src.algos.ppo_clip_runner").build_model),
        ("ppo_kl", build_ppo_kl_model),
        ("trpo", build_trpo_model),
    ],
)
def test_all_runners_keep_matched_actor_critic_architecture(
    algorithm: str,
    builder,
    tmp_path: Path,
) -> None:
    env = _make_env()
    try:
        model = builder(_job_for_algo(algorithm, tmp_path), env, "cpu")
        policy = model.policy

        assert len(policy.optimizer.param_groups) == 2
        actor_group_ids = {id(param) for param in policy.optimizer.param_groups[0]["params"]}
        critic_group_ids = {id(param) for param in policy.optimizer.param_groups[1]["params"]}
        assert actor_group_ids
        assert critic_group_ids
        assert actor_group_ids.isdisjoint(critic_group_ids)

        policy_hidden = [
            layer.out_features for layer in policy.mlp_extractor.policy_net if isinstance(layer, nn.Linear)
        ]
        value_hidden = [
            layer.out_features for layer in policy.mlp_extractor.value_net if isinstance(layer, nn.Linear)
        ]
        assert policy_hidden == [64, 64]
        assert value_hidden == [64, 64]
        assert policy.activation_fn is nn.Tanh
        assert policy.share_features_extractor is False
        assert float(policy.log_std.detach().mean().item()) == pytest.approx(-0.5)
    finally:
        env.close()


def test_ppo_kl_contract_and_smoke_update_logging(tmp_path: Path) -> None:
    job = _job_for_algo(
        "ppo_kl",
        tmp_path,
        overrides={
            "minibatch_size": 16,
            "max_policy_epochs": 2,
        },
    )
    env = _make_env()
    try:
        model = build_ppo_kl_model(job, env, "cpu")
        assert job.config["algo"]["ratio_clipping"] is False
        assert job.config["algo"]["early_stop_on_target_kl"] is True
        assert model.clip_range(1.0) == NON_OPERATIVE_CLIP_RANGE

        updates_before = int(model._n_updates)
        model.learn(total_timesteps=job.config["environment"]["rollout_batch_size"] * 2, progress_bar=False)

        assert int(model._n_updates) == updates_before + 2
        assert int(model.last_training_info["update_index"]) == int(model._n_updates)
        assert math.isnan(float(model.last_training_info["clip_fraction"]))
    finally:
        env.close()


def test_trpo_lr_update_only_touches_critic_group(tmp_path: Path) -> None:
    job = _job_for_algo("trpo", tmp_path, overrides={"critic_minibatch_size": 16})
    env = _make_env()
    try:
        model = build_trpo_model(job, env, "cpu")
        actor_before = float(model.policy.optimizer.param_groups[0]["lr"])
        _set_trpo_critic_lr(model.policy, 0.123)
        assert float(model.policy.optimizer.param_groups[0]["lr"]) == pytest.approx(actor_before)
        assert float(model.policy.optimizer.param_groups[1]["lr"]) == pytest.approx(0.123)
    finally:
        env.close()


def test_unstable_update_uses_configured_kl_budget_for_constrained_methods() -> None:
    constrained = evaluate_unstable_update(
        StabilityInputs(
            algorithm="ppo_kl",
            mean_kl_old_new=0.021,
            nominal_kl_budget=0.01,
            next_eval_return=90.0,
            trailing_eval_mean_prev3=100.0,
            max_eval_so_far=105.0,
            initial_eval_mean_first3=80.0,
        )
    )
    assert constrained.kl_condition is True
    assert constrained.unstable_update is True

    constrained_relaxed = evaluate_unstable_update(
        StabilityInputs(
            algorithm="trpo",
            mean_kl_old_new=0.03,
            nominal_kl_budget=0.02,
            next_eval_return=90.0,
            trailing_eval_mean_prev3=100.0,
            max_eval_so_far=105.0,
            initial_eval_mean_first3=80.0,
        )
    )
    assert constrained_relaxed.kl_condition is False
    assert constrained_relaxed.unstable_update is False

    unconstrained = evaluate_unstable_update(
        StabilityInputs(
            algorithm="a2c",
            mean_kl_old_new=0.06,
            nominal_kl_budget=0.5,
            next_eval_return=90.0,
            trailing_eval_mean_prev3=100.0,
            max_eval_so_far=105.0,
            initial_eval_mean_first3=80.0,
        )
    )
    assert unconstrained.kl_condition is True
    assert unconstrained.unstable_update is True
