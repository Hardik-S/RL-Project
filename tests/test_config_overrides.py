from src.algos.common import apply_overrides


def test_apply_overrides_recomputes_rollout_batch_size() -> None:
    config = {
        "algo": {"actor_lr": 3e-4},
        "env": {"total_timesteps": 100000},
        "evaluation": {"checkpoint_every_env_steps": 10000, "episodes": 10},
        "environment": {"n_envs": 8, "steps_per_env": 256, "rollout_batch_size": 2048},
    }
    updated = apply_overrides(
        config,
        {
            "actor_lr": 1e-4,
            "n_envs": 4,
            "steps_per_env": 128,
            "eval_every": 5000,
        },
    )
    assert updated["algo"]["actor_lr"] == 1e-4
    assert updated["environment"]["n_envs"] == 4
    assert updated["environment"]["steps_per_env"] == 128
    assert updated["environment"]["rollout_batch_size"] == 512
    assert updated["evaluation"]["checkpoint_every_env_steps"] == 5000
