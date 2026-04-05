"""Environment construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.envs.normalization import ObservationNormalizationConfig, build_vecnormalize_kwargs, sync_obs_rms
from src.envs.wrappers import WrapperConfig

ENV_ID_BY_KEY = {
    "pendulum_v1": "Pendulum-v1",
    "hopper_v4": "Hopper-v4",
    "walker2d_v4": "Walker2d-v4",
    "halfcheetah_v4": "HalfCheetah-v4",
}


@dataclass(frozen=True)
class EnvFactoryConfig:
    env_key: str
    n_envs: int
    seed: int
    observation_normalization: ObservationNormalizationConfig
    wrappers: WrapperConfig

    @property
    def env_id(self) -> str:
        return ENV_ID_BY_KEY[self.env_key]


@dataclass(frozen=True)
class EnvBundle:
    train_env: Any
    eval_env: Any


def make_single_env(env_id: str, seed: int) -> Any:
    import gymnasium as gym
    from stable_baselines3.common.monitor import Monitor

    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return Monitor(env)


def make_env_fns(config: EnvFactoryConfig) -> list[Callable[[], Any]]:
    return [
        (lambda env_id=config.env_id, seed=config.seed + offset: make_single_env(env_id, seed))
        for offset in range(config.n_envs)
    ]


def make_eval_env(config: EnvFactoryConfig) -> Any:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    vec_env = DummyVecEnv([lambda: make_single_env(config.env_id, config.seed + 10_000)])
    vec_env = VecMonitor(vec_env)
    eval_env = VecNormalize(vec_env, **build_vecnormalize_kwargs(config.observation_normalization))
    eval_env.training = False
    eval_env.norm_reward = False
    return eval_env


def make_training_env(config: EnvFactoryConfig) -> Any:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

    vec_env = DummyVecEnv(make_env_fns(config))
    vec_env = VecMonitor(vec_env)
    return VecNormalize(vec_env, **build_vecnormalize_kwargs(config.observation_normalization))


def make_env_bundle(config: dict[str, Any]) -> EnvBundle:
    obs_norm = ObservationNormalizationConfig(
        enabled=config["environment"]["observation_normalization"],
        clip_obs=config["environment"]["observation_clip"],
        normalize_reward=config["environment"]["reward_normalization"],
    )
    factory_config = EnvFactoryConfig(
        env_key=config["env"]["key"],
        n_envs=config["environment"]["n_envs"],
        seed=config["run"]["seed"],
        observation_normalization=obs_norm,
        wrappers=WrapperConfig(
            bootstrap_on_time_limit_truncation=config["environment"]["bootstrap_on_time_limit_truncation"],
            action_squash=config["environment"]["action_squash"],
        ),
    )
    train_env = make_training_env(factory_config)
    eval_env = make_eval_env(factory_config)
    sync_obs_rms(train_env, eval_env)
    return EnvBundle(train_env=train_env, eval_env=eval_env)
