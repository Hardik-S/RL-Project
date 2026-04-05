from src.envs.make_env import ENV_ID_BY_KEY


def test_frozen_environment_keys_match_protocol() -> None:
    assert ENV_ID_BY_KEY == {
        "pendulum_v1": "Pendulum-v1",
        "hopper_v4": "Hopper-v4",
        "walker2d_v4": "Walker2d-v4",
        "halfcheetah_v4": "HalfCheetah-v4",
    }
