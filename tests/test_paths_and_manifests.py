from pathlib import Path

from src.utils.manifests import manifest_paths, stable_config_hash
from src.utils.paths import run_dir, suite_manifest_dir, variant_tag


def test_suite_run_dir_is_readable_and_tagged() -> None:
    path = run_dir("ppo_kl", "hopper_v4", 2, suite_name="sweep_hopper_v4", run_tag=variant_tag("target_kl", 0.02))
    assert path.parts[-5:] == ("sweep_hopper_v4", "ppo_kl", "hopper_v4", "seed_2", "target_kl_0p02")


def test_variant_tag_defaults_and_float_format() -> None:
    assert variant_tag() == "default"
    assert variant_tag("actor_lr", 0.0003) == "actor_lr_0p0003"


def test_manifest_paths_are_suite_scoped() -> None:
    paths = manifest_paths("main_benchmark")
    assert paths["root"] == suite_manifest_dir("main_benchmark")
    assert paths["completed"].name == "completed_runs.jsonl"
    assert paths["failed"].name == "failed_runs.jsonl"


def test_stable_config_hash_is_deterministic() -> None:
    payload = {"b": 2, "a": 1}
    assert stable_config_hash(payload) == stable_config_hash({"a": 1, "b": 2})
