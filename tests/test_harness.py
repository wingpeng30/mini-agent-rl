import json
import sys

import pytest

from mini_agent_rl.training.harness import inspect, preflight, run


def write_config(tmp_path, run_name="run", data="data.jsonl"):
    (tmp_path / data).write_text("{}\n", encoding="utf-8")
    config = tmp_path / "experiment.toml"
    config.write_text(
        "[experiment]\nname='unit'\nrun_dir='" + run_name + "'\ntimeout_seconds=10\n"
        "[command]\nargv=['" + sys.executable.replace("\\", "\\\\") + "','-c','print(123)']\n"
        "[inputs]\ndata='" + data + "'\n",
        encoding="utf-8",
    )
    return config


def test_preflight_rejects_final_test_and_existing_directory(tmp_path):
    config = write_config(tmp_path, data="final-test.jsonl")
    assert not preflight(config)["passed"]
    config = write_config(tmp_path, run_name="exists")
    (tmp_path / "exists").mkdir()
    assert not preflight(config)["passed"]


def test_supervised_run_writes_independent_manifest_and_status(tmp_path):
    config = write_config(tmp_path)
    assert preflight(config)["passed"]
    result = run(config)
    assert result["status"] == "COMPLETED"
    state = inspect(tmp_path / "run")
    assert state["manifest"]["version"] == "v0.8.3"
    assert state["status"]["status"] == "COMPLETED"


def test_preflight_rejects_expired_shared_deadline(tmp_path):
    config = write_config(tmp_path)
    text = config.read_text(encoding="utf-8").replace("timeout_seconds=10", "timeout_seconds=10\ndeadline_unix=1")
    config.write_text(text, encoding="utf-8")
    report = preflight(config)
    assert not report["passed"]
    assert any("截止时间" in item for item in report["failures"])
