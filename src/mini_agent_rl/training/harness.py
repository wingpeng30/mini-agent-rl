"""v0.8.3 实验 harness：冻结配置、监督子进程并保留可复现实验护照。"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FINAL_MARKERS = ("final-test", "hotpot-agent-v072")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(args: list[str]) -> str | None:
    try:
        # Git 差异可能包含中文；显式解码，避免 Windows 默认 GBK 破坏护照生成。
        return subprocess.check_output(["git", *args], text=True, encoding="utf-8", errors="surrogateescape", stderr=subprocess.DEVNULL, timeout=5).strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    run_dir: Path
    command: list[str]
    working_dir: Path
    timeout_seconds: int = 1800
    deadline_unix: float | None = None
    data_path: Path | None = None
    corpus_path: Path | None = None
    model_path: Path | None = None
    adapter_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        source = Path(path)
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
        experiment, command, inputs = raw.get("experiment", {}), raw.get("command", {}), raw.get("inputs", {})
        argv = command.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
            raise ValueError("TOML 必须包含 command.argv 字符串数组")
        root = source.parent
        resolve = lambda value: (root / value).resolve() if value else None
        return cls(name=str(experiment.get("name", source.stem)), run_dir=resolve(experiment.get("run_dir")) or (root / "runs" / source.stem),
                   command=argv, working_dir=resolve(experiment.get("working_dir")) or root,
                   timeout_seconds=int(experiment.get("timeout_seconds", 1800)), deadline_unix=float(experiment["deadline_unix"]) if experiment.get("deadline_unix") else None,
                   data_path=resolve(inputs.get("data")),
                   corpus_path=resolve(inputs.get("corpus")), model_path=resolve(inputs.get("model")), adapter_path=resolve(inputs.get("adapter")))


def preflight(config_path: str | Path) -> dict[str, Any]:
    """不创建任何文件的预检；封存最终集和既有运行目录都拒绝继续。"""
    config = ExperimentConfig.load(config_path)
    failures: list[str] = []
    for label, path in (("data", config.data_path), ("corpus", config.corpus_path), ("model", config.model_path), ("adapter", config.adapter_path)):
        if path is not None and not path.exists():
            failures.append(f"{label} 不存在：{path}")
    if not config.working_dir.is_dir():
        failures.append(f"working_dir 不存在：{config.working_dir}")
    if config.data_path and any(marker in str(config.data_path).lower() for marker in FINAL_MARKERS):
        failures.append("禁止将封存 final-test 用于 harness 实验")
    if config.run_dir.exists():
        failures.append(f"运行目录已存在，拒绝覆盖：{config.run_dir}")
    if config.timeout_seconds < 0:
        failures.append("timeout_seconds 必须非负；0 表示不限时")
    if config.deadline_unix is not None and config.deadline_unix <= time.time():
        failures.append("统一截止时间已经过去")
    return {"version": "v0.8.3", "config": str(Path(config_path).resolve()), "name": config.name,
            "run_dir": str(config.run_dir), "working_dir": str(config.working_dir), "command": config.command, "passed": not failures, "failures": failures}


def _manifest(config: ExperimentConfig, config_path: str | Path, status: str) -> dict[str, Any]:
    """运行护照只记录环境和文件哈希，不记录 API key、样本内容或模型权重。"""
    return {"version": "v0.8.3", "name": config.name, "status": status, "created_at": time.time(),
            "config_path": str(Path(config_path).resolve()), "config_sha256": _sha256(Path(config_path)),
            "command": config.command, "working_dir": str(config.working_dir), "timeout_seconds": config.timeout_seconds,
            "deadline_unix": config.deadline_unix,
            "inputs": {"data": str(config.data_path) if config.data_path else None, "data_sha256": _sha256(config.data_path) if config.data_path else None,
                       "corpus": str(config.corpus_path) if config.corpus_path else None, "corpus_sha256": _sha256(config.corpus_path) if config.corpus_path else None,
                       "model": str(config.model_path) if config.model_path else None, "adapter": str(config.adapter_path) if config.adapter_path else None},
            "code": {"commit": _git_value(["rev-parse", "HEAD"]), "dirty_diff_sha256": hashlib.sha256((_git_value(["diff", "--binary"]) or "").encode()).hexdigest()},
            "environment": {"platform": os.name, "python": os.sys.version, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}}


def run(config_path: str | Path) -> dict[str, Any]:
    """在独立目录监督用户冻结的命令；超时是唯一自动终止条件，不自动重试。"""
    check = preflight(config_path)
    if not check["passed"]:
        raise RuntimeError("预检失败：" + "；".join(check["failures"]))
    config = ExperimentConfig.load(config_path)
    config.run_dir.mkdir(parents=True)
    manifest_path = config.run_dir / "manifest.json"
    status_path = config.run_dir / "status.json"
    manifest = _manifest(config, config_path, "RUNNING")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.monotonic()
    with (config.run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (config.run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        process = subprocess.Popen(config.command, cwd=config.working_dir, stdout=stdout, stderr=stderr, text=True,
                                   start_new_session=os.name != "nt", creationflags=creationflags)
        while process.poll() is None:
            elapsed = time.monotonic() - started
            status_path.write_text(json.dumps({"status": "RUNNING", "pid": process.pid, "elapsed_seconds": elapsed}, ensure_ascii=False, indent=2), encoding="utf-8")
            deadline_reached = config.deadline_unix is not None and time.time() >= config.deadline_unix
            if (config.timeout_seconds and elapsed > config.timeout_seconds) or deadline_reached:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=15)
                else:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=15)
                    process.wait(timeout=15)
                result = {"status": "DEADLINE_REACHED" if deadline_reached else "TIMED_OUT", "exit_code": process.returncode, "elapsed_seconds": elapsed,
                          "process_group_cleaned": process.poll() is not None}
                status_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                return result
            time.sleep(1)
    result = {"status": "COMPLETED" if process.returncode == 0 else "FAILED", "exit_code": process.returncode,
              "elapsed_seconds": time.monotonic() - started}
    status_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def inspect(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    result: dict[str, Any] = {"run_dir": str(root.resolve())}
    for name in ("manifest.json", "status.json"):
        path = root / name
        result[name.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return result
