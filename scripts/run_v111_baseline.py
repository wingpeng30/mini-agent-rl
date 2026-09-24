"""Frozen independent100 baseline; launch inside an externally supervised process group."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from mini_agent_rl.evaluation import load_hotpot_environment
from mini_agent_rl.model.local import LocalModelConfig, TransformersModelClient
from mini_agent_rl.cli import default_reward
from mini_agent_rl.runtime import AgentRunner
from mini_agent_rl.storage import SQLiteStore
from mini_agent_rl.tools import ToolRegistry, SearchTool

ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG = dict(max_steps=3, max_concurrency=1, temperature=0, base_seed=42,
                  anti_repeat_prompt=False, enable_thinking=False, prompt_variant='direct-v090')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def model_config(model_path, adapter_path):
    return LocalModelConfig(model_id=str(model_path), adapter_path=str(adapter_path),
        device='cuda', dtype='bfloat16', load_in_4bit=False, max_input_tokens=2048,
        max_new_tokens=128, enable_thinking=False, timeout=600,
        gradient_checkpointing=False, observation_encoding='compact-v3',
        deterministic=True, local_files_only=True, trust_remote_code=False, logprob_mode='bf16')


def tree_fingerprint(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError('model_and_adapter_must_be_local_directories')
    files = sorted(p for p in path.rglob('*') if p.is_file() and '.cache' not in p.relative_to(path).parts)
    if not files:
        raise ValueError('empty_model_directory')
    return {p.relative_to(path).as_posix(): sha256(p) for p in files}


def fingerprint(env, config, expected_adapter_sha256):
    if sha256(Path(config.adapter_path) / 'adapter_model.safetensors') != expected_adapter_sha256:
        raise RuntimeError('frozen_v101_seed42_adapter_mismatch')
    if len(env.tasks) != 100 or len(set(env.task_ids)) != 100 or env.corpus_document_count != 986:
        raise RuntimeError('independent100_986_document_environment_required')
    code = sorted((ROOT / 'src/mini_agent_rl').rglob('*.py')) + [Path(__file__), ROOT / 'pyproject.toml']
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'pydantic'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'version': 'v111', 'environment': env.fingerprint('bm25-v2', 'compact-v3'),
            'model_config': config.model_dump(), 'runner_config': RUN_CONFIG,
            'model_files': tree_fingerprint(config.model_id),
            'adapter_files': tree_fingerprint(config.adapter_path),
            'frozen_v101_seed42_adapter_sha256': expected_adapter_sha256,
            'code': {p.relative_to(ROOT).as_posix(): sha256(p) for p in code}, 'packages': versions}


def validate_resume(existing, task_ids):
    if len({r.task_id for r in existing}) != len(existing):
        raise RuntimeError('duplicate_rollouts')
    if any(r.status.value not in {'SUCCEEDED', 'FAILED'} for r in existing):
        raise RuntimeError('incomplete_prior_rollout_requires_reconciliation')
    if any(r.task_id not in set(task_ids) for r in existing):
        raise RuntimeError('foreign_task_in_database')
    for rollout in existing:
        check_infrastructure_error(rollout)
    return {r.task_id for r in existing}


def check_infrastructure_error(rollout):
    error = rollout.error or ''
    # Local client converts ActionParseError into response.parse_error, and the
    # runner records invalid_action. Any model_error is an actual exception.
    if error or rollout.termination_reason not in {'answer', 'invalid_action', 'repeated_action', 'max_steps'}:
        raise RuntimeError('infrastructure_failure: ' + (error or str(rollout.termination_reason)))


def check_peak_memory(stats):
    for key in ('peak_allocated_gb', 'peak_reserved_gb'):
        value = stats.get(key)
        if value is None or not math.isfinite(float(value)) or float(value) > 21:
            raise RuntimeError('gpu_peak_memory_gate: ' + key)


def device_memory_gib():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True, timeout=20)
    memory = [float(line.split(',')[1]) / 1024 for line in raw.splitlines()
              if line.split(',')[0].strip() == str(os.getpid())]
    if not memory or max(memory) > 21:
        raise RuntimeError('device_memory_gate')
    return max(memory)


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'model-path', 'adapter-path', 'expected-adapter-sha256', 'db', 'report'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--limit', type=int, choices=(20, 100), default=100)
    args = parser.parse_args()
    env = load_hotpot_environment(args.data, task_limit=100, corpus_limit=100)
    config = model_config(args.model_path, args.adapter_path)
    fp = fingerprint(env, config, args.expected_adapter_sha256)
    manifest = Path(args.db + '.manifest.json')
    if Path(args.db).exists() and not manifest.exists():
        raise RuntimeError('unmanifested_db')
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != fp:
        raise RuntimeError('changed_environment')
    write(manifest, fp)
    store = SQLiteStore(args.db)
    start = time.perf_counter()
    try:
        done = validate_resume(store.list_rollouts(), env.task_ids)
        registry = ToolRegistry()
        registry.register(SearchTool(env.corpus, 'bm25-v2'))
        client = TransformersModelClient(config)
        runner = AgentRunner(client, registry, default_reward(), store, environment_fingerprint=fp, **RUN_CONFIG)
        async def go():
            rows = []
            requested_ids = {task.id for task in env.tasks[:args.limit]}
            def report():
                write(args.report, {'version': 'v111', 'completed': len(done & requested_ids),
                    'total_database_completed': len(done), 'requested_count': args.limit,
                    'evaluation_denominator': args.limit, 'fingerprint': fp, 'new_rows': rows,
                    'elapsed_seconds': time.perf_counter() - start})
            report()
            for task in env.tasks[:args.limit]:
                if task.id in done:
                    continue
                # Outer supervisor must terminate the entire process group on timeout.
                if hasattr(signal, 'alarm'):
                    signal.alarm(660)
                try:
                    group = await runner.run_group(task, 1)
                finally:
                    if hasattr(signal, 'alarm'):
                        signal.alarm(0)
                rollout = store.get_rollout(group.rollout_ids[0])
                check_infrastructure_error(rollout)
                memory = device_memory_gib()
                stats = client.gpu_memory_stats()
                check_peak_memory(stats)
                done.add(task.id)  # Failed task outcomes remain in the denominator.
                rows.append({'task_id': task.id, 'rollout_id': rollout.id,
                             'termination': rollout.termination_reason,
                             'memory': stats, 'device_process_gib': memory})
                report()
                print(f'{len(done)}/{args.limit} {rollout.termination_reason}', flush=True)
        asyncio.run(go())
    finally:
        store.close()


if __name__ == '__main__':
    main()
