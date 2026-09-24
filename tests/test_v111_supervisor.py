"""Supervisor argument checks are platform-independent and launch no processes."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

SPEC = importlib.util.spec_from_file_location('v111_supervisor', Path(__file__).parents[1] / 'scripts/supervise_v111.py')
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


def test_command_preserves_tokens():
    args = supervisor.parse_args(['--timeout-seconds', '12', '--log', 'a.log', '--report', 'a.json',
                                  '--', 'python', 'script.py', '--limit', '20'])
    assert args.timeout_seconds == 12
    assert args.command == ['python', 'script.py', '--limit', '20']


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf'])
def test_invalid_timeout(value):
    with pytest.raises(SystemExit):
        supervisor.parse_args(['--timeout-seconds', value, '--log', 'a', '--report', 'b', '--', 'python'])


def test_requires_command_and_distinct_paths():
    with pytest.raises(SystemExit):
        supervisor.parse_args(['--log', 'a', '--report', 'b'])
    with pytest.raises(SystemExit):
        supervisor.parse_args(['--log', 'a', '--report', 'a', '--', 'python'])


def test_gpu_gate_tracks_peak_and_rejects_disappearance_and_excess():
    result = {'gpu_observed': False, 'device_peak_gib': 0.0}
    supervisor.check_gpu_sample(result, None, 21)
    supervisor.check_gpu_sample(result, 20, 21)
    supervisor.check_gpu_sample(result, 19, 21)
    assert result == {'gpu_observed': True, 'device_peak_gib': 20}
    with pytest.raises(RuntimeError, match='unverifiable'):
        supervisor.check_gpu_sample(result, None, 21)
    with pytest.raises(RuntimeError, match='exceeded'):
        supervisor.check_gpu_sample(result, 21.1, 21)
    assert result['device_peak_gib'] == 21.1


@pytest.mark.skipif(os.name != 'posix', reason='POSIX process groups required')
def test_posix_normal_child_success(tmp_path):
    report = tmp_path / 'report.json'
    log = tmp_path / 'child.log'
    result = subprocess.run([sys.executable, str(Path(supervisor.__file__)),
        '--timeout-seconds', '5', '--log', str(log), '--report', str(report), '--',
        sys.executable, '-c', "print('supervised-child-ok', flush=True)"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=12)
    assert result.returncode == 0, result.stderr
    payload = json.loads(report.read_text())
    assert payload['status'] == 'completed' and payload['child_returncode'] == 0
    assert 'supervised-child-ok' in log.read_text()


def _live_non_zombie(pid):
    # Linux kill(pid, 0) also succeeds for an already-killed orphan zombie.
    stat = Path(f'/proc/{pid}/stat')
    if stat.exists():
        try:
            return stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
        except FileNotFoundError:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(os.name != 'posix', reason='POSIX process groups required')
def test_posix_timeout_kills_child_and_grandchild(tmp_path):
    report = tmp_path / 'report.json'
    log = tmp_path / 'child.log'
    pid_file = tmp_path / 'grandchild.pid'
    # Grandchild deliberately ignores TERM, exercising the KILL escalation.
    grandchild_code = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    child_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild_code!r}]); time.sleep(60)"
    )
    command = [sys.executable, str(Path(supervisor.__file__)), '--timeout-seconds', '1',
               '--log', str(log), '--report', str(report), '--', sys.executable, '-c', child_code]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=14)
        assert process.returncode == 124, (stdout, stderr)
        payload = json.loads(report.read_text())
        assert payload['status'] == 'timeout'
        assert pid_file.exists(), 'grandchild must have started before timeout'
        grandchild_pid = int(pid_file.read_text())
        deadline = time.monotonic() + 0.5
        while _live_non_zombie(grandchild_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _live_non_zombie(payload['pid'])
        assert not _live_non_zombie(grandchild_pid)
    finally:
        # Keep a failing test from leaving its synthetic sleepers behind.
        if process.poll() is None:
            process.terminate()
        if report.exists():
            payload = json.loads(report.read_text())
            if 'pgid' in payload:
                try:
                    os.killpg(payload['pgid'], 9)
                except ProcessLookupError:
                    pass
        if process.poll() is None:
            process.kill()
        process.wait(timeout=1)
