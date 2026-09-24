"""POSIX process-group supervisor for the frozen independent baseline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def positive_seconds(value):
    import math
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('timeout must be finite and positive')
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout-seconds', type=positive_seconds, default=3600)
    parser.add_argument('--gpu-limit-gib', type=positive_seconds)
    parser.add_argument('--log', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == '--':
        args.command = args.command[1:]
    if not args.command:
        parser.error('a command after -- is required')
    if Path(args.log).resolve() == Path(args.report).resolve():
        parser.error('log and report must use different paths')
    return args


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def cleanup(process):
    signal_group(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader, including when descendants remain.
        if not group_exists(process.pid):
            break
        time.sleep(0.1)
    if group_exists(process.pid):
        signal_group(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


def gpu_sample(pid):
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                   '--format=csv,noheader,nounits'], text=True, timeout=5)
    values = [float(line.split(',')[1]) / 1024 for line in raw.splitlines()
              if line.split(',')[0].strip() == str(pid)]
    return max(values) if values else None


def check_gpu_sample(result, sample, limit):
    import math
    if sample is None:
        if result['gpu_observed']:
            raise RuntimeError('gpu_memory_unverifiable_after_observation')
        return
    if not math.isfinite(sample) or sample < 0:
        raise RuntimeError('gpu_memory_invalid_sample')
    result['gpu_observed'] = True
    result['device_peak_gib'] = max(result['device_peak_gib'], sample)
    if sample > limit:
        raise RuntimeError('gpu_memory_limit_exceeded')


def supervise(args):
    if os.name != 'posix':
        raise RuntimeError('this supervisor requires POSIX process groups')
    log = Path(args.log)
    report = Path(args.report)
    log.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    received = []
    previous = {}
    process = None
    started = time.monotonic()
    result = {'command': args.command, 'timeout_seconds': args.timeout_seconds,
              'log': str(log), 'status': 'starting', 'gpu_limit_gib': args.gpu_limit_gib,
              'gpu_observed': False, 'device_peak_gib': 0.0}

    def save():
        temporary = report.with_name(report.name + '.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
        temporary.replace(report)

    def interrupted(signum, _frame):
        received.append(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        save()
        with log.open('ab', buffering=0) as stream:
            process = subprocess.Popen(args.command, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            result.update(status='running', pid=process.pid, pgid=process.pid)
            save()
            next_gpu_poll = time.monotonic()
            while process.poll() is None:
                if received:
                    result.update(status='interrupted', signal=received[0], exit_code=128 + received[0])
                    cleanup(process)
                    break
                if time.monotonic() - started >= args.timeout_seconds:
                    result.update(status='timeout', exit_code=124)
                    cleanup(process)
                    break
                if args.gpu_limit_gib is not None and time.monotonic() >= next_gpu_poll:
                    try:
                        sample = gpu_sample(process.pid)
                        # Normal process exit releases its GPU context; poll again
                        # before classifying a missing PID as a monitoring failure.
                        if sample is None and process.poll() is not None:
                            continue
                        check_gpu_sample(result, sample, args.gpu_limit_gib)
                    except Exception as exc:
                        result.update(status='gpu_gate_failed', error=str(exc), exit_code=125)
                        cleanup(process)
                        break
                    next_gpu_poll = time.monotonic() + 1
                    save()
                time.sleep(0.2)
            else:
                result.update(status='completed' if process.returncode == 0 else 'failed',
                              exit_code=process.returncode if process.returncode >= 0 else 128 - process.returncode)
            if result['status'] == 'completed' and args.gpu_limit_gib is not None and not result['gpu_observed']:
                result.update(status='gpu_gate_failed', error='gpu_never_observed', exit_code=125)
            # A successful leader may leave descendants: do not detach them.
            if group_exists(process.pid):
                cleanup(process)
            result['child_returncode'] = process.returncode
    except BaseException as exc:
        result.update(status='supervisor_error', error=str(exc), exit_code=1)
        if process is not None:
            cleanup(process)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        save()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return result['exit_code']


def main():
    raise SystemExit(supervise(parse_args()))


if __name__ == '__main__':
    main()
