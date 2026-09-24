"""Scoped v111 evidence archive, with a read-only backup of the shared ledger."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import tarfile

PATTERNS = (
    'scripts/*v111*.py', 'tests/*v111*.py', 'src/mini_agent_rl/hybrid_v111.py',
    'src/mini_agent_rl/hybrid_v110.py', 'src/mini_agent_rl/hybrid_answer.py',
    'src/mini_agent_rl/answer_scoring.py', 'src/mini_agent_rl/tools/search.py',
    'scripts/run_v110.py', 'scripts/build_v110_inputs.py', 'scripts/prepare_v110_holdout.py',
    'scripts/prepare_v110_train_corpus.py', 'scripts/audit_v110_legacy.py',
    'configs/v111*.json', 'configs/v110-hybrid.json', 'docs/v111*.md',
    'docs/v110-preregistration.md', 'docs/v110-accuracy-results.md', 'reports/v111*', 'backups/v111*',
    'backups/v110-train-inputs.jsonl', 'backups/v110-train-inputs.jsonl.manifest.json',
    'reports/v110*audit*', 'data/hybrid-v110/*', 'data/hotpot-v110-independent/*',
    'data/hotpot-source-v110/*provenance*.json',
)
MAX_BYTES = 1_000_000_000
MANIFEST = 'reports/v111-artifact-manifest.json'
SUMS = 'reports/v111-SHA256SUMS'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root, output):
    files = set()
    for pattern in PATTERNS:
        for path in root.glob(pattern):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root).as_posix()
            name = path.name.lower()
            if path.resolve() == output or relative in {MANIFEST, SUMS}:
                continue
            if name.endswith(('.tar.gz', '.tgz', '.tar', '.zip', '-wal', '-shm', '.tmp')):
                continue
            if name.startswith('.env') or any(word in name for word in ('secret', 'credential', 'api-key', 'apikey')):
                continue
            if not path.resolve().is_relative_to(root):
                raise ValueError('archive source escapes workspace')
            files.add(path)
    return files


def archive(root, output):
    root = Path(root).resolve()
    output = Path(output)
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    if not output.is_relative_to(root / 'backups') or not output.name.endswith('.tar.gz'):
        raise ValueError('archive output must be a tar.gz inside workspace backups')
    files = collect(root, output)
    ledger = root / 'runs/v110-api.sqlite'
    destination = root / 'backups/v111-api.sqlite'
    files.discard(destination)
    estimated = sum(p.stat().st_size for p in files)
    if ledger.exists():
        estimated += ledger.stat().st_size
    if estimated > MAX_BYTES:
        raise RuntimeError('archive exceeds 1 GB limit')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if ledger.exists():
        temporary = destination.with_suffix('.sqlite.tmp')
        # Read-only source plus SQLite backup includes committed WAL transactions.
        source = sqlite3.connect(ledger.as_uri() + '?mode=ro', uri=True)
        target = sqlite3.connect(temporary)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        temporary.replace(destination)
        files.add(destination)
    entries = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                'sha256': sha256(p)} for p in sorted(files)]
    total = sum(entry['bytes'] for entry in entries)
    manifest_text = json.dumps({'files': entries, 'total_bytes': total}, indent=2) + '\n'
    sums_text = ''.join(entry['sha256'] + '  ' + entry['path'] + '\n' for entry in entries)
    if total + len(manifest_text.encode()) + len(sums_text.encode()) > MAX_BYTES:
        raise RuntimeError('archive exceeds 1 GB limit')
    for relative, text in ((MANIFEST, manifest_text), (SUMS, sums_text)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8', newline='\n')
        files.add(path)
    temporary_archive = output.with_name(output.name + '.tmp')
    with tarfile.open(temporary_archive, 'w:gz') as tar:
        for path in sorted(files):
            tar.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    temporary_archive.replace(output)
    return {'files': len(files), 'uncompressed_bytes': total, 'archive_bytes': output.stat().st_size,
            'archive_sha256': sha256(output)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--output', default='backups/v111-artifacts.tar.gz')
    args = parser.parse_args()
    print(json.dumps(archive(args.root, args.output)))


if __name__ == '__main__':
    main()
