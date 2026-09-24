"""Small synthetic archive checks; no experiment data or credentials accessed."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import tarfile

import pytest

SPEC = importlib.util.spec_from_file_location('v111_archive', Path(__file__).parents[1] / 'scripts/archive_v111.py')
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def test_scoped_archive_snapshot_and_lf_manifest(tmp_path):
    (tmp_path / 'backups').mkdir()
    (tmp_path / 'runs').mkdir()
    (tmp_path / 'backups/v111-input.json').write_text('{}')
    (tmp_path / 'backups/v111-old.tar.gz').write_bytes(b'not included')
    (tmp_path / 'backups/v111-secret.json').write_text('not included')
    with sqlite3.connect(tmp_path / 'runs/v110-api.sqlite') as db:
        db.execute('CREATE TABLE ledger (cost INTEGER)')
        db.execute('INSERT INTO ledger VALUES (3)')
    db.close()
    archive.archive(tmp_path, 'backups/v111-artifacts.tar.gz')
    with tarfile.open(tmp_path / 'backups/v111-artifacts.tar.gz') as tar:
        names = tar.getnames()
    assert 'backups/v111-api.sqlite' in names
    assert not any('secret' in name or name.endswith('.tar.gz') for name in names)
    manifest = json.loads((tmp_path / archive.MANIFEST).read_text())
    assert not any(e['path'] in {archive.MANIFEST, archive.SUMS} for e in manifest['files'])
    assert b'\r' not in (tmp_path / archive.SUMS).read_bytes()
    with sqlite3.connect(tmp_path / 'backups/v111-api.sqlite') as db:
        assert db.execute('SELECT cost FROM ledger').fetchone() == (3,)
    db.close()
    # Re-running excludes the existing archive and generated manifests.
    archive.archive(tmp_path, 'backups/v111-artifacts.tar.gz')


def test_size_guard_and_output_scope(tmp_path, monkeypatch):
    (tmp_path / 'backups').mkdir()
    (tmp_path / 'backups/v111-input.json').write_text('{}')
    monkeypatch.setattr(archive, 'MAX_BYTES', 1)
    with pytest.raises(RuntimeError, match='1 GB'):
        archive.archive(tmp_path, 'backups/v111-artifacts.tar.gz')
    with pytest.raises(ValueError, match='inside workspace backups'):
        archive.archive(tmp_path, 'outside.tar.gz')
