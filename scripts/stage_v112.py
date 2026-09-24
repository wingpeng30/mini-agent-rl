"""Freeze approved training inputs and a read-only ledger receipt for AutoDL."""
from __future__ import annotations
import hashlib,json,sqlite3,tarfile
from pathlib import Path
from build_v112 import ROOT,OUT,read
from train_v112 import check_budget
from mini_agent_rl.evidence_sft_v112 import validate_dataset

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    gate_path=ROOT/'reports/v112-data-gate.json';gate=json.loads(gate_path.read_text(encoding='utf-8'))
    if not gate.get('approved'):raise RuntimeError('data_not_qualified_no_staging')
    check_budget(gate,gate_path)
    rows={name:read(OUT/(name+'.jsonl')) for name in ('train','validation')}
    if not validate_dataset(rows['train'],rows['validation'])['passed']:raise RuntimeError('data_gate_recheck')
    for name in rows:
        if sha(OUT/(name+'.jsonl'))!=gate['artifacts'][name]['sha256']:raise RuntimeError('data_hash_changed')
    snapshot=ROOT/'backups/v112-api-pretrain.sqlite'
    archive=ROOT/'backups/v112-training-inputs.tar.gz'
    frozen=ROOT/'reports/v112-frozen-data-gate.json'
    if any(p.exists() for p in (snapshot,archive,frozen)):raise FileExistsError('refuse_to_replace_frozen_training_stage')
    with sqlite3.connect((ROOT/'runs/v110-api.sqlite').as_uri()+'?mode=ro',uri=True) as source,sqlite3.connect(snapshot) as target:
        if source.execute("select count(*) from calls where status!='complete'").fetchone()[0]:raise RuntimeError('pending_api')
        source.backup(target)
    gate['ledger_snapshot']={'path':snapshot.relative_to(ROOT).as_posix(),'sha256':sha(snapshot),'purpose':'read-only settled verification snapshot; authoritative API ledger remains local'}
    frozen.write_text(json.dumps(gate,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    names=['scripts/train_v112.py','scripts/evaluate_v112.py','scripts/build_v112.py','scripts/run_v111_baseline.py',
      'src/mini_agent_rl/evidence_sft_v112.py','configs/v112-budget-policy.json','reports/v112-frozen-data-gate.json',
      'backups/v112-api-pretrain.sqlite','data/evidence-sft-v112/train.jsonl','data/evidence-sft-v112/validation.jsonl',
      'data/evidence-sft-v112/manifest.json','data/evidence-sft-v112/train-corpus.json']
    hashes={name:sha(ROOT/name) for name in names}
    manifest=ROOT/'reports/v112-training-transfer.json';manifest.write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf-8')
    with tarfile.open(archive,'w:gz') as tar:
        for name in names:tar.add(ROOT/name,arcname=name,recursive=False)
        tar.add(manifest,arcname='reports/v112-training-transfer.json',recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        if any(hashlib.sha256(tar.extractfile(n).read()).hexdigest()!=h for n,h in hashes.items()):raise RuntimeError('archive_verification_failed')
    print(json.dumps({'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(hashes)}))
if __name__=='__main__':main()
