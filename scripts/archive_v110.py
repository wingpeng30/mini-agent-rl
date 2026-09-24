"""Create a bounded, key-free v110 artifact archive and SHA256 manifest."""
import argparse,hashlib,json,sqlite3,tarfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='backups/v110-artifacts.tar.gz');args=p.parse_args()
    root=Path('.').resolve();files=set()
    for pattern in ('scripts/*v110*.py','tests/*v110*.py','src/mini_agent_rl/hybrid_v110.py','src/mini_agent_rl/tools/search.py','configs/v110*.json','docs/v110*.md','docs/v109-hybrid-answer-results.md','reports/v110*','backups/v110*.jsonl','backups/v110*.json','data/hybrid-v110/*.json','data/hotpot-v110-independent/*','data/hotpot-source-v110/*provenance*.json'):
        files.update(p for p in root.glob(pattern) if p.is_file())
    ledger=root/'runs/v110-api.sqlite'
    if ledger.exists():
        dest=root/'backups/v110-api.sqlite';src=sqlite3.connect(f'file:{ledger.as_posix()}?mode=ro',uri=True);out=sqlite3.connect(dest);src.backup(out);out.close();src.close();files.add(dest)
    files.discard(root/'reports/v110-artifact-manifest.json');files.discard(root/'reports/v110-SHA256SUMS')
    entries=[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]
    total=sum(e['bytes'] for e in entries)
    if total>1_000_000_000:raise RuntimeError('archive exceeds approved automatic backup size')
    manifest=root/'reports/v110-artifact-manifest.json';manifest.write_text(json.dumps({'files':entries,'total_bytes':total},indent=2),encoding='utf-8');files.add(manifest)
    sums=root/'reports/v110-SHA256SUMS';sums.write_text(''.join(e['sha256']+'  '+e['path']+'\n' for e in entries),encoding='utf-8',newline='\n');files.add(sums)
    with tarfile.open(args.output,'w:gz') as tar:
        for path in sorted(files):tar.add(path,arcname=path.relative_to(root).as_posix())
    archive=Path(args.output);print(json.dumps({'files':len(files),'uncompressed_bytes':total,'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
