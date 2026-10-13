"""Package frozen offline assets for an isolated remote experiment."""
import argparse
import json
from pathlib import Path
import tarfile
from mini_agent_rl.retrieval.local import ROOT, sha

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    folder=args.folder;cfg=json.loads((folder/'retrieval-source.json').read_text(encoding='utf-8'))
    destination='/root/autodl-tmp/hybrid-adaptation-20261006'
    remote=dict(cfg,corpus=destination+'/experiment/corpus-dev.json',model_path=destination+'/minilm',
        index_dir=destination+'/index',run_dir=destination+'/experiment')
    (folder/'retrieval-remote.json').write_text(json.dumps(remote,indent=2),encoding='utf-8')
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    manifest['frozen_files']={p.name:sha(p) for p in folder.iterdir() if p.suffix in ('.json','.jsonl') and p.name!='manifest.json'}
    tracked=('scripts/run_hybrid_adaptation.py','scripts/run_v133_goal_q_search.py','src/mini_agent_rl/model/local.py','src/mini_agent_rl/retrieval/local.py')
    manifest['code_files']={name:sha(ROOT/name) for name in tracked}
    manifest['semantic_review_stage']='64-case body/action first-pass complete; causal review pending'
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with tarfile.open(folder/'bundle.tar.gz','w:gz') as archive:
        for section in ('src','scripts'):
            for path in (ROOT/section).rglob('*.py'):
                if '__pycache__' not in path.parts:archive.add(path,arcname='code/'+str(path.relative_to(ROOT)).replace('\\','/'))
        for path in folder.iterdir():
            if path.suffix in ('.json','.jsonl'):archive.add(path,arcname='experiment/'+path.name)
        for path in (folder/'goal-z-backup').iterdir():
            archive.add(path,arcname='experiment/goal-z-backup/'+path.name)
        for key,section in (('model_path','minilm'),('index_dir','index')):
            for path in Path(cfg[key]).rglob('*'):
                if path.is_file():archive.add(path,arcname=section+'/'+str(path.relative_to(cfg[key])).replace('\\','/'))
    print(json.dumps({'archive':str(folder/'bundle.tar.gz'),'bytes':(folder/'bundle.tar.gz').stat().st_size,'sha256':sha(folder/'bundle.tar.gz')}))
