"""Package only reviewed derived states and the conditional-stage runner."""
import argparse
from pathlib import Path
import tarfile
from mini_agent_rl.retrieval.local import ROOT,sha
from run_hybrid_adaptation import write

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);folder=parser.parse_args().folder
    files={}
    for name in ('approved-training-pairs.jsonl','training-pairs-freeze.json','training-pair-preparation.json'):
        files['experiment/'+name]=folder/name
    files['code/scripts/continue_hybrid_adaptation.py']=ROOT/'scripts/continue_hybrid_adaptation.py'
    for p in (folder/'training-index').iterdir():
        if p.is_file():files['experiment/training-index/'+p.name]=p
    write(folder/'training-stage-manifest.json',dict(files={name:sha(p) for name,p in files.items()},
        training_input_sha256=sha(folder/'training-inputs.jsonl'),gpu_seconds_cap=10800,
        optimizer=dict(lr=5e-6,betas=[.9,.999],eps=1e-8,weight_decay=0,clip=1),
        arms=['A','B'],seed=42,updates_per_arm=128,training_conditional=True))
    files['experiment/training-stage-manifest.json']=folder/'training-stage-manifest.json'
    with tarfile.open(folder/'training-stage.tar.gz','w:gz') as archive:
        for name,p in files.items():archive.add(p,arcname=name)
    print(dict(bytes=(folder/'training-stage.tar.gz').stat().st_size,sha256=sha(folder/'training-stage.tar.gz')))
