"""CPU-only decode and snapshot of completed frozen evaluation splits.

Run remotely while fixed GPU queue proceeds. Does not generate or resume runs.
"""
import subprocess,tarfile,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/v133-goal-ah-entity-relation'
SPLITS={'goal_z_dev':64,'goal_y_regression':48,'aa_regression':64,
        'old_main':16,'old_regression':16,'ag_regression':64}
def main():
    files=[];counts={}
    for label in ['42-A','43-A','42-B','43-B']:
        for split,n in SPLITS.items():
            raw=OUT/f'eval-{split}-{label}.jsonl'
            if not raw.exists():continue
            count=sum(1 for _ in raw.open());counts[f'{label}:{split}']=count
            if count!=n:continue
            decoded=OUT/f'actual-inputs-{split}-{label}.jsonl'
            if not decoded.exists():
                subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/review_v133_goal_ah_actual_inputs.py'),'--model',label,'--split',split],cwd=ROOT,check=True)
            files.extend([raw,decoded])
    for name in ['candidate-evaluation-stage.json','launch-candidate-stage.json']:
        p=OUT/name
        if p.exists():files.append(p)
    for p in OUT.glob('candidate-exit-*.json'):files.append(p)
    # Snapshot archive may be replaced; immutable JSONL outputs never are.
    with tarfile.open(OUT/'candidate-completed-snapshot.tar.gz','w:gz') as t:
        for p in files:t.add(p,arcname=p.name)
    print(json.dumps(dict(completed_splits=sum(v==SPLITS[k.split(':')[1]] for k,v in counts.items()),counts=counts,
        failure=(OUT/'experiment-failure.json').exists(),stage=json.loads((OUT/'candidate-evaluation-stage.json').read_text()))))
if __name__=='__main__':main()
