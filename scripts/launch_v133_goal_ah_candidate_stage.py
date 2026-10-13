"""Four frozen real-search evaluations after baseline protection is frozen."""
import json,subprocess
from pathlib import Path
import run_v133_goal_ah_entity_relation as ah

def main():
    cfg=ah.approved_config()
    if ah.read(ah.OUT/'training-stage.json')['status']!='completed':raise RuntimeError('Training unfinished')
    freeze=ah.read(ah.OUT/'protection-freeze.json')
    if not freeze['frozen_before_candidate_evaluation'] or freeze['baseline_cases']!=272:raise RuntimeError('Protection not frozen')
    for name,digest in freeze['files'].items():
        if ah.sha(ah.OUT/name)!=digest:raise RuntimeError('Protection source changed')
    for label in ['baseline','42-A','43-A','42-B','43-B']:
        if len(ah.rows(ah.OUT/f'mechanism-{label}.jsonl'))!=16:raise RuntimeError('Mechanism stage unfinished')
    pid=ah.read(ah.OUT/'launch-mechanism-stage.json')['pid'];stat=Path(f'/proc/{pid}/stat')
    if stat.exists() and stat.read_text().split()[2]!='Z':raise RuntimeError('GPU mechanism process still alive')
    stage=ah.OUT/'candidate-evaluation-stage.json'
    ah.write_once(stage,dict(status='running',models=['42-A','43-A','42-B','43-B'],trajectories=1088,
        protection_sha256=ah.sha(ah.OUT/'protection-freeze.json'),automatic_retry=False))
    for label in ['42-A','43-A','42-B','43-B']:
        if (ah.OUT/'experiment-failure.json').exists():raise RuntimeError('Failure latch: stop queue')
        with (ah.OUT/f'gpu-evaluate-{label}.log').open('xb') as log:
            r=subprocess.run([str(ah.ROOT/'.venv/bin/python'),'-u',str(ah.ROOT/'scripts/run_v133_goal_ah_entity_relation.py'),
                'evaluate','--model',label],cwd=ah.ROOT,stdout=log,stderr=subprocess.STDOUT)
        ah.write_once(ah.OUT/f'evaluation-exit-{label}.json',dict(returncode=r.returncode))
        if r.returncode:
            stage.write_text(json.dumps(dict(status='stopped_failure',failed_model=label,automatic_retry=False))+'\n')
            return
    stage.write_text(json.dumps(dict(status='completed',models=4,trajectories=1088,automatic_retry=False))+'\n')

if __name__=='__main__':main()
