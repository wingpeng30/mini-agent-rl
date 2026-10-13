"""Execute the four already approved training commands once, sequentially."""
import json,subprocess,time
from pathlib import Path
import run_v133_goal_ah_entity_relation as ah

def main():
    cfg=ah.approved_config()
    for spec in cfg['evaluation']:
        if len(ah.rows(ah.OUT/f"eval-{spec['split']}-baseline.jsonl"))!=spec['count']:
            raise RuntimeError('Baseline GPU stage not complete')
    # A live baseline CUDA process must not overlap training.
    launch=ah.read(ah.OUT/'launch-baseline.json')
    stat=Path(f"/proc/{launch['pid']}/stat")
    if stat.exists() and stat.read_text().split()[2]!='Z':raise RuntimeError('Baseline process still alive')
    stage=ah.OUT/'training-stage.json'
    ah.write_once(stage,dict(status='running',commands=['42-A','43-A','42-B','43-B'],automatic_retry=False))
    for seed,arm in [(42,'A'),(43,'A'),(42,'B'),(43,'B')]:
        if (ah.OUT/'experiment-failure.json').exists():raise RuntimeError('Failure latch: stop queue')
        label=f'{seed}-{arm}';log=ah.OUT/f'gpu-train-{label}.log'
        with log.open('xb') as stream:
            result=subprocess.run([str(ah.ROOT/'.venv/bin/python'),'-u',str(ah.ROOT/'scripts/run_v133_goal_ah_entity_relation.py'),
                'train','--seed',str(seed),'--arm',arm],cwd=ah.ROOT,stdout=stream,stderr=subprocess.STDOUT)
        ah.write_once(ah.OUT/f'training-exit-{label}.json',dict(returncode=result.returncode))
        if result.returncode:
            stage.write_text(json.dumps(dict(status='stopped_failure',failed_run=label,automatic_retry=False))+'\n')
            return
    stage.write_text(json.dumps(dict(status='completed',runs=4,total_updates=512,automatic_retry=False))+'\n')

if __name__=='__main__':main()
