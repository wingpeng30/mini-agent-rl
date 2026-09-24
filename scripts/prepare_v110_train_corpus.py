"""Freeze the existing first-100-task training corpus without changing its contents."""
import json
from pathlib import Path
from mini_agent_rl.evaluation import load_hotpot_environment

if __name__=='__main__':
    env=load_hotpot_environment('data/hotpot-agent-v041/train.jsonl',task_limit=100,corpus_limit=100)
    root=Path('data/hybrid-v110');root.mkdir(parents=True,exist_ok=True)
    (root/'train-corpus.json').write_text(json.dumps(env.corpus,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'train-environment.json').write_text(json.dumps(env.fingerprint('bm25-v2','compact-v3'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(len(env.tasks),len(env.corpus),env.corpus_sha256)
