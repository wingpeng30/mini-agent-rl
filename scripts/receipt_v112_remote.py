"""Collect read-only receipts for the completed fixed v112 pilot."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]
paths=['reports/v112-training-report.json','reports/v112-prompt-replay-remote.json',
       'reports/v112-internal-comparison.json']
paths += [f'reports/v112-replay2-{arm}-{mode}.json' for arm in ('baseline','candidate') for mode in ('states','tasks')]
paths += ['checkpoints/v112-evidence-sft-seed42/formal/adapter_model.safetensors',
          'checkpoints/v112-evidence-sft-seed42/formal/adapter_config.json']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0].rsplit(',',1)
receipt={'schema_version':1,'version':'v112','captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'instance_running':True,'gpu':{'name':gpu[0].strip(),'memory_used_mib':int(gpu[1])},
    'source_adapter_sha256':sha(root/'checkpoints/v101-answer50-seed42/adapter_model.safetensors'),
    'remote_same_path_ledger_present':(root/'runs/v110-api.sqlite').exists(),
    'independent100_evaluated':False,'final_test_accessed':False,
    'access_claim_basis':'All v112 scripts in this run were restricted to the frozen training source and its fixed internal split; this is an execution-scope record, not a system-wide filesystem audit.',
    'artifacts':[{'path':p,'sha256':sha(root/('checkpoints/v112-evidence-sft-seed42/training_report.json' if p=='reports/v112-training-report.json' else p))} for p in paths]}
out=root/'reports/v112-remote-receipt.json'
if out.exists():raise ValueError('receipt_already_exists')
out.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
