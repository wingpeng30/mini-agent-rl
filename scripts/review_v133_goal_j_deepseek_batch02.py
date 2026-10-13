"""One-shot frozen DeepSeek review of Goal J's second deterministic wave."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.v130_deepseek_review import MODEL,_credentials,_no_redirect_opener,run_batch
OUT=ROOT/"runs/v133-goal-j-transfer"; PACKETS=OUT/"review-packets-batch-02.jsonl"
def main():
 p=argparse.ArgumentParser();p.add_argument('--batch-index',type=int,choices=range(4),required=True);a=p.parse_args()
 rows=[json.loads(x) for x in PACKETS.read_text(encoding='utf-8').splitlines() if x.strip()]
 digest=hashlib.sha256(PACKETS.read_bytes()).hexdigest();lo=a.batch_index*8;hi=min(lo+8,len(rows));group=rows[lo:hi]
 if len(rows)!=29:raise ValueError('frozen_batch02_count_mismatch')
 key,base=_credentials()
 if not key or base!='https://api.deepseek.com':raise RuntimeError('existing_deepseek_credential_or_host_unavailable')
 batch=f"j2-{lo:02d}-{hi-1:02d}";folder=OUT/'deepseek'/batch
 result=run_batch(batch=batch,packets=group,packet_file_sha256=digest,freeze_sha256=digest,batch_dir=folder,key=key,opener=_no_redirect_opener())
 raw=sorted((folder/'responses-private').glob('attempt-*-raw.json'))[-1];model=json.loads(raw.read_text(encoding='utf-8')).get('model')
 ident={'requested_model':MODEL,'response_model':model,'model_identity_verified':model==MODEL,'packet_file_sha256':digest,'review_count':len(result['reviews']),'approved':sum(x['approved'] for x in result['reviews'])}
 (folder/'identity-check.json').write_text(json.dumps(ident,indent=2)+'\n',encoding='utf-8');print(json.dumps(ident))
if __name__=='__main__':main()
