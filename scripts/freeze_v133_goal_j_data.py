"""Validate Goal J blind votes, isolation, and CPU checks; freeze 16+16 only."""
from __future__ import annotations
import hashlib,json,sys,unicodedata,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/v133-goal-j-transfer";sys.path.insert(0,str(ROOT))
from scripts import run_v133_goal_g_cpu as g
from scripts.run_v133_goal_j_cpu import j_exposure,sha,rd,wrl,wr
from scripts.v130_review_protocol import valid_vote

def norm(s): return g.norm(s)
def load_votes(path):
 x=json.loads(path.read_text(encoding='utf-8'))
 return {r['case_id']:r for r in (x['reviews'] if isinstance(x,dict) else x)}
def luna_valid(packet,vote):
 if vote.get('case_id')!=packet['case_id'] or type(vote.get('approved')) is not bool or not isinstance(vote.get('reason'),str) or not vote['reason'].strip():return False,'vote_shape'
 docs={d['id']:d['text'] for d in packet['before_action']}
 c=vote.get('citations'); entries=[]
 if isinstance(c,dict): entries=[{'id':k,'excerpt':v} for k,v in c.items()]
 elif isinstance(c,list): entries=c
 exact=set()
 for item in entries:
  did=item.get('id',item.get('document_id')); quote=item.get('excerpt',item.get('quote'))
  if did not in docs or not isinstance(quote,str) or not quote.strip() or quote not in docs[did]:return False,'citation_not_exact_visible_quote'
  exact.add(did)
 if len(exact)<2:return False,'must_cite_both_documents'
 return True,'valid_exact_two_document_citations'

def main():
 prot=OUT/'PROTOCOL.md'; loc=json.loads((OUT/'locator-summary.json').read_text(encoding='utf-8')); part=json.loads((OUT/'partition-summary.json').read_text(encoding='utf-8'))
 if sha(prot)!=loc['protocol_sha256'] or sha(OUT/'partitioned-candidates.jsonl')!=part['partition_sha256']:raise ValueError('protocol_or_partition_hash_changed')
 pfiles=[OUT/'review-packets.jsonl',OUT/'review-packets-batch-02.jsonl'];mfiles=[OUT/'review-private-map.jsonl',OUT/'review-private-map-batch-02.jsonl']
 packets={};mapping={}
 for f in pfiles:
  for r in rd(f):
   if r['case_id'] in packets:raise ValueError('duplicate_packet_id')
   packets[r['case_id']]=r
 for f in mfiles:
  for r in rd(f):
   if r['case_id'] in mapping:raise ValueError('duplicate_private_map_id')
   mapping[r['case_id']]=r
 if set(packets)!=set(mapping):raise ValueError('packet_map_coverage_mismatch')
 luna={**load_votes(OUT/'luna-votes-attempt-01.json'),**load_votes(OUT/'luna-votes-batch-02.json')}
 deep={}
 for f in sorted((OUT/'deepseek').glob('j*/validated-review-private.json')):
  blob=json.loads(f.read_text(encoding='utf-8'))
  if blob.get('model')!='deepseek-flash':raise ValueError('deepseek_model_identity_mismatch')
  identity=json.loads((f.parent/'identity-check.json').read_text(encoding='utf-8'))
  if not identity.get('model_identity_verified'):raise ValueError('deepseek_response_identity_unverified')
  for r in blob['reviews']:
   if r['case_id'] in deep:raise ValueError('duplicate_deepseek_vote')
   deep[r['case_id']]=r
 if set(luna)!=set(packets) or set(deep)!=set(packets):raise ValueError('dual_review_coverage_mismatch')
 dispositions=[]
 for cid,pkt in packets.items():
  lv=luna[cid];dv=deep[cid];lok,lwhy=luna_valid(pkt,lv);dok,dwhy=valid_vote(pkt,dv)
  if not dok:raise ValueError(f'deepseek_vote_invalid:{cid}:{dwhy}')
  if not lok:raise ValueError(f'luna_vote_invalid:{cid}:{lwhy}')
  dispositions.append({'case_id':cid,'source_id':mapping[cid]['source_id'],'split':mapping[cid]['split'],'luna_approved':lv['approved'],'deepseek_approved':dv['approved'],'luna_valid':lok,'deepseek_valid':dok,'disposition':'qualified' if lv['approved'] and dv['approved'] else 'excluded','luna_reason':lv['reason'],'deepseek_reason':dv['reason']})
 eligible={r['case_id'] for r in dispositions if r['disposition']=='qualified'}
 selected={}
 for split in ('train','evaluation'):
  queue=sorted([mapping[c] for c in eligible if mapping[c]['split']==split],key=lambda x:x['sort_key'])
  if len(queue)<16:raise ValueError(f'qualified_target_shortfall:{split}:{len(queue)}')
  selected[split]=queue[:16]
 if any(len(selected[s])<8 for s in selected):raise ValueError('minimum_split_count_not_met')
 ids={s:{r['source_id'] for r in selected[s]} for s in selected}
 if ids['train']&ids['evaluation']:raise ValueError('source_overlap_between_splits')
 qhash={s:{r['question_sha256'] for r in selected[s]} for s in selected}
 if qhash['train']&qhash['evaluation']:raise ValueError('question_overlap_between_splits')
 group={s:{r['group_sha256'] for r in selected[s]} for s in selected}
 if group['train']&group['evaluation']:raise ValueError('entity_group_overlap_between_splits')
 exposed_ids,exposed_q,exposed_titles,input_hashes=j_exposure()
 chosenids=ids['train']|ids['evaluation']; chosenq=qhash['train']|qhash['evaluation']
 if chosenids&exposed_ids or chosenq&exposed_q:raise ValueError('historical_source_or_question_exposure')
 old_titles={norm(x) for x in exposed_titles}
 case_lookup={r['source_id']:r for r in rd(OUT/'provisional.jsonl')}
 selected_rows={}
 for split in ('train','evaluation'):
  out=[]
  for m in selected[split]:
   source=case_lookup[m['source_id']]
   if any(norm(t) in old_titles for t in source['evidence_titles']):raise ValueError(f'historical_entity_title_overlap:{m["source_id"]}')
   row=g.make_case(source);row['case_id']=m['source_id'];row['source_id']=m['source_id'];row['split']=split
   packet=packets[m['case_id']]
   if row['messages'][:6]!=packet['visible_messages'] or json.loads(row['messages'][-1]['content'])!=packet['proposed_action']:raise ValueError('reviewed_context_answer_binding_mismatch')
   out.append(row)
  selected_rows[split]=out
 # Check all Goal G opened titles/questions/source IDs for cross-wave entity isolation.
 gscreen=rd(ROOT/'runs/v133-goal-g-cpu/screen-ledger.jsonl')
 g_titles={norm(t) for r in gscreen for t in r.get('evidence_titles',[])}
 for rows in selected_rows.values():
  for r in rows:
   if any(norm(t) in g_titles for t in r['evidence_titles']):raise ValueError(f'Goal_G_screened_entity_overlap:{r["source_id"]}')
 # Cross-split groups also reject exact document-title or ID overlap.
 train_titles={norm(t) for r in selected_rows['train'] for t in r['evidence_titles']};eval_titles={norm(t) for r in selected_rows['evaluation'] for t in r['evidence_titles']}
 if train_titles&eval_titles:raise ValueError('entity_title_alias_overlap_between_splits')
 # Audit packets and votes are frozen; do not expose new-eval inputs to training configuration.
 pre={r['source_id']:r for f in (OUT/'cpu-preflight.jsonl',OUT/'cpu-preflight-batch-02.jsonl') for r in rd(f)}
 from transformers import AutoProcessor
 import torch,importlib.metadata as md
 if torch.cuda.is_available():raise RuntimeError('CPU_only_data_freeze_required')
 proc=AutoProcessor.from_pretrained('/root/autodl-tmp/qwen_08b',local_files_only=True);cpu=[]
 for split,rows in selected_rows.items():
  for row in rows:
   check=g.cpu_check(proc,row); prior=pre[row['source_id']]
   if prior['status']!='pass' or check['prompt_sha256']!=prior['check']['prompt_sha256']:raise ValueError('cpu_preflight_hash_mismatch')
   row['cpu_check']=check;cpu.append({'case_id':row['case_id'],'split':split,'status':'pass','check':check})
 wrl(OUT/'train-frozen.jsonl',selected_rows['train']);wrl(OUT/'evaluation-frozen.jsonl',selected_rows['evaluation']);wrl(OUT/'cpu-final.jsonl',cpu)
 wrl(OUT/'review-disposition.jsonl',dispositions)
 manifest={'version':'v133-goal-j-freeze-v1','status':'cpu_ready','cpu_ready':True,'evaluation_isolation_verified':True,'dual_review_complete':True,'training_count':16,'evaluation_count':16,'candidate_full_rows_opened':200,'candidate_structural_count':193,'review_packet_count':len(packets),'review_both_approved_count':len(eligible),'train_selected_private_ids':[r['source_id'] for r in selected['train']],'evaluation_selected_private_ids':[r['source_id'] for r in selected['evaluation']],'unmet_conditions':[],'input_hashes':input_hashes,'protocol_sha256':sha(prot),'locator_sha256':sha(OUT/'locator.jsonl'),'partition_sha256':sha(OUT/'partitioned-candidates.jsonl'),'screen_ledger_sha256':sha(OUT/'screen-ledger.jsonl'),'provisional_sha256':sha(OUT/'provisional.jsonl'),'packets_sha256':{f.name:sha(f) for f in pfiles},'luna_votes_sha256':{(OUT/f).name:sha(OUT/f) for f in ('luna-votes-attempt-01.json','luna-votes-batch-02.json')},'deepseek_batch_hashes':{f.parent.name:sha(f) for f in (OUT/'deepseek').glob('j*/validated-review-private.json')},'train_sha256':sha(OUT/'train-frozen.jsonl'),'evaluation_sha256':sha(OUT/'evaluation-frozen.jsonl'),'cpu_sha256':sha(OUT/'cpu-final.jsonl'),'review_disposition_sha256':sha(OUT/'review-disposition.jsonl'),'tokenizer_dir':'/root/autodl-tmp/qwen_08b','versions':{n:md.version(n) for n in ('torch','transformers','tokenizers','peft')},'gpu_used':False,'training_executable':True}
 wr(OUT/'freeze-manifest.json',manifest)
 print(json.dumps({'status':manifest['status'],'train':16,'evaluation':16,'reviewed':len(packets),'both_approved':len(eligible),'train_sha256':manifest['train_sha256'],'evaluation_sha256':manifest['evaluation_sha256'],'manifest_sha256':sha(OUT/'freeze-manifest.json')}))
if __name__=='__main__':main()
