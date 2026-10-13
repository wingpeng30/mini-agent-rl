"""Goal AO: matched relation-contrast action SFT from Goal Z.

Every mutable stage is bound to an immutable manifest; failures latch the run.
The script is intended to be synced and executed in the remote AutoDL checkout.
"""
from __future__ import annotations
import argparse, asyncio, gc, hashlib, json, math, re, sys, time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import run_v133_goal_am as am
import run_v133_goal_an as an
import run_v133_goal_q_search as q

OUT=ROOT/"runs/v133-goal-ao-relation-contrast-sft-v2"
Z=ROOT/"runs/v133-goal-z-multihop-sft"
AM=ROOT/"runs/v133-goal-am-action-sft-v2"
AN=ROOT/"runs/v133-goal-an-hard-state-sft-v2"
AA=ROOT/"runs/v133-goal-aa-audit-confirmation"
SOURCE=am.SOURCE; BASE=am.BASE; EXPECTED=am.EXPECTED_ADAPTER
PC=dict(am.PC); OPT=dict(am.OPT)
CAPS=dict(updates=512,generations=1232,searches=768)
GROUPS=[
 {"case_id":"musique:2hop__407314_80178","question":"Who gave the Believe Tour?","answer":"Justin Bieber","rationale":"tour recipient versus collaboration performer"},
 {"case_id":"2wiki:b2addc160bdc11eba7f7acde48001122","question":"Who was the son of Ossian Donner identified as a British Conservative MP?","answer":"Patrick Donner","rationale":"son's identity versus father's workplace"},
 {"case_id":"2wiki:fa9654640bdb11eba7f7acde48001122","question":"When did the director of film Return Of The Seven die?","answer":"February 15, 2001","rationale":"director death date versus birth date"},
 {"case_id":"2wiki:a874fa560bda11eba7f7acde48001122","question":"When was the director of film Terror Mountain born?","answer":"June 28, 1898","rationale":"director birth date versus birthplace"},
 {"case_id":"musique:2hop__829831_135304","question":"Who was the father of the performer of Animal Justice?","answer":"Will Cale","rationale":"father versus mother"},
 {"case_id":"2wiki:67adecd80bdc11eba7f7acde48001122","question":"What is the date of death of Richilde Of Provence herself?","answer":"2 June 910","rationale":"Richilde herself versus spouse's death date"},
 {"case_id":"musique:2hop__418766_73460","question":"On what date did the country containing Kavisigamuwa become a republic?","answer":"May 22, 1972","rationale":"republic date versus independence date"},
 {"case_id":"musique:2hop__135820_578368","question":"What is the ZIP code of Carpenter in the state that SoDak Con was named after?","answer":"57322","rationale":"ZIP code versus county"},
]

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def write_once(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(v,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def write_jsonl(p,values):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8',newline='\n') as f:
  for v in values:f.write(json.dumps(v,ensure_ascii=False,sort_keys=True)+'\n')
def append(p,v):
 with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(v,ensure_ascii=False,sort_keys=True)+'\n');f.flush()
def tok():return am.tokenizer_from_base()
def decode(t,ids):return t.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
def parse_messages(text):
 # Inputs use Qwen ChatML, with a final open assistant generation header.
 out=[];pos=0
 while pos<len(text):
  if text.startswith('<|im_start|>assistant',pos) and text[pos:].rstrip() in ('<|im_start|>assistant','<|im_start|>assistant\n'):
   pos=len(text);break
  if not text.startswith('<|im_start|>',pos):
   if text[pos:].strip():raise ValueError('unexpected text outside ChatML message')
   break
  hstart=pos+len('<|im_start|>');sep=text.find('\n',hstart)
  if sep<0:raise ValueError('ChatML header missing newline')
  header=text[hstart:sep].strip();role=header.split()[0]
  if role not in ('system','user','assistant','tool'):raise ValueError(f'unsupported role {role}')
  end=text.find('<|im_end|>',sep+1)
  if end<0 and role=='assistant':
   # Generation headers may include the model's open empty thinking prefix.
   # It is supplied again by the production chat template during re-render.
   break
  if end<0:raise ValueError('unterminated non-assistant ChatML message')
  out.append(dict(role=role,content=text[sep+1:end]))
  pos=end+len('<|im_end|>')
  if text.startswith('\n',pos):pos+=1
 if not out:raise ValueError('input has no complete messages before generation prefix')
 return out
def render_ids(client,messages):
 x=client._render_chat([__import__('mini_agent_rl.domain.models',fromlist=['Message']).Message(**m) for m in messages])
 return x['input_ids'][0].tolist()
def target_ids(tokenizer,action):
 raw=json.dumps(action,ensure_ascii=False,separators=(',',':'))
 end=tokenizer.convert_tokens_to_ids('<|im_end|>')
 ids=tokenizer.encode(raw,add_special_tokens=False)+[int(end)]
 return raw,ids
def action_state(tokenizer,client,source,question,answer,kind,base_messages=None):
 full_source_ids=list(source['prompt_token_ids']); full_source_text=decode(tokenizer,full_source_ids)
 messages=[dict(m) for m in base_messages] if base_messages is not None else parse_messages(full_source_text)
 user_positions=[i for i,m in enumerate(messages) if m['role']=='user']
 if not user_positions:raise ValueError('no initial user message')
 ui=user_positions[0];old=source['question']
 if messages[ui]['content'].count(old)!=1:raise ValueError(f'question not unique in first user message: {source["state_id"]}')
 original_ids=render_ids(client,messages)
 if base_messages is None and original_ids!=full_source_ids:raise ValueError(f'original input does not reproduce exactly: {source["state_id"]}')
 if base_messages is not None and messages!=parse_messages(full_source_text)[:len(messages)]:raise ValueError('selected source prefix is not a true history prefix')
 edited=[dict(m) for m in messages];edited[ui]['content']=edited[ui]['content'].replace(old,question,1)
 new_ids=render_ids(client,edited)
 for i,(before,after) in enumerate(zip(messages,edited)):
  if i!=ui and before!=after:raise ValueError('non-user history modified')
 if len(edited)!=len(messages):raise ValueError('message count changed')
 act={'action':'answer','answer':answer}
 if kind=='original':
  ids=list(source['action_token_ids']);raw=decode(tokenizer,ids)
 else:raw,ids=target_ids(tokenizer,act)
 text=decode(tokenizer,ids)
 if json.JSONDecoder().raw_decode(text)[0]!=act:raise ValueError('target action token roundtrip failed')
 citations=[]
 for c in (c for group in source['citations'] for c in (group if isinstance(group,list) else [group])):
  # AM citations are exact body quote/title records. Resolve positions in the actual prompt.
  found=an.prior.cite(new_ids,tokenizer,c['document_id'],c['quote'])
  if found is None:raise ValueError(f'citation absent from rewritten input: {source["state_id"]}/{c["document_id"]}')
  citations.append(dict(**found,triple=c.get('triple')))
 return dict(state_id=f"ao-{kind}:{source['state_id']}",source_state_id=source['state_id'],case_id=source['case_id'],
  group_id=source['case_id'],state_type='derived_sufficient_evidence_answer',variant=kind,question=question,answer=answer,
  target_action=act,action_token_ids=ids,prompt_token_ids=new_ids,prompt_attention_mask=[1]*len(new_ids),
  source_prompt_token_ids=original_ids,full_source_prompt_token_ids=full_source_ids,source_answer=source['target_action']['answer'],
  source_prefix_kind='earliest-real-evidence-complete-prefix-before-Justin-Bieber-query' if base_messages is not None else 'frozen-AM-answer-state',
  changed_only_initial_user_question=True,messages_before=messages,messages_after=edited,
  citations=citations,teacher_supplemented=True,autonomous_search_trajectory=False,evaluation_use_forbidden=True,
  rationale='same visible evidence; relation contrast: '+next(g['rationale'] for g in GROUPS if g['case_id']==source['case_id']),
  target_json=raw)

def prepare():
 if not OUT.exists():raise FileNotFoundError(OUT)
 unexpected=[p.name for p in OUT.iterdir() if p.name not in ('PLAN.md','preparation-attempts.jsonl')]
 if unexpected:raise FileExistsError(f'AO output is not empty: {unexpected}')
 if sha(SOURCE/'adapter_model.safetensors')!=EXPECTED:raise RuntimeError('Goal Z adapter identity mismatch')
 if not read(AA/'product-decision.json').get('published'):raise RuntimeError('AA release record missing')
 from transformers import AutoProcessor
 processor=AutoProcessor.from_pretrained(str(BASE),local_files_only=True)
 tokenizer=getattr(processor,'tokenizer',processor)
 states=rows(AM/'candidate-training-states.jsonl')
 bycase={s['case_id']:s for s in states if s['state_type']=='answer'}
 if len(bycase)!=24:raise RuntimeError('AM answer state source set changed')
 # The product renderer is CPU-only until generation/prepare_for_training.
 c,_,_=q.make_client(SOURCE,deterministic=True,product_config=PC,model_path=BASE,corpus=Z/'corpus-dev.json')
 if c._model is not None:raise RuntimeError('prepare must not load weights')
 c._processor=processor
 source_groups=[];derived=[]
 eval_ids,eval_questions,_=am.eval_isolation()
 for i,g in enumerate(GROUPS):
  s=bycase.get(g['case_id'])
  if s is None:raise ValueError('fixed group has no AM answer state: '+g['case_id'])
  if s['target_action']!={'action':'answer','answer':s['source_review_reason'] if False else s['target_action'].get('answer')}:
   raise ValueError('malformed AM answer target')
  # Recheck all source evidence and preserve AM's corrected citation revisions.
  ok,_=am.cite_check(tokenizer,s,[[c] for c in s['citations']])
  if not ok:raise ValueError('source state citation failed: '+g['case_id'])
  source_messages=parse_messages(decode(tokenizer,s['prompt_token_ids']))
  base_messages=None
  if g['case_id']=='musique:2hop__407314_80178':
   if len(source_messages)<4 or [m['role'] for m in source_messages[:4]]!=['system','user','assistant','user']:
    raise ValueError('Believe Tour safe prefix structure changed')
   first_action=json.loads(source_messages[2]['content'].split('</think>')[-1].strip())
   if first_action.get('action')!='search' or am.key(first_action.get('query',''))!=am.key(s['question']):
    raise ValueError('safe prefix is not the source question search')
   base_messages=source_messages[:4]
   prefix_ids=render_ids(c,base_messages)
   needed=[x for group in s['citations'] for x in (group if isinstance(group,list) else [group])]
   if not all(an.prior.cite(prefix_ids,tokenizer,x['document_id'],x['quote']) is not None for x in needed):
    raise ValueError('pre-leak prefix does not preserve all original supporting quotes as parsed document bodies')
   if g['answer'].casefold() in source_messages[2]['content'].casefold():raise ValueError('pre-leak search action still reveals contrast answer')
  original=action_state(tokenizer,c,s,s['question'],s['target_action']['answer'],'original',base_messages)
  if base_messages is None and original['prompt_token_ids']!=s['prompt_token_ids']:
   raise ValueError('source replay differs from AM token archive')
  alt=action_state(tokenizer,c,s,g['question'],g['answer'],'relation-contrast',base_messages)
  for x in (original,alt):
   qn=am.key(x['question'])
   if x['case_id'] in eval_ids or qn in eval_questions:raise ValueError('evaluation overlap: '+x['state_id'])
   if len(x['prompt_token_ids'])>PC['max_input_tokens'] or len(x['action_token_ids'])>PC['max_new_tokens']:
    raise ValueError('state exceeds frozen inference budget')
  if len(original['messages_after'])!=len(alt['messages_after']):raise ValueError('message count differs')
  first_user=next(k for k,m in enumerate(alt['messages_after']) if m['role']=='user')
  if any(a!=b for j,(a,b) in enumerate(zip(original['messages_after'],alt['messages_after'])) if j!=first_user):
   raise ValueError('counterfactual changed non-question history')
  # Correct answers should occur in tool evidence, but must not be embedded in
  # an earlier assistant action/query that would leak the target.
  for j,m in enumerate(alt['messages_after']):
   if j!=first_user and m['role']=='assistant' and g['answer'].casefold() in m['content'].casefold():
    raise ValueError('contrast answer leaks in historical assistant action')
  doc_records=an.prior.documents(am.decode_input(tokenizer,alt['prompt_token_ids']))
  evidence_hits=[]
  for d in doc_records:
   body=d.get('text','');at=body.casefold().find(g['answer'].casefold())
   if at>=0:
    lo=max(0,body.rfind('.',0,at)+1);hi=body.find('.',at)
    if hi<0:hi=len(body)
    evidence_hits.append(dict(document_id=d['document_id'],title=d['title'],quote=body[lo:hi+1].strip(),
      body_char_span=[at,at+len(g['answer'])]))
  if not evidence_hits:raise ValueError('contrast answer absent from actual retrieved document body')
  source_groups.append(dict(group_id=g['case_id'],group_order=i,source_state_id=s['state_id'],case_id=s['case_id'],
   original_question=s['question'],original_answer=s['target_action']['answer'],contrast_question=g['question'],contrast_answer=g['answer'],
   rationale=g['rationale'],source='AM-candidate-training-state',contrast_candidate_evidence=evidence_hits,
   states=[original,alt],admission='pending_semantic_review'))
  derived.extend([original,alt])
 if len(source_groups)!=8 or len(derived)!=16:raise RuntimeError('fixed AO material count mismatch')
 if len({x['state_id'] for x in derived})!=16:raise RuntimeError('duplicate AO state ids')
 # Explicit pairwise relationship checks rely on original body quotes, not answer strings.
 evidence_review=[]
 for gr in source_groups:
  evidence_review.append(dict(group_id=gr['group_id'],case_id=gr['case_id'],source_state_id=gr['source_state_id'],
   same_documents=[c['document_id'] for c in gr['states'][0]['citations']],
   original_question=gr['original_question'],original_answer=gr['original_answer'],contrast_question=gr['contrast_question'],contrast_answer=gr['contrast_answer'],
   original_quotes=[{k:c.get(k) for k in ('document_id','title','quote','triple','char_start','char_end','token_start','token_end')} for c in gr['states'][0]['citations']],
  contrast_quotes=[{k:c.get(k) for k in ('document_id','title','quote','triple','char_start','char_end','token_start','token_end')} for c in gr['states'][1]['citations']],
   contrast_candidate_evidence=gr['contrast_candidate_evidence'],
   same_history_except_initial_question=True,derived_state_not_autonomous=True,eval_isolated=True))
 # Reuse AM probes and AN partial probes without changing inputs.
 probes=rows(AM/'frozen-probes.jsonl');complete=[p for p in probes if p['category']=='complete']
 partial=rows(AN/'partial-probes.jsonl')
 if len(complete)!=24 or len(partial)!=8:raise RuntimeError('frozen probe set mismatch')
 ztrain=rows(Z/'training-inputs.jsonl')
 if len(ztrain)!=128:raise RuntimeError('Goal Z original trajectory count mismatch')
 order=read(AM/'execution-freeze.json')['order']
 if any(len(v)!=128 for v in order.values()):raise RuntimeError('AM frozen order not complete')
 # Snapshot exact dependencies; source artifacts are read-only.
 files={'runner':Path(__file__),'am_runner':ROOT/'scripts/run_v133_goal_am.py','an_runner':ROOT/'scripts/run_v133_goal_an.py',
  'q_runner':ROOT/'scripts/run_v133_goal_q_search.py','model_client':ROOT/'src/mini_agent_rl/model/local.py',
  'search':ROOT/'src/mini_agent_rl/tools/search.py','optimizer':ROOT/'src/mini_agent_rl/training/grpo.py',
  'adapter':SOURCE/'adapter_model.safetensors','aa_release':AA/'product-decision.json','am_states':AM/'candidate-training-states.jsonl',
  'am_freeze':AM/'execution-freeze.json','am_report':AM/'REPORT.md','am_probes':AM/'frozen-probes.jsonl',
  'am_evaluation':AM/'evaluation-cases.jsonl','am_adapter_A':AM/'seed42-A/adapter-final/adapter_model.safetensors',
  'am_adapter_B':AM/'seed42-B/adapter-final/adapter_model.safetensors','an_partial':AN/'partial-probes.jsonl',
  'z_training':Z/'training-inputs.jsonl','z_corpus_train':Z/'corpus-train.json','z_corpus_dev':Z/'corpus-dev.json',
  'tokenizer':BASE/'tokenizer.json','base_config':BASE/'config.json'}
 for p in sorted(BASE.glob('model.safetensors-*-of-*.safetensors')):files[f'base_shard_{len([k for k in files if k.startswith("base_shard_")]):02d}']=p
 hashes={k:sha(v) for k,v in files.items()}
 paths={k:str(v.resolve()) for k,v in files.items()}
 write_once(OUT/'ao-groups.json',source_groups)
 write_jsonl(OUT/'derived-states.jsonl',derived)
 write_once(OUT/'evidence-admission.json',evidence_review)
 write_jsonl(OUT/'frozen-probes.jsonl',complete+partial)
 write_jsonl(OUT/'training-fit-probes.jsonl',[dict(probe_id=f"fit:{s['state_id']}",case_id=s['case_id'],group_id=s['group_id'],variant=s['variant'],question=s['question'],answer=s['answer'],prompt_token_ids=s['prompt_token_ids'],expected=s['target_action'],no_search_execution=True) for s in derived])
 write_jsonl(OUT/'evaluation-cases.jsonl',rows(AM/'evaluation-cases.jsonl'))
 cfg=dict(goal='AO',status='cpu_preflight_pending',source_model=str(SOURCE),base=str(BASE),expected_adapter_sha256=EXPECTED,
  product=PC,optimizer=OPT,seeds=[42,43],updates_per_arm=128,trajectory_count=128,am_states=34,relation_groups=8,
  derived_question_states=16,group_schedule='updates 72..103, table order repeated four times',
  arms=dict(A='same Z+AM; each group: original question duplicated',B='same Z+AM; each group: original question plus contrast question'),
  loss=dict(formula='Ltrajectory + 0.25*LAM_state + 0.25*Lrelation_group',trajectory='all assistant target tokens normalized by token count',
   state='current JSON action + end token, mean NLL',relation='mean of two per-action mean NLLs; A repeats identical original twice'),
  order=order,probes=dict(dev=64,complete=24,partial=8,fit=16),
  gate=dict(supported_B_at_least=23,relative_supported_gain_B_vs_A=2,complete_B_at_least=22,relative_complete_gain=2,
   requires_binding_repair=True,errors_B_no_higher_than_A_and_at_most=7,action_problem_cases_at_most=1,protect_AM_Rapana_and_Duke=True,
   old_protected_losses_at_most=1,partial_degradation_none=True,partial_legal_search_min=7),
  limits=dict(training_runs=4,updates=512,real_trajectories=256,generations=1232,searches=768,new_confirmation=0,
   stage1=dict(generations=624,searches=384,updates=256)),hashes=hashes,paths=paths,
  data_policy='training-only derived sufficient-evidence states; evaluation excluded by ID and normalized question',
  failure_policy='identity/GPU/numeric/infrastructure: latch, preserve, stop; no retry/tuning',gpu_authorized=True,released=False)
 cfg['output_hashes']={name:sha(OUT/name) for name in ('ao-groups.json','derived-states.jsonl','evidence-admission.json','frozen-probes.jsonl','training-fit-probes.jsonl','evaluation-cases.jsonl')}
 write_once(OUT/'manifest.json',cfg)
 digest=sha(OUT/'manifest.json')
 write_once(OUT/'gpu-approval.json',dict(human_authorized=True,manifest_sha256=digest,authorization_basis='用户已明确授权Goal AO GPU执行并说明以后不需再次确认',scope='frozen AO plan; seed43 conditional on frozen gate'))
 write_once(OUT/'status.json',dict(status='cpu_preflight_pending',generations=0,searches=0,updates=0,prepared_at=time.time()))
 print(json.dumps(dict(prepared=True,groups=len(source_groups),states=len(derived),manifest_sha256=digest),ensure_ascii=False))

def frozen(require_preflight=True):
 if (OUT/'experiment-failure.json').exists():raise RuntimeError('AO failure latch is present')
 cfg=read(OUT/'manifest.json')
 for k,p in cfg['paths'].items():
  if sha(p)!=cfg['hashes'][k]:raise RuntimeError('frozen source changed: '+k)
 for name,digest in cfg['output_hashes'].items():
  if sha(OUT/name)!=digest:raise RuntimeError('frozen AO material changed: '+name)
 if not cfg.get('materials_adjudicated'):raise RuntimeError('8 fixed groups have not passed primary source adjudication')
 if sha(SOURCE/'adapter_model.safetensors')!=EXPECTED:raise RuntimeError('Goal Z adapter identity mismatch')
 if sha(OUT/'manifest.json')!=(read(OUT/'gpu-approval.json').get('manifest_sha256')):raise RuntimeError('authorization does not match AO freeze')
 if require_preflight:
  pf=read(OUT/'cpu-preflight.json')
  if not pf.get('passed') or pf.get('manifest_sha256')!=sha(OUT/'manifest.json'):raise RuntimeError('AO CPU preflight missing/mismatched')
 return cfg

def counts(kind):return sum(json.loads(x).get('kind')==kind for x in (OUT/'gpu-usage.jsonl').read_text().splitlines()) if (OUT/'gpu-usage.jsonl').exists() else 0
def record(kind,stage,case):
 cfg=read(OUT/'manifest.json');limit=cfg['limits'][kind]
 if counts(kind)>=limit:raise RuntimeError(f'Goal AO {kind} cap exceeded')
 append(OUT/'gpu-usage.jsonl',dict(kind=kind,stage=stage,case_id=case,attempt=counts(kind)+1,status='started',time=time.time()))
def product_client(adapter,corpus):return q.make_client(adapter,deterministic=True,product_config=PC,model_path=BASE,corpus=corpus)
def target_loss(client,action,scale=1.):
 import torch
 vals=client.score_action_tensor([],action['action_token_ids'],True,action['prompt_token_ids'],action['prompt_attention_mask']).float()
 if len(vals)!=len(action['action_token_ids']) or not torch.isfinite(vals).all():raise RuntimeError('invalid action token score')
 loss=-scale*vals.mean()
 if not torch.isfinite(loss):raise RuntimeError('nonfinite AO loss')
 loss.backward();return float(loss.detach())
def sample_state(s):return dict(prompt_token_ids=s['prompt_token_ids'],prompt_attention_mask=s['prompt_attention_mask'],action_token_ids=s['action_token_ids'])

def train(seed,arm):
 cfg=frozen();label=f'{seed}-{arm}'
 if seed not in (42,43) or arm not in ('A','B'):raise ValueError('invalid frozen arm')
 if seed==43 and not read(OUT/'seed42-gate.json').get('route_passed'):raise RuntimeError('seed43 gate is closed')
 if counts('updates')+128>CAPS['updates']:raise RuntimeError('update quota would be exceeded')
 folder=OUT/f'seed{label}';folder.mkdir(exist_ok=False)
 import torch
 from transformers import set_seed
 from mini_agent_rl.training.grpo import guarded_optimizer_step
 import run_v133_goal_o_budget as budget
 if torch.cuda.device_count()!=1:raise RuntimeError('AO requires exactly one visible GPU')
 set_seed(seed);client,wrapped,registry=product_client(SOURCE,Z/'corpus-train.json')
 params_named=client.prepare_for_training();params=[p for _,p in params_named]
 if not params or any('lora_' not in n for n,_ in params_named):raise RuntimeError('only LoRA may be trainable')
 base_before=budget.parameter_hashes(client._model,lora=False);lora_before=budget.parameter_hashes(client._model,lora=True)
 traj={r['id']:r for r in rows(Z/'training-inputs.jsonl')};states=rows(AM/'candidate-training-states.jsonl')
 derived=rows(OUT/'derived-states.jsonl');bygroup={}
 for s in derived:bygroup.setdefault(s['group_id'],{})[s['variant']]=s
 order=cfg['order'][str(seed)];opt=cfg['optimizer'];optimizer=torch.optim.AdamW(params,lr=opt['learning_rate'],betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=opt['weight_decay'])
 am_schedule={2+2*i:s for i,s in enumerate(states)}
 group_ids=[g['group_id'] for g in read(OUT/'ao-groups.json')]
 group_schedule={72+i:group_ids[i%8] for i in range(32)}
 for update,cid in enumerate(order,1):
  optimizer.zero_grad(set_to_none=True);client.set_training_forward_mode()
  acts=traj[cid]['actions'];denom=sum(len(a['action_token_ids']) for a in acts)
  tl=sum(target_loss(client,a,scale=len(a['action_token_ids'])/denom) for a in acts)
  ams=am_schedule.get(update);sl=0.
  if ams:sl=target_loss(client,ams,scale=.25)
  gid=group_schedule.get(update);gl=0.
  if gid:
   pair=bygroup[gid]
   first=pair['original'];second=pair['original'] if arm=='A' else pair['relation-contrast']
   gl=target_loss(client,first,scale=.25/2)+target_loss(client,second,scale=.25/2)
  grad=float(torch.nn.utils.clip_grad_norm_(params,opt['max_grad_norm']))
  if not math.isfinite(grad) or grad<=0:raise RuntimeError('invalid AO gradient')
  record('updates',f'train-{label}',cid);guarded_optimizer_step(optimizer,params,client.gpu_memory_stats(),21.)
  if any(not torch.isfinite(p).all() for p in params):raise RuntimeError('nonfinite LoRA parameter')
  row=dict(update=update,trajectory_id=cid,am_state=ams['state_id'] if ams else None,relation_group=gid,
    relation_variants=([first['variant'],second['variant']] if gid else []),trajectory_loss=tl,am_state_loss=sl,relation_loss=gl,
    gradient_norm=grad,peak_memory=client.gpu_memory_stats())
  append(folder/'updates.jsonl',row)
  print(json.dumps(dict(stage=label,**row),ensure_ascii=False),flush=True)
 if budget.parameter_hashes(client._model,lora=False)!=base_before:raise RuntimeError('base weights changed during AO')
 lora_after=budget.parameter_hashes(client._model,lora=True)
 changed=sum(v!=lora_after[n] for n,v in lora_before.items())
 if changed!=192:raise RuntimeError(f'expected 192 changed LoRA tensors, got {changed}')
 adapter=folder/'adapter-final';client._model.save_pretrained(adapter)
 action=traj[order[0]]['actions'][-1];expected=client.score_action_tensor([],action['action_token_ids'],False,action['prompt_token_ids'],action['prompt_attention_mask']).detach().float().cpu()
 del client,wrapped,registry,optimizer,params,params_named;gc.collect();torch.cuda.empty_cache()
 reload,_,_=product_client(adapter,Z/'corpus-train.json')
 actual=reload.score_action_tensor([],action['action_token_ids'],False,action['prompt_token_ids'],action['prompt_attention_mask']).detach().float().cpu()
 delta=float((expected-actual).abs().max())
 if not torch.isfinite(actual).all() or delta>.001:raise RuntimeError('AO independent reload check failed')
 write_once(folder/'summary.json',dict(seed=seed,arm=arm,updates=128,base_unchanged=True,changed_lora_tensors=changed,reload_max_abs=delta,adapter_sha256=sha(adapter/'adapter_model.safetensors')))

def instrument(wrapped,registry,stage,active):
 gen=wrapped.generate;execute=registry.execute
 async def counted(*a,**kw):record('generations',stage,active['id']);return await gen(*a,**kw)
 async def counted_exec(*a,**kw):
  if a and a[0]=='search':record('searches',stage,active['id'])
  return await execute(*a,**kw)
 wrapped.generate=counted;registry.execute=counted_exec
def evaluate(label):
 frozen();
 if label not in ('42-A','42-B','43-A','43-B'):raise ValueError('unfrozen model selector')
 if label.startswith('43-') and not read(OUT/'seed42-gate.json').get('route_passed'):raise RuntimeError('seed43 gate closed')
 adapter=OUT/f'seed{label}/adapter-final';summ=read(adapter.parent/'summary.json')
 if sha(adapter/'adapter_model.safetensors')!=summ['adapter_sha256']:raise RuntimeError('candidate adapter hash mismatch')
 out=OUT/f'eval-{label}.jsonl'
 if out.exists():raise FileExistsError(out)
 client,wrapped,registry=product_client(adapter,Z/'corpus-dev.json');active={'id':'unassigned'};instrument(wrapped,registry,'eval-'+label,active)
 cases=rows(OUT/'evaluation-cases.jsonl')
 for i,case in enumerate(cases,1):
  active['id']=case['case_id'];r=asyncio.run(q.rollout(case['question'],wrapped,registry,sample=False,seed=42,product_config=PC))
  r.update(case_id=case['case_id'],model=label,official_answer=case['answer'],**q.score(r['answer'],case['answer']))
  if r['reason']=='tool_error':raise RuntimeError('search tool infrastructure failure')
  if r['model_calls']>4 or r['searches']>3 or any(len(t['prompt_token_ids'])>16384 or len(t['action_token_ids'])>2048 for t in r['trace']):raise RuntimeError('product budget overflow')
  append(out,r);print(json.dumps(dict(stage='eval-'+label,case=i,em=r['em'],reason=r['reason'])),flush=True)
 del client,wrapped,registry;gc.collect();import torch;torch.cuda.empty_cache()
def probe(label,kind):
 frozen();
 if label not in ('AM','42-A','42-B','43-A','43-B'):raise ValueError('unfrozen probe model')
 if label=='AM':adapter=AM/'seed42-A/adapter-final'
 else:
  adapter=OUT/f'seed{label}/adapter-final'
 if kind=='training-fit':source=rows(OUT/'training-fit-probes.jsonl');out=OUT/f'probe-{label}-training-fit.jsonl'
 elif kind in ('complete','partial'):source=[p for p in rows(OUT/'frozen-probes.jsonl') if p['category']==kind];out=OUT/f'probe-{label}-{kind}.jsonl'
 else:raise ValueError(kind)
 if out.exists():raise FileExistsError(out)
 import torch
 c,_,_=product_client(adapter,Z/'corpus-dev.json')
 c._inputs=lambda messages,phase='unknown':dict(input_ids=torch.tensor([current['prompt_token_ids']],device=c._model.device),attention_mask=torch.ones((1,len(current['prompt_token_ids'])),dtype=torch.long,device=c._model.device))
 for current in source:
  record('generations','probe-'+label+'-'+kind,current['case_id']);start=time.monotonic();r=c._generate_sync([],0.,42);elapsed=time.monotonic()-start
  if r.prompt_token_ids!=current['prompt_token_ids']:raise RuntimeError('probe input changed')
  if not r.action_token_ids or len(r.action_token_ids)>2048:raise RuntimeError('probe action missing/too long')
  from run_v133_goal_an import parsed_action
  append(out,dict(model=label,probe_id=current['probe_id'],case_id=current['case_id'],group_id=current.get('group_id'),variant=current.get('variant'),
   category=kind,action=parsed_action(r),answer=r.content,query=r.tool_call.arguments.get('query') if r.tool_call else None,
   raw=r.raw_text,parse_error=r.parse_error,prompt_token_ids=r.prompt_token_ids,action_token_ids=r.action_token_ids,
   usage=r.usage,elapsed_seconds=elapsed,no_search_execution=True))
 del c;gc.collect();torch.cuda.empty_cache()

def cpu_preflight():
 cfg=frozen(False)
 # Re-check pair/mask/count invariants and loss scaling analytically without loading model weights or using GPU.
 groups=read(OUT/'ao-groups.json');states=rows(OUT/'derived-states.jsonl');amstates=rows(AM/'candidate-training-states.jsonl')
 if len(groups)!=8 or len(states)!=16 or len(amstates)!=34:raise RuntimeError('frozen material counts mismatch')
 if sum(1 for i in range(1,129) if i in range(2,69,2))!=34:raise RuntimeError('AM schedule mismatch')
 gs=[72+i for i in range(32)]
 if gs!=list(range(72,104)):raise RuntimeError('relation schedule mismatch')
 for g in groups:
  a,b=g['states'];
  if a['prompt_token_ids']!=a['source_prompt_token_ids']:raise RuntimeError('original input changed')
  if a['target_action']['answer']!=g['original_answer'] or b['target_action']['answer']!=g['contrast_answer']:raise RuntimeError('pair target mismatch')
  if a['messages_after']!=b['messages_after'] and not all(x==y for j,(x,y) in enumerate(zip(a['messages_after'],b['messages_after'])) if x['role']!='user'):
   raise RuntimeError('history changed beyond initial user question')
 # Analytical objective yields .25 mean state gradient and .125 per each paired relation action.
 import torch
 x=torch.tensor([-1.,-2.],requires_grad=True);(-.25*x.mean()).backward()
 if not torch.allclose(x.grad,torch.tensor([-.125,-.125])):raise RuntimeError('state loss normalization failure')
 y=torch.tensor([-1.,-3.],requires_grad=True);(-(.25/2)*y.mean()).backward()
 if not torch.allclose(y.grad,torch.tensor([-.0625,-.0625])):raise RuntimeError('relation pair mean loss failure')
 all_actions=[a for r in rows(Z/'training-inputs.jsonl') for a in r['actions']]+amstates+states
 for a in all_actions:
  if len(a['prompt_token_ids'])!=len(a['prompt_attention_mask']):raise RuntimeError('mask length mismatch')
 if len(rows(OUT/'evaluation-cases.jsonl'))!=64:raise RuntimeError('dev set count mismatch')
 if am.sha(SOURCE/'adapter_model.safetensors')!=EXPECTED:raise RuntimeError('Goal Z hash mismatch')
 # Regression checks for model/client/product path without loading model.
 c,_,r=q.make_client(SOURCE,deterministic=True,product_config=PC,model_path=BASE,corpus=Z/'corpus-dev.json')
 if c._model is not None or not r.definitions():raise RuntimeError('CPU preflight unexpectedly loaded weights or lost tools')
 for k,v in PC.items():
  attr={'thinking':'enable_thinking'}.get(k,k)
  if hasattr(c.config,attr) and getattr(c.config,attr)!=v:raise RuntimeError('product config drift: '+k)
 write_once(OUT/'cpu-preflight.json',dict(passed=True,manifest_sha256=sha(OUT/'manifest.json'),groups=8,derived_states=16,am_states=34,z_trajectories=128,
  dev=64,complete_probes=24,partial_probes=8,relation_updates=32,am_state_updates=34,loss_scaling_checked=True,mask_checked=True,
  product_factory_config_checked=True,goal_z_adapter_sha256=EXPECTED,gpu_generations=0,optimizer_updates=0))
 report=f"""# Goal AO 冻结 GPU 前提

Manifest SHA-256：`{sha(OUT/'manifest.json')}`  
起点 Goal Z SHA-256：`{EXPECTED}`；AA发布记录、模型/语料/tokenizer与代码身份在manifest中逐项冻结。CPU预检通过，GPU尚未开始。

训练材料：128条Goal Z原轨迹、34个AM状态、8组/16个派生充分证据状态。材料按case_id+state_id冻结；每个原问题和反事实问题均使用同一历史正文，逐条保存引用、token位置及原始消息。评价ID/问题已排除。见 `evidence-admission.json` 与 `ao-groups.json`。

A每关系组重复原问答两次；B使用原问答与关系对照问答。两臂相同数据数量、顺序、损失系数和128更新。`L=L轨迹+0.25 L_AM+0.25 L组`，组内两个动作均值NLL取平均；当前动作JSON与终止符监督，其余prompt/历史/正文mask。优化器LR5e-6、AdamW(0.9,0.999)/eps1e-8/wd0、clip1、LoRA-only、单卡BF16不量化。

条件统一：Goal Q、direct-v090、first-json-v1、thinking=false、legacy、BM25-v2/top-3、greedy、16384/2048、4调用/3搜索。

seed42两臂各评估64题真实搜索、24完整证据、8缺链动作和16训练拟合状态；AM普通SFT先检查同16状态。探针不继续搜索。seed43只在seed42 B满足PLAN.md全部冻结门槛时执行。总额度最多4训练/512更新、256真实轨迹、1232生成、768搜索；第一阶段最多2训练/256更新、128轨迹、624生成、384搜索。新确认0，不发布。

GPU、数值、身份、哈希或基础设施失败保存现场并停止；不自动重试或改变条件。GPU执行已获用户授权，不再重复确认。
"""
 (OUT/'PRE_GPU_REVIEW.md').write_text(report,encoding='utf-8',newline='\n')
 print(json.dumps(dict(cpu_preflight=True,manifest_sha256=sha(OUT/'manifest.json'),gpu_calls=0),ensure_ascii=False))

def freeze_materials():
 """Seal the one-pass primary semantic review before CPU/GPU work."""
 cfg=read(OUT/'manifest.json');review=read(OUT/'evidence-admission.json')
 if len(review)!=8:raise RuntimeError('semantic review must cover exactly the 8 frozen groups')
 for r in review:
  decision=r.get('primary_adjudication',{})
  if decision.get('status')!='pass' or not decision.get('reason') or not decision.get('supported_quotes'):
   raise RuntimeError('every group needs a source-grounded primary adjudication: '+r['case_id'])
  if r.get('eval_isolated') is not True or r.get('derived_state_not_autonomous') is not True:
   raise RuntimeError('training/evaluation boundary invalid: '+r['case_id'])
 cfg['output_hashes']={name:sha(OUT/name) for name in ('ao-groups.json','derived-states.jsonl','evidence-admission.json','frozen-probes.jsonl','training-fit-probes.jsonl','evaluation-cases.jsonl')}
 cfg['materials_adjudicated']=True;cfg['adjudication_sha256']=sha(OUT/'evidence-admission.json')
 tmp=OUT/'manifest.json.tmp';tmp.write_text(json.dumps(cfg,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8');tmp.replace(OUT/'manifest.json')
 digest=sha(OUT/'manifest.json')
 approval=OUT/'gpu-approval.json';approval.write_text(json.dumps(dict(human_authorized=True,manifest_sha256=digest,
  authorization_basis='用户已明确授权Goal AO GPU执行并说明以后不需再次确认',scope='frozen AO plan; seed43 conditional on frozen gate'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(materials_adjudicated=True,manifest_sha256=digest,groups=len(review)),ensure_ascii=False))

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True)
 s.add_parser('prepare');s.add_parser('freeze-materials');s.add_parser('cpu-preflight')
 t=s.add_parser('train');t.add_argument('--seed',type=int,choices=[42,43],required=True);t.add_argument('--arm',choices=['A','B'],required=True)
 e=s.add_parser('evaluate');e.add_argument('--model',choices=['42-A','42-B','43-A','43-B'],required=True)
 pr=s.add_parser('probe');pr.add_argument('--model',choices=['AM','42-A','42-B','43-A','43-B'],required=True);pr.add_argument('--kind',choices=['complete','partial','training-fit'],required=True)
 a=p.parse_args()
 try:
  if a.cmd=='prepare':prepare()
  elif a.cmd=='freeze-materials':freeze_materials()
  elif a.cmd=='cpu-preflight':cpu_preflight()
  else:
   frozen()
   if a.cmd=='train':train(a.seed,a.arm)
   elif a.cmd=='evaluate':evaluate(a.model)
   else:probe(a.model,a.kind)
 except BaseException as exc:
  if a.cmd=='prepare' and OUT.exists():
   append(OUT/'preparation-attempts.jsonl',dict(stage='prepare',error_type=type(exc).__name__,error=str(exc),time=time.time(),gpu_generations=0,updates=0,searches=0))
  if OUT.exists() and (OUT/'manifest.json').exists() and not (OUT/'experiment-failure.json').exists():
   write_once(OUT/'experiment-failure.json',dict(stage=a.cmd,error_type=type(exc).__name__,error=str(exc),time=time.time(),updates=counts('updates'),generations=counts('generations'),searches=counts('searches')))
  raise
if __name__=='__main__':main()
