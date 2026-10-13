"""One-shot Goal J training and paired evaluations from frozen CPU inputs."""
from __future__ import annotations
import argparse,gc,hashlib,importlib.metadata as md,json,math,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/v133-goal-j-transfer';MODEL=Path('/root/autodl-tmp/qwen_08b')
V118=ROOT/'backups/v118/extracted/runs/v118/formal/adapter';G_EVAL=ROOT/'runs/v133-goal-g-cpu/evaluation-frozen.jsonl';CAP=7200
sys.path.insert(0,str(ROOT))
from scripts import run_v133_goal_g_cpu as g
from scripts import run_v133_goal_i_microfit as ih
ih.OUT=OUT
EXPECTED={'torch':'2.5.1+cu124','transformers':'5.17.0','peft':'0.20.0','tokenizers':'0.23.2','safetensors':'0.8.0','datasets':'5.0.1','accelerate':'1.15.0'}
V118_SHA='ea894d235b6779bc8d15d1327a953773f946c347dfa7fa55ee3d2c2d5bd4a0a5'
G_SHA='f689f94b598a1c0d89f1b87f0b008c8a03158a7ab479028368285564244aad69'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def readl(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def writel(p,rs):Path(p).write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rs),encoding='utf-8')

def frozen_inputs():
 manifest=json.loads((OUT/'freeze-manifest.json').read_text(encoding='utf-8'))
 if manifest.get('status')!='cpu_ready' or manifest.get('training_count')!=16 or manifest.get('evaluation_count')!=16 or manifest.get('evaluation_isolation_verified') is not True or manifest.get('dual_review_complete') is not True:raise ValueError('goal_j_freeze_gate_failed')
 train=readl(OUT/'train-frozen.jsonl'); ev=readl(OUT/'evaluation-frozen.jsonl'); reg=readl(G_EVAL)
 if len(train)!=16 or len(ev)!=16 or len(reg)!=16:raise ValueError('frozen_set_count_mismatch')
 if sha(OUT/'train-frozen.jsonl')!=manifest['train_sha256'] or sha(OUT/'evaluation-frozen.jsonl')!=manifest['evaluation_sha256'] or sha(G_EVAL)!=G_SHA:raise ValueError('frozen_input_hash_mismatch')
 if set(r['case_id'] for r in train)&set(r['case_id'] for r in ev):raise ValueError('train_eval_case_overlap')
 if any(r.get('split')!='train' for r in train) or any(r.get('split')!='evaluation' for r in ev):raise ValueError('frozen_partition_tag_mismatch')
 return manifest,train,ev,reg

def preflight():
 import torch
 from transformers import AutoProcessor
 if torch.cuda.is_available():raise RuntimeError('cpu_only_preflight_requires_cuda_hidden')
 manifest,train,ev,reg=frozen_inputs();base=json.loads((ROOT/'data/evidence-sft-v133-goal-c-private/frozen/freeze-manifest-private.json').read_text(encoding='utf-8'))
 hashes=ih.verify_model_files({'actual_input_sha256':{str(MODEL/name):digest for group in ('tokenizer_sha256','model_runtime_files_sha256','model_weight_files_sha256') for name,digest in base['baseline'][group].items()}})
 if sha(V118/'adapter_model.safetensors')!=V118_SHA:raise ValueError('v118_adapter_hash_mismatch')
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True);cpu=[]
 for group,rows in (('train',train),('new_eval',ev),('goal_G_regression',reg)):
  for row in rows:
   c=g.cpu_check(processor,row)
   if c['prompt_sha256']!=row['cpu_check']['prompt_sha256']:raise ValueError(f'cpu_prompt_hash_mismatch:{group}:{row["case_id"]}')
   cpu.append({'case_id':row['case_id'],'split':group,'status':'pass','prompt_sha256':c['prompt_sha256'],'prompt_tokens':c['prompt_tokens'],'target_tokens':c['target_tokens'],'total_tokens':c['total_tokens']})
 code={str(p):sha(p) for p in (Path(__file__).resolve(),ROOT/'src/mini_agent_rl/training/sft.py',ROOT/'scripts/run_v133_goal_g_cpu.py',ROOT/'scripts/run_v133_goal_i_microfit.py',ROOT/'scripts/v130_review_protocol.py')}
 versions={n:md.version(n) for n in EXPECTED}
 if versions!=EXPECTED:raise ValueError(f'runtime_version_mismatch:{versions}')
 if (OUT/'run-preflight.json').exists():raise FileExistsError('preflight_already_frozen')
 result={'status':'preflight_pass','training_started':False,'inference_started':False,'gpu_used':False,'new_eval_exposed_to_trainer':False,'freeze_manifest_sha256':sha(OUT/'freeze-manifest.json'),'verified_assets':hashes,'v118_adapter_sha256':sha(V118/'adapter_model.safetensors'),'new_eval_sha256':sha(OUT/'evaluation-frozen.jsonl'),'goal_G_sha256':sha(G_EVAL),'cpu_cases':len(cpu),'cpu_checks_sha256':hashlib.sha256(json.dumps(cpu,sort_keys=True).encode()).hexdigest(),'code_sha256':code,'runtime_versions':versions,'gpu_hard_cap_seconds':CAP}
 write(OUT/'run-preflight.json',result);writel(OUT/'cpu-runtime-preflight.jsonl',cpu)
 print(json.dumps({'status':result['status'],'asset_files':len(hashes),'cpu_cases':len(cpu),'versions':versions}))

def run():
 frozen_inputs()
 if (OUT/'run-state.json').exists() or (OUT/'trajectories.jsonl').exists():raise FileExistsError('Goal_J_one_shot_no_retry')
 try:r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker'],cwd=ROOT,timeout=CAP,check=False)
 except subprocess.TimeoutExpired:
  state=json.loads((OUT/'run-state.json').read_text()) if (OUT/'run-state.json').exists() else {}
  write(OUT/'controller-stop.json',{'status':'hard_timeout','seconds':CAP,'completed':state.get('completed'),'last_status':state.get('status')});raise SystemExit('Goal_J_hard_timeout') from None
 if r.returncode:raise SystemExit(f'Goal_J_worker_failed:{r.returncode}')

def state_write(s):write(OUT/'run-state.json',s)
def worker():
 import torch
 from transformers import set_seed
 from safetensors.torch import load_file
 from mini_agent_rl.training.sft import build_lora_model,train_sft
 manifest,train,new_eval,reg=frozen_inputs()
 pf=json.loads((OUT/'run-preflight.json').read_text(encoding='utf-8'))
 if pf.get('status')!='preflight_pass' or pf.get('new_eval_exposed_to_trainer') is not False or pf.get('freeze_manifest_sha256')!=sha(OUT/'freeze-manifest.json'):raise ValueError('preflight_gate_failed')
 if sha(V118/'adapter_model.safetensors')!=V118_SHA:raise ValueError('v118_changed_after_preflight')
 if not torch.cuda.is_available():raise RuntimeError('authorized_GPU_unavailable')
 free,total=torch.cuda.mem_get_info()
 if total/1024**3<23.5 or free/1024**3<22:raise RuntimeError('GPU_memory_below_frozen_floor')
 state={'status':'running','cap_seconds':CAP,'gpu':torch.cuda.get_device_name(0),'completed':0,'planned':{'baseline_train':16,'baseline_G':16,'trained_train':16,'trained_G':16,'new_eval_v118':16,'new_eval_trained':16,'training_steps':20},'elapsed_seconds':0,'train_sha256':sha(OUT/'train-frozen.jsonl'),'evaluation_sha256':sha(OUT/'evaluation-frozen.jsonl'),'adapter_loads':{},'stop_reason':None,'new_eval_accessed':False}
 state_write(state);started=time.monotonic();mean_gen=None
 def timed_gen(model,proc,rows,arm,split,device):
  nonlocal mean_gen
  for row in rows:
   if time.monotonic()-started>=CAP:raise TimeoutError('hard_cap_before_generation')
   before=len(readl(OUT/'trajectories.jsonl')) if (OUT/'trajectories.jsonl').exists() else 0
   ih.generate(model,proc,row,arm,split,device,started);state['completed']+=1;state['elapsed_seconds']=round(time.monotonic()-started,3);state_write(state)
   if state['completed']==2:
    tr=readl(OUT/'trajectories.jsonl')[-2:];mean_gen=sum(x['generation_seconds'] for x in tr)/2
    projection=time.monotonic()-started+mean_gen*(96-2)+120
    state['speed_check']={'after_two_formal_generations':True,'mean_generation_seconds':round(mean_gen,3),'projected_generation_and_load_reserve_seconds':round(projection,3),'decision':'continue' if projection<=CAP else 'stop'};state_write(state)
    if projection>CAP:raise TimeoutError('initial_speed_projection_over_cap')
  set_seed(42)
 # v118 baseline. New evaluation rows are intentionally not loaded by Trainer and are not evaluated yet.
 model,proc=build_lora_model(str(MODEL),dtype='bfloat16',gradient_checkpointing=False,adapter_path=str(V118));model.eval()
 active=model.active_adapters() if callable(getattr(model,'active_adapters',None)) else [model.active_adapter]
 modules=sum(1 for _,m in model.named_modules() if hasattr(m,'lora_A') and 'default' in m.lora_A)
 state['adapter_loads']['v118_baseline']={'active':active,'lora_modules':modules,'dtype':str(next(model.parameters()).dtype)}
 if 'default' not in active or modules!=96:raise ValueError('v118_adapter_not_active_or_shape_mismatch')
 device=next(model.parameters()).device;timed_gen(model,proc,train,'v118','train',device);timed_gen(model,proc,reg,'v118','goal_G_regression',device)
 base_scores=ih.score_targets(model,proc,train,'v118',device,started);write(OUT/'target-token-scores-v118.json',base_scores)
 del model,proc;gc.collect();torch.cuda.empty_cache()
 elapsed=time.monotonic()-started;reserve=max(300,math.ceil(64*(mean_gen or 1.0)*1.5+180));budget=CAP-int(elapsed)-reserve
 if budget<=0:raise TimeoutError('no_training_budget_after_frozen_reserve')
 train_path=OUT/'train-frozen.jsonl';adapter_out=OUT/'adapter-final'
 config={'model_path':str(MODEL),'train_path':str(train_path),'validation_path':None,'output_dir':str(adapter_out),'max_steps':20,'epochs':20,'max_length':2176,'seed':42,'batch_size':1,'gradient_accumulation_steps':len(train),'gradient_checkpointing':False,'adapter_path':str(V118),'enable_thinking':False,'learning_rate':1e-4,'save_steps':20,'eval_steps':None,'warmup_ratio':0.05,'wall_clock_cap_seconds':budget}
 write(OUT/'training-config-effective.json',{'trainer_kwargs':config,'optimizer':'AdamW','weight_decay':0.0,'max_grad_norm':1.0,'scheduler':'linear','warmup_steps':1,'evaluation_enabled':False,'validation_path':None,'post_eval_reserve_seconds':reserve,'new_evaluation_path_passed_to_trainer':False})
 set_seed(42);metrics=train_sft(**config);write(OUT/'training-metrics.json',metrics)
 if metrics['global_step']!=20 or metrics['speed_check']['status']!='within_cap' or metrics['evaluation_enabled'] is not False:raise TimeoutError('20_step_training_or_no_eval_contract_failed')
 origin=load_file(str(V118/'adapter_model.safetensors'),device='cpu');trained=load_file(str(adapter_out/'adapter_model.safetensors'),device='cpu')
 if set(origin)!=set(trained):raise ValueError('adapter_tensor_keys_changed')
 diff=[trained[k].float()-origin[k].float() for k in origin];delta=math.sqrt(sum(float(torch.sum(x*x)) for x in diff));norm=math.sqrt(sum(float(torch.sum(x.float()*x.float())) for x in origin.values()))
 adapter_hash={'adapter_model_sha256':sha(adapter_out/'adapter_model.safetensors'),'changed_tensor_count':sum(bool(torch.any(x!=0)) for x in diff),'tensor_keys':len(diff),'delta_l2_norm':delta,'starting_weight_l2_norm':norm,'relative_delta_l2':delta/max(norm,1e-12)}
 write(OUT/'trained-adapter-hashes.json',adapter_hash);state['adapter_final_frozen']=True;state['adapter_sha256']=adapter_hash['adapter_model_sha256'];state_write(state)
 # Regress on Goal G and train only after final adapter is fixed.
 set_seed(42);model,proc=build_lora_model(str(MODEL),dtype='bfloat16',gradient_checkpointing=False,adapter_path=str(adapter_out));model.eval();device=next(model.parameters()).device
 active=model.active_adapters() if callable(getattr(model,'active_adapters',None)) else [model.active_adapter]
 state['adapter_loads']['trained_regression']={'active':active,'lora_modules':sum(1 for _,m in model.named_modules() if hasattr(m,'lora_A') and 'default' in m.lora_A),'dtype':str(next(model.parameters()).dtype)}
 timed_gen(model,proc,train,'goal_J_sft','train',device);timed_gen(model,proc,reg,'goal_J_sft','goal_G_regression',device)
 after_scores=ih.score_targets(model,proc,train,'goal_J_sft',device,started);write(OUT/'target-token-scores-trained.json',after_scores)
 del model,proc;gc.collect();torch.cuda.empty_cache()
 # Only now read/use the new evaluation set, after final adapter/config are frozen.
 new_eval=readl(OUT/'evaluation-frozen.jsonl');state['new_eval_accessed']=True;state_write(state)
 for arm,adapter in (('v118_new_eval',V118),('goal_J_sft',adapter_out)):
  if time.monotonic()-started>=CAP:raise TimeoutError('cap_before_new_eval')
  set_seed(42);model,proc=build_lora_model(str(MODEL),dtype='bfloat16',gradient_checkpointing=False,adapter_path=str(adapter));model.eval();device=next(model.parameters()).device
  state['adapter_loads'][arm]={'active':model.active_adapters() if callable(getattr(model,'active_adapters',None)) else [model.active_adapter],'lora_modules':sum(1 for _,m in model.named_modules() if hasattr(m,'lora_A') and 'default' in m.lora_A),'dtype':str(next(model.parameters()).dtype)}
  timed_gen(model,proc,new_eval,arm,'new_eval',device);del model,proc;gc.collect();torch.cuda.empty_cache()
 if len(readl(OUT/'trajectories.jsonl'))!=96:raise ValueError('trajectory_count_mismatch')
 state['status']='complete';state['completed']=96;state['elapsed_seconds']=round(time.monotonic()-started,3);state['new_eval_prompt_hashes_verified']=True;state_write(state)

def summarize():
 s=json.loads((OUT/'run-state.json').read_text(encoding='utf-8'))
 if s['status']!='complete':raise ValueError('cannot_summarize_incomplete')
 rows=readl(OUT/'trajectories.jsonl');by={(r['arm'],r['case_id']):r for r in rows}
 train=readl(OUT/'train-frozen.jsonl');ev=readl(OUT/'evaluation-frozen.jsonl');reg=readl(G_EVAL)
 arms={'train':('v118','goal_J_sft'),'goal_G_regression':('v118','goal_J_sft'),'new_eval':('v118_new_eval','goal_J_sft')};sets={'train':train,'goal_G_regression':reg,'new_eval':ev}
 results={}
 for split,(a,b) in arms.items():
  x=sets[split];left=[by[(a,r['case_id'])] for r in x];right=[by[(b,r['case_id'])] for r in x]
  if any(p['prompt_sha256']!=q['prompt_sha256'] for p,q in zip(left,right)):raise ValueError(f'paired_prompt_mismatch:{split}')
  def met(z):return {'n':len(z),'strict_em':sum(t['em'] for t in z),'mean_f1':sum(t['f1'] for t in z)/len(z),'protocol_valid':sum(t['protocol_valid'] for t in z)}
  lm={z['case_id']:z for z in left};rm={z['case_id']:z for z in right}
  results[split]={'v118':met(left),'goal_J_sft':met(right),'new_correct':[c for c in rm if not lm[c]['em'] and rm[c]['em']],'lost_correct':[c for c in rm if lm[c]['em'] and not rm[c]['em']],'retained_correct':[c for c in rm if lm[c]['em'] and rm[c]['em']],'changed_raw':[c for c in rm if lm[c]['raw_output']!=rm[c]['raw_output']]}
 token0={x['case_id']:x for x in json.loads((OUT/'target-token-scores-v118.json').read_text(encoding='utf-8'))};token1={x['case_id']:x for x in json.loads((OUT/'target-token-scores-trained.json').read_text(encoding='utf-8'))}
 summary={'status':'complete_exploratory_transfer','elapsed_seconds':s['elapsed_seconds'],'speed_check':s.get('speed_check'),'sets':results,'training_tokens':{k:{'before':token0[k],'after':token1[k]} for k in token0},'training_metrics':json.loads((OUT/'training-metrics.json').read_text(encoding='utf-8')),'adapter_sha256':s['adapter_sha256'],'train_sha256':sha(OUT/'train-frozen.jsonl'),'evaluation_sha256':sha(OUT/'evaluation-frozen.jsonl'),'trajectories_sha256':sha(OUT/'trajectories.jsonl'),'final_test_accessed':False,'sealed_independent_100_accessed':False,'additional_training':False,'confirmatory_claim':False}
 write(OUT/'summary.json',summary);print(json.dumps({'status':summary['status'],'elapsed_seconds':summary['elapsed_seconds'],'new_eval':results['new_eval'],'train':results['train']}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('preflight','run','worker','summarize'));a=p.parse_args();{'preflight':preflight,'run':run,'worker':worker,'summarize':summarize}[a.phase]()
