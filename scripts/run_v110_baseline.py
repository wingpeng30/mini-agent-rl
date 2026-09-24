"""Independent100 baseline with fixed full-context corpus and per-task checkpoints."""
import argparse,asyncio,hashlib,json,os,signal,subprocess,time
from pathlib import Path
from mini_agent_rl.evaluation import load_hotpot_environment
from mini_agent_rl.model.local import LocalModelConfig,TransformersModelClient
from mini_agent_rl.reward import default_reward
from mini_agent_rl.runtime import AgentRunner
from mini_agent_rl.storage import SQLiteStore
from mini_agent_rl.tools import ToolRegistry,SearchTool

def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--model-path',required=True);p.add_argument('--adapter-path',required=True);p.add_argument('--db',required=True);p.add_argument('--report',required=True);p.add_argument('--limit',type=int,default=100);args=p.parse_args()
    env=load_hotpot_environment(args.data,task_limit=100,corpus_limit=100)
    fp=env.fingerprint('bm25-v2','compact-v3');fp['adapter_sha256']=hashlib.sha256((Path(args.adapter_path)/'adapter_model.safetensors').read_bytes()).hexdigest()
    manifest=Path(args.db+'.manifest.json')
    if Path(args.db).exists() and not manifest.exists():raise RuntimeError('unmanifested_db')
    if manifest.exists() and json.loads(manifest.read_text())!=fp:raise RuntimeError('changed_environment')
    write(manifest,fp);store=SQLiteStore(args.db);existing=store.list_rollouts()
    if len({r.task_id for r in existing})!=len(existing):raise RuntimeError('duplicate_rollouts')
    if any(r.status.value not in {'SUCCEEDED','FAILED'} for r in existing):raise RuntimeError('incomplete_prior_rollout_requires_reconciliation')
    if any(r.task_id not in set(env.task_ids) for r in existing):raise RuntimeError('foreign_task_in_database')
    done={r.task_id for r in existing}
    registry=ToolRegistry();registry.register(SearchTool(env.corpus,'bm25-v2'))
    client=TransformersModelClient(LocalModelConfig(model_id=args.model_path,adapter_path=args.adapter_path,max_input_tokens=2048,max_new_tokens=128,enable_thinking=False,timeout=600,gradient_checkpointing=False,observation_encoding='compact-v3'))
    runner=AgentRunner(client,registry,default_reward(),store,max_steps=3,max_concurrency=1,temperature=0,base_seed=42,anti_repeat_prompt=False,environment_fingerprint=fp,enable_thinking=False,prompt_variant='direct-v090')
    rows=[];start=time.perf_counter()
    async def go():
        for task in env.tasks[:args.limit]:
            if task.id in done:continue
            # A process-level alarm terminates thread-backed generation on timeout.
            signal.alarm(660)
            group=await runner.run_group(task,1)
            signal.alarm(0)
            rollout=store.get_rollout(group.rollout_ids[0]);stats=client.gpu_memory_stats()
            raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
            mem=[float(line.split(',')[1])/1024 for line in raw.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
            if not mem or max(mem)>21:raise RuntimeError('device_memory_gate')
            rows.append({'task_id':task.id,'rollout_id':rollout.id,'termination':rollout.termination_reason,'memory':stats,'device_process_gib':max(mem)})
            write(args.report,{'version':'v110','completed':len(existing)+len(rows),'fingerprint':fp,'new_rows':rows,'elapsed_seconds':time.perf_counter()-start})
            print(f'{len(existing)+len(rows)}/{args.limit} {rollout.termination_reason}',flush=True)
            if rollout.error and any(s in rollout.error.lower() for s in ('out of memory','timeout','non-finite','cuda error')):raise RuntimeError(rollout.error)
    try:asyncio.run(go())
    finally:store.close()
if __name__=='__main__':main()
