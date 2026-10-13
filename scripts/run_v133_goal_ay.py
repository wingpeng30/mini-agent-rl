"""AY bounded collection, no answer labels in production inputs."""
import argparse,asyncio,json,time,sys,os,tempfile
from pathlib import Path
import run_v133_goal_aw as aw
ap=aw.ap;OUT=ap.ROOT/os.environ.get('GOAL_AY_RUN_DIR','runs/v133-goal-ay-real-state-distillation-v2')
CAPS=dict(generation=1120,search=480,updates=256)
def write(p,obj):
    with p.open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def record_at(folder,kind,cid):
    used=sum(r['kind']==kind for r in ap.rows(folder/'usage.jsonl'))
    if used>=CAPS[kind]:raise RuntimeError('AY quota')
    ap.append(folder/'usage.jsonl',dict(kind=kind,case_id=cid,attempt=used+1,time=time.time()))
def self_test():
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        with (folder/'usage.jsonl').open('x',encoding='utf8'):pass
        if ap.rows(folder/'usage.jsonl')!=[]:raise RuntimeError('Empty initialization failed')
        record_at(folder,'generation','test');record_at(folder,'search','test')
        if [r['attempt'] for r in ap.rows(folder/'usage.jsonl')]!=[1,1]:raise RuntimeError('Independent counters failed')
        with (folder/'usage.jsonl').open('w',encoding='utf8') as f:
            for _ in range(CAPS['generation']):f.write('{"kind":"generation"}\n')
        try:record_at(folder,'generation','test')
        except RuntimeError:pass
        else:raise RuntimeError('Quota rejection failed')
    print('CPU ledger checks passed; no model loading or generation',flush=True)
def prepare():
    aw.frozen()
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir()
    source=ap.Z/'selected-train.jsonl';train=ap.rows(source);original=ap.rows(ap.Z/'training-inputs.jsonl')
    if len(train)!=128 or [r['id'] for r in train]!=[r['id'] for r in original]:raise RuntimeError('Original training order mismatch')
    cases=train[:32]
    self_test()
    write(OUT/'manifest.json',dict(parent_sha256=ap.AM_HASH,product=aw.PC,source=str(source),source_sha256=ap.sha(source),corpus_sha256=ap.sha(ap.Z/'corpus-train.json'),runner_sha256=ap.sha(Path(__file__)),case_ids=[r['id'] for r in cases],caps=CAPS,collection_cases=32,max_corrections=16,min_distinct_corrections=8,teacher='Qwen/Qwen3-4B-Instruct-2507',teacher_not_loaded=True,labels_not_in_product_input=True,prior_failed_run='runs/v133-goal-ay-real-state-distillation-v1',prior_generation=0))
    # Initialize the append-only attempt ledger before the first generation.
    with (OUT/'usage.jsonl').open('x',encoding='utf8'):pass
    write(OUT/'cpu-preflight.json',dict(passed=True,initialized_ledger=True,independent_counters=True,cap_rejection=True,no_model_loaded=True))
    print('AY CPU prepare passed: AM42-A, 32 fixed training cases, 16384/2048, 4 actions/3 searches, no auxiliary calls; no training until >=8 verified corrections.',flush=True)
def collect():
    aw.frozen()
    if (OUT/'experiment-failure.json').exists():raise RuntimeError('AY failure latch')
    m=ap.read(OUT/'manifest.json')
    if not ap.read(OUT/'cpu-preflight.json')['passed']:raise RuntimeError('AY missing CPU checks')
    if m['runner_sha256']!=ap.sha(Path(__file__)) or m['source_sha256']!=ap.sha(m['source']) or m['corpus_sha256']!=ap.sha(ap.Z/'corpus-train.json'):raise RuntimeError('AY frozen drift')
    if ap.rows(OUT/'usage.jsonl') or (OUT/'collected-trajectories.jsonl').exists():raise RuntimeError('AY refuse inference rerun')
    cases=ap.rows(m['source'])[:32]
    if [r['id'] for r in cases]!=m['case_ids']:raise RuntimeError('AY case identity drift')
    client,wrapped,registry=ap.q.make_client(ap.AMADAPTER,deterministic=True,product_config=aw.PC,model_path=ap.BASE,corpus=ap.Z/'corpus-train.json')
    base_generate=wrapped.generate;execute=registry.execute;active={}
    def record(kind):
        record_at(OUT,kind,active['id'])
    async def generate(*args,**kwargs):
        record('generation');r=await base_generate(*args,**kwargs);aw.guard(r);return r
    async def search(*args,**kwargs):record('search');return await execute(*args,**kwargs)
    wrapped.generate=generate;registry.execute=search
    for case in cases:
        active['id']=case['id']
        r=asyncio.run(ap.q.rollout(case['question'],wrapped,registry,sample=False,seed=42,product_config=aw.PC))
        if r['reason']=='tool_error':raise RuntimeError('AY search infrastructure')
        r.update(case_id=case['id'],diagnostic_gold=case['answer'],training_source=True)
        ap.append(OUT/'collected-trajectories.jsonl',r)
        print(json.dumps(dict(case_id=case['id'],answer=r['answer'],calls=r['model_calls'],searches=r['searches'])),flush=True)
    write(OUT/'collection-complete.json',dict(cases=32,generation=sum(r['kind']=='generation' for r in ap.rows(OUT/'usage.jsonl')),search=sum(r['kind']=='search' for r in ap.rows(OUT/'usage.jsonl'))))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['self-test','prepare','collect']);args=parser.parse_args()
    try:{'self-test':self_test,'prepare':prepare,'collect':collect}[args.stage]()
    except BaseException as e:
        if OUT.exists() and not (OUT/'experiment-failure.json').exists():write(OUT/'experiment-failure.json',dict(type=type(e).__name__,error=str(e),stop_whole_sequence=True,no_auto_rerun=True))
        raise
