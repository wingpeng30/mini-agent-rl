"""Isolated hybrid diagnostic runner. Never edits historical runs or web defaults."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from uuid import uuid4

from mini_agent_rl.domain import Message, ToolCall
from mini_agent_rl.evaluation import answer_metrics
from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
from mini_agent_rl.prompts import system_prompt
from mini_agent_rl.retrieval.local import ROOT, LocalRetrieval, config, sha
from mini_agent_rl.tools import ToolRegistry
from mini_agent_rl.tools.search import SearchTool
import run_v133_goal_q_search as q

PREVIOUS = Path('E:/AgentLightningAssets/runs/retrieval-v1/agent-paired-128-84217a89bd0b4af8b870156d1ba208bc')
CASES = Path('E:/AgentLightningAssets/runs/retrieval-v1/paired-128-frozen-0b6100db84b249cfa9a82c0861cb3852/cases.jsonl')
MODES = ('bm25-v2', 'hybrid-rrf-v1')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]

def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def append(path, obj):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(obj, ensure_ascii=False) + '\n')
        stream.flush()

def prefix(raw, target):
    messages = [Message(role='system', content=system_prompt(False, 'direct-v090')),
                Message(role='user', content=raw['question'])]
    for step in raw['trace'][:target]:
        if 'observation' not in step:
            raise ValueError('Cannot replay a terminated prefix')
        call = ToolCall(name='search', arguments={'query': step['query']})
        payload = {k: v for k, v in step['observation'].items() if k != 'latency_ms'}
        messages.extend([Message(role='assistant', content=json.dumps({'action':'search', 'query':step['query']}, ensure_ascii=False), tool_calls=[call]),
                         Message(role='tool', content=json.dumps(payload, ensure_ascii=False), tool_call_id=call.id)])
    return messages

def change_view(messages, view, engine=None):
    result = copy.deepcopy(messages)
    query = None
    for i, message in enumerate(result):
        if message.role == 'assistant':
            action = json.loads(message.content)
            query = action.get('query')
        if message.role != 'tool':
            continue
        payload = json.loads(message.content)
        if view == 'score-free-v1':
            payload['result'] = [{k:v for k,v in d.items() if k != 'score'} for d in payload['result']]
        elif view == 'bm25-order':
            ranks = {d['id']: rank for rank,d in enumerate(engine.bm25.search(query, len(engine.documents)))}
            payload['result'] = sorted(payload['result'], key=lambda d:(ranks.get(d['id'], len(ranks)),d['id']))
        elif view != 'legacy':
            raise ValueError(view)
        result[i] = message.model_copy(update={'content':json.dumps(payload, ensure_ascii=False)})
    return result

def metrics(answer, case):
    golds = [case['answer']] + case.get('answer_aliases', case.get('aliases', []))
    scores = [answer_metrics(answer or '', g) for g in golds]
    return {key:max(s[key] for s in scores) for key in scores[0]}

def prepare(folder):
    from run_local_relation_review import cpu_client
    folder.mkdir(parents=True, exist_ok=False)
    settings = read(ROOT/'configs/demo-web.json')
    client = cpu_client(settings['product'], settings['base_model'])
    cases = rows(CASES)
    previous = rows(PREVIOUS/'ledger.jsonl')
    paired = {c['id']:{r['retriever']:r for r in previous if r['case_id']==c['id']} for c in cases}
    natural = [c for c in cases if c['stratum'] != 'constructed_singlehop']
    regressed = [c['id'] for c in natural if paired[c['id']][MODES[0]]['normalized_exact_match'] > paired[c['id']][MODES[1]]['normalized_exact_match']]
    protected = [c['id'] for c in natural if paired[c['id']][MODES[0]]['normalized_exact_match'] < paired[c['id']][MODES[1]]['normalized_exact_match']]
    wrong = [c for c in natural if not paired[c['id']][MODES[1]]['normalized_exact_match']]
    keys = {(r['question'],r['retriever'],r['elapsed_seconds']):r for r in previous}
    originals = {}
    for path in Path('E:/AgentLightningAssets/runs/web').glob('*.json'):
        saved = read(path); r = saved.get('result',{})
        key = (r.get('question'),r.get('retrieval',{}).get('retriever'),r.get('elapsed_seconds'))
        if key in keys:
            old = keys[key]
            originals[old['case_id'],old['retriever']] = saved['raw_trace']
    engine = LocalRetrieval()
    documents = {d['title']:d for d in engine.documents}
    verified = 0
    states = []
    for case in wrong:
        old = paired[case['id']][MODES[1]]
        raw = originals[case['id'],MODES[1]]
        for target, step in enumerate(raw['trace']):
            messages = prefix(raw,target)
            if client._inputs(messages, 'diagnostic-preflight')['input_ids'][0].tolist() != step['prompt_token_ids']:
                raise RuntimeError('Token mismatch: '+case['id']+' step '+str(target))
            verified += 1
        append(folder/'audit-material.jsonl', dict(case=case, run=old, raw=raw,
            labeled_support_documents=[documents[t] for t in case['support_titles']],
            semantic_review_status='pending'))
    for cid in regressed + protected:
        case = next(c for c in cases if c['id']==cid)
        raw = originals[cid,MODES[1]]
        other = originals[cid,MODES[0]]
        divergence = next((i for i,(a,b) in enumerate(zip(raw['trace'],other['trace'])) if (a.get('query'),a.get('answer'))!=(b.get('query'),b.get('answer'))), len(raw['trace'])-1)
        for target in sorted({divergence,len(raw['trace'])-1}):
            messages = prefix(raw,target)
            if client._inputs(messages, 'diagnostic-preflight')['input_ids'][0].tolist()!=raw['trace'][target]['prompt_token_ids']:
                raise RuntimeError('Protected token mismatch: '+cid)
            for view in ('legacy','score-free-v1','bm25-order'):
                states.append(dict(case=case, cohort='regression' if cid in regressed else 'protection',target=target,
                    final=target==len(raw['trace'])-1, view=view, original_action=raw['trace'][target],
                    messages=[m.model_dump() for m in change_view(messages,view,engine)]))
    shutil.copy2(CASES,folder/'cases.jsonl')
    shutil.copy2(PREVIOUS/'ledger.jsonl',folder/'local-baseline.jsonl')
    for name in ('selected-train.jsonl','training-inputs.jsonl','corpus-train.json','corpus-dev.json'):
        shutil.copy2(ROOT/'runs/v133-goal-z-multihop-sft'/name,folder/name)
    write(folder/'states.json',states)
    cfg = config()
    write(folder/'retrieval-source.json',cfg)
    write(folder/'settings.json',settings)
    write(folder/'manifest.json',dict(kind='exposed diagnostic and regression',seed=42,gpu_seconds_cap=10800,
        wrong_natural=len(wrong),regressions=regressed,protections=protected,exact_original_inputs=verified,
        frozen_files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
        base_sha256=settings['base_weights_sha256'], adapter_sha256=settings['models']['goal-z']['sha256'],
        training_authorized_conditionally=True, semantic_audit_complete=False))
    print(json.dumps({'prepared':str(folder),'wrong':len(wrong),'verified':verified,'states':len(states)}),flush=True)

class ViewClient:
    def __init__(self, base, view):
        self.base, self.config, self.view = base, base.config, view
    @property
    def _context_records(self):
        return self.base._context_records
    async def generate(self,messages,*args,**kwargs):
        return await self.base.generate(change_view(messages,self.view),*args,**kwargs)

class RetrievalRegistry:
    def __init__(self, engine, mode):
        self.engine,self.mode=engine,mode
        self.registry=ToolRegistry()
        self.registry.register(SearchTool(engine.documents,'bm25-v2'))
    def definitions(self):
        return self.registry.definitions()
    async def execute(self,name,args):
        start=time.monotonic()
        return dict(ok=True,result=self.engine.search(args['query'],self.mode),latency_ms=(time.monotonic()-start)*1000)

def create_client(folder,adapter=None):
    settings=read(folder/'settings.json'); pc=settings['product']
    base=folder/'base-model'; goal=folder/'goal-z'
    client=TransformersModelClient(LocalModelConfig(model_id=str(base),adapter_path=str(adapter or goal),
        max_input_tokens=pc['max_input_tokens'],max_new_tokens=pc['max_new_tokens'],enable_thinking=False,
        device='cuda',dtype='bfloat16',load_in_4bit=False,deterministic=True,
        action_completion=pc['action_completion'],observation_encoding=pc['observation_encoding'],
        observation_presentation=pc['observation_presentation'],action_constraint='none',recompute_old_logprobs=False))
    return client,pc

class Budget:
    """Persist charged wall time, conservatively including idle loaded-model time."""
    def __init__(self,folder):
        self.path=folder/'gpu-budget.json'
        self.prior=read(self.path)['charged_seconds'] if self.path.exists() else 0
        self.start=time.monotonic()
    def check(self):
        spent=self.prior+time.monotonic()-self.start
        write(self.path,dict(charged_seconds=spent,cap_seconds=10800))
        # Reserve five minutes for one generation, synchronization and saving.
        if spent>=10500:
            raise RuntimeError('GPU time cap reached; no further candidate or restart')

def evaluate(folder,client,pc,engine,budget,label,modes,view='legacy'):
    cases=rows(folder/'cases.jsonl'); path=folder/(label+'-ledger.jsonl')
    done={(r['case_id'],r['retriever']) for r in rows(path)} if path.exists() else set()
    for index,case in enumerate(cases):
        for mode in (modes if index%2==0 else modes[::-1]):
            if (case['id'],mode) in done: continue
            budget.check()
            wrapped=q.SearchTextAliasClient(ViewClient(client,view))
            result=asyncio.run(q.rollout(case['question'],wrapped,RetrievalRegistry(engine,mode),sample=False,seed=42,product_config=pc))
            trace=result['trace']
            seen={d['title'] for s in trace for d in s.get('observation',{}).get('result',[])}
            result.update(case_id=case['id'],stratum=case['stratum'],retriever=mode,retrieval_view=view,reasoning_mode='off',
                **metrics(result['answer'],case),both_support_seen=set(case['support_titles']).issubset(seen),
                repeated_actions=sum(bool(s.get('repeated')) for s in trace),invalid_actions=sum(bool(s.get('parse_error')) for s in trace),
                input_tokens=sum(s.get('usage',{}).get('prompt_tokens',0) for s in trace),
                output_tokens=sum(s.get('usage',{}).get('completion_tokens',0) for s in trace),retrieval=engine.metadata(mode))
            if result['model_calls']>4 or result['searches']>3 or result['auxiliary_calls']!=0:
                raise RuntimeError('Budget invariant violated')
            append(path,result);budget.check()
            print(json.dumps(dict(stage=label,case=case['id'],mode=mode,em=result['normalized_exact_match'],elapsed=result['elapsed_seconds'])),flush=True)
    if len(rows(path))!=len(cases)*len(modes): raise RuntimeError('Incomplete paired evaluation')

def remote(folder):
    budget=Budget(folder)
    manifest=read(folder/'manifest.json')
    for name,expected in manifest['frozen_files'].items():
        if sha(folder/name)!=expected: raise RuntimeError('Asset drift: '+name)
    for name,expected in manifest['code_files'].items():
        if sha(folder.parent/'code'/name)!=expected:raise RuntimeError('Code drift: '+name)
    if sha(folder/'goal-z/adapter_model.safetensors')!=manifest['adapter_sha256']:raise RuntimeError('Adapter drift')
    if sha(folder/'base-model/model.safetensors-00001-of-00001.safetensors')!=manifest['base_sha256']:raise RuntimeError('Base drift')
    engine=LocalRetrieval(read(folder/'retrieval-remote.json'))
    client,pc=create_client(folder)
    try:
        evaluate(folder,client,pc,engine,budget,'goal-z',list(MODES))
        path=folder/'state-ledger.jsonl'
        done={(r['case_id'],r['target'],r['view']) for r in rows(path)} if path.exists() else set()
        for state in read(folder/'states.json'):
            key=(state['case']['id'],state['target'],state['view'])
            if key in done:continue
            budget.check()
            messages=[Message.model_validate(m) for m in state['messages']]
            started=time.monotonic()
            definitions=RetrievalRegistry(engine,'hybrid-rrf-v1').definitions()
            response=asyncio.run(q.SearchTextAliasClient(client).generate(messages,definitions,temperature=0.,seed=42+state['target']))
            if state['view']=='legacy' and response.prompt_token_ids!=state['original_action']['prompt_token_ids']:
                raise RuntimeError('Remote original-state token mismatch: '+key[0])
            action=dict(answer=response.content,query=response.tool_call.arguments.get('query') if response.tool_call else None)
            append(path,dict(case_id=key[0],target=key[1],view=key[2],cohort=state['cohort'],final=state['final'],
                **action,raw=response.raw_text,parse_error=response.parse_error,usage=response.usage,
                elapsed_seconds=time.monotonic()-started,prompt_token_ids=response.prompt_token_ids,
                action_token_ids=response.action_token_ids,**metrics(response.content,state['case'])))
            budget.check()
            print(json.dumps(dict(stage='state',case=key[0],target=key[1],view=key[2],answer=response.content)),flush=True)
        data=rows(path)
        # Final-answer states provide the preregistered repair/protection gate.
        paired={}
        for r in data:
            if r['final']:paired.setdefault(r['case_id'],{})[r['view']]=r
        repaired=sum(p['score-free-v1']['normalized_exact_match']>p['legacy']['normalized_exact_match'] for p in paired.values() if p['legacy']['cohort']=='regression')
        lost=sum(p['score-free-v1']['normalized_exact_match']<p['legacy']['normalized_exact_match'] for p in paired.values() if p['legacy']['cohort']=='protection')
        gate=dict(repaired=repaired,protection_losses=lost,passed=repaired>=2 and lost<=1)
        write(folder/'state-gate.json',gate)
        if gate['passed']:evaluate(folder,client,pc,engine,budget,'score-free', ['hybrid-rrf-v1'],'score-free-v1')
        write(folder/'remote-stage.json',dict(baseline_complete=True,state_complete=True,engineering_evaluated=gate['passed'],
            training_started=False,next_required='Semantic audit and conditional training gate'))
    finally:budget.check()

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','remote'));parser.add_argument('--folder',type=Path)
    args=parser.parse_args()
    folder=args.folder or Path(config()['run_dir'])/('hybrid-adaptation-'+uuid4().hex)
    try:
        (prepare if args.command=='prepare' else remote)(folder)
    except Exception as exc:
        if folder.exists():write(folder/'failure.json',dict(error=repr(exc),stage=args.command,complete=False))
        raise
