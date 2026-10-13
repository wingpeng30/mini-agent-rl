"""Frozen staged relation-review experiment. No training or external services."""
import argparse
import asyncio
from collections import Counter
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import time
from uuid import uuid4

from mini_agent_rl.domain import Message, ToolCall
from mini_agent_rl.evaluation import answer_metrics
from mini_agent_rl.model import LocalModelConfig, TransformersModelClient
from mini_agent_rl.model.final_review import FinalReview, review_messages, validate_review
from mini_agent_rl.prompts import system_prompt
from mini_agent_rl.retrieval.local import config, sha
from mini_agent_rl.web.server import DemoService, ROOT, read
from mini_agent_rl.tools import ToolRegistry
import run_v133_goal_q_search as q

SUPPLEMENT = Path('E:/AgentLightningAssets/runs/retrieval-v1/supplement-reviewed-64-7011e1801e974f0eb055256abddbeab9')
PRIOR = Path('E:/AgentLightningAssets/runs/retrieval-v1/agent-paired-dev-29c9988ce7ca4dac882a3dbb73dd5c9f')
CASES = ('musique:2hop__82766_67668', 'musique:2hop__219456_87295', '2wiki:2b8703320bde11eba7f7acde48001122')
ARMS = {'A': 'off', 'B': 'generic-review-v1', 'C': 'relation-review-v1'}


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def append(path, obj):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(obj, ensure_ascii=False) + '\n')


def prefix(row):
    messages = [Message(role='system', content=system_prompt(False, 'direct-v090')),
        Message(role='user', content=row['question'])]
    for step in row['trace']:
        if step.get('action') == 'answer':
            break
        if 'observation' not in step:
            raise ValueError('Cannot reconstruct an incomplete search prefix')
        call = ToolCall(name='search', arguments=dict(query=step['query']))
        payload = {k: v for k, v in step['observation'].items() if k != 'latency_ms'}
        messages.extend([Message(role='assistant', content=json.dumps(dict(action='search', query=step['query']), ensure_ascii=False), tool_calls=[call]),
            Message(role='tool', content=json.dumps(payload, ensure_ascii=False), tool_call_id=call.id)])
    return messages


def cpu_client(pc, base):
    from transformers import AutoProcessor
    client = TransformersModelClient(LocalModelConfig(model_id=base, max_input_tokens=pc['max_input_tokens'],
        max_new_tokens=pc['max_new_tokens'], enable_thinking=False, device='cpu',
        action_completion=pc['action_completion'], observation_encoding=pc['observation_encoding'],
        observation_presentation=pc['observation_presentation'], action_constraint='none', recompute_old_logprobs=False))
    client._processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    client._model = SimpleNamespace(device='cpu')
    return client


def prepare():
    folder = Path(config()['run_dir']) / ('relation-review-' + uuid4().hex)
    folder.mkdir(parents=True)
    settings = read(ROOT / 'configs/demo-web.json')
    client = cpu_client(settings['product'], settings['base_model'])
    source_cases = {r['id']: r for r in rows(SUPPLEMENT / 'natural.jsonl')}
    previous = [r for r in rows(PRIOR / 'ledger.jsonl') if r['case_id'] in CASES]
    expected = {(r['question'], r['retriever'], r['elapsed_seconds']): r for r in previous}
    originals = {}
    for path in Path('E:/AgentLightningAssets/runs/web').glob('*.json'):
        saved = read(path)
        result = saved.get('result', {})
        key = (result.get('question'), result.get('retrieval', {}).get('retriever'), result.get('elapsed_seconds'))
        if key in expected:
            originals[expected[key]['case_id'], expected[key]['retriever']] = dict(path=str(path), sha256=sha(path), row=saved['raw_trace'])
    if len(originals) != 6:
        raise ValueError('Missing original raw traces: ' + str(len(originals)))
    engine = DemoService(settings, runtime_dir=folder).retrieval_engine()
    states, preflight, bridge_diagnostics = [], [], []
    for cid in CASES:
        case = source_cases[cid]
        for mode in ('bm25-v2', 'hybrid-rrf-v1'):
            original = originals[cid, mode]
            row = original['row']
            messages = prefix(row)
            actual = client._inputs(messages, 'exact_state_preflight')
            if actual['input_ids'][0].tolist() != row['trace'][-1]['prompt_token_ids']:
                write(folder / 'failure.json', dict(stage='cpu', case_id=cid, retriever=mode, reason='Original input token mismatch'))
                raise RuntimeError('Original input token mismatch: ' + cid + ' ' + mode)
            preflight.append(dict(case_id=cid, retriever=mode, exact_tokens=True, original_path=original['path'], original_sha256=original['sha256']))
            bridge_diagnostics.append(dict(case_id=cid, retriever=mode, queries=[s.get('query') for s in row['trace'] if s.get('action')=='search'],
                original_answer=row['answer'], truncated=any(s.get('context_usage', {}).get('prompt_truncated') for s in row['trace'])))
            if mode == 'bm25-v2':
                states.append(dict(case_id=cid, variant='original_bm25', comparable=True, oracle=False, messages=[m.model_dump() for m in messages]))
                continue
            for variant in ('original_hybrid', 'id_order', 'reverse_order', 'support_only', 'replace_distractors'):
                modified = copy.deepcopy(messages)
                comparable, reason = True, None
                for index, message in enumerate(modified):
                    if message.role != 'tool':
                        continue
                    payload = json.loads(message.content)
                    docs = payload['result']
                    support = set(case['support_document_ids'])
                    if variant == 'id_order': docs = sorted(docs, key=lambda d: d['id'])
                    elif variant == 'reverse_order': docs = docs[::-1]
                    elif variant == 'support_only': docs = [d for d in docs if d['id'] in support]
                    elif variant == 'replace_distractors':
                        query = json.loads(modified[index-1].content)['query']
                        alternatives = [d for d in engine.search(query, 'bm25-v2') if d['id'] not in support and d['id'] not in {x['id'] for x in docs if x['id'] in support}]
                        slots = sum(d['id'] not in support for d in docs)
                        if len(alternatives) < slots:
                            comparable, reason = False, 'Same-query BM25 top3 has too few non-support documents'
                            break
                        iterator = iter(alternatives)
                        docs = [d if d['id'] in support else next(iterator) for d in docs]
                    payload['result'] = docs
                    modified[index] = message.model_copy(update={'content': json.dumps(payload, ensure_ascii=False)})
                states.append(dict(case_id=cid, variant=variant, comparable=comparable, reason=reason,
                    oracle=variant=='support_only', messages=[m.model_dump() for m in modified]))
    for filename in ('natural.jsonl', 'constructed.jsonl', 'manifest.json'):
        (folder / filename).write_bytes((SUPPLEMENT / filename).read_bytes())
    dev = ROOT / 'runs/v133-goal-z-multihop-sft/selected-dev.jsonl'
    (folder / 'development.jsonl').write_bytes(dev.read_bytes())
    write(folder / 'states.json', states)
    write(folder / 'preflight.json', dict(passed=True, exact_inputs=preflight, bridge_diagnostics=bridge_diagnostics))
    tracked = [Path(__file__), ROOT/'src/mini_agent_rl/model/final_review.py', ROOT/'scripts/run_v133_goal_q_search.py', ROOT/'configs/demo-web.json']
    manifest = dict(kind='Exposed relation-review diagnosis and regression; no held-out claim', product=settings,
        files={str(p): sha(p) for p in tracked}, frozen_data={name:sha(folder/name) for name in ('states.json','natural.jsonl','constructed.jsonl','development.jsonl','manifest.json')},
        caps=dict(generations=1406, searches=888, trajectories=296, probe_generations=54), training=False, arms=ARMS,
        parent='goal-z', gpu_restart_required=False)
    write(folder / 'experiment-manifest.json', manifest)
    print(json.dumps(dict(prepared=str(folder), original_token_inputs_verified=6, states=len(states), incomparable=sum(not s['comparable'] for s in states))), flush=True)


def metrics(answer, case):
    values = [answer_metrics(answer, a) for a in [case['answer']] + case.get('answer_aliases', [])]
    return {key:max(v[key] for v in values) for key in values[0]}


def summarize(values):
    result = {}
    for arm in sorted({r['arm'] for r in values}):
        group = [r for r in values if r['arm']==arm]
        result[arm] = dict(count=len(group), correct=sum(r['normalized_exact_match'] for r in group),
            em=sum(r['normalized_exact_match'] for r in group)/len(group), f1=sum(r['token_f1'] for r in group)/len(group),
            all_support_seen=sum(r['all_support_seen'] for r in group)/len(group),
            reasons=dict(Counter(r['reason'] for r in group)), actual_model_calls=sum(r['model_calls'] for r in group),
            elapsed_seconds=sum(r['elapsed_seconds'] for r in group),
            auxiliary_seconds=sum(s.get('usage',{}).get('final_review',{}).get('elapsed_seconds',0) for r in group for s in r['trace']))
        for target,field in (('input_tokens','prompt_token_ids'),('output_tokens','action_token_ids')):
            total=0
            for row in group:
                for step in row['trace']:
                    review=step.get('usage',{}).get('final_review')
                    total+=sum(len(review[k].get(field) or []) for k in ('candidate','response')) if review else len(step.get(field) or [])
            result[arm][target]=total
        result[arm]['invalid_reviews']=sum(bool(s.get('usage',{}).get('final_review')) and not s['usage']['final_review']['accepted'] for r in group for s in r['trace'])
    import random
    if 'C' in result and 'A' in result:
        pairs={}
        for row in values: pairs.setdefault(row['case_id'],{})[row['arm']]=row
        deltas=[p['C']['normalized_exact_match']-p['A']['normalized_exact_match'] for p in pairs.values()]
        rng=random.Random(42)
        draws=sorted(sum(rng.choices(deltas,k=len(deltas)))/len(deltas) for _ in range(2000))
        result['paired_C_minus_A']=dict(improved=sum(d>0 for d in deltas),regressed=sum(d<0 for d in deltas),
            unchanged=sum(d==0 for d in deltas),delta=sum(deltas)/len(deltas),bootstrap_95=[draws[50],draws[1949]])
    return result


async def experiment(folder):
    manifest = read(folder/'experiment-manifest.json')
    for path, digest in manifest['files'].items():
        if sha(path)!=digest: raise RuntimeError('Frozen code/config changed: '+path)
    for name,digest in manifest['frozen_data'].items():
        if sha(folder/name)!=digest: raise RuntimeError('Frozen data changed: '+name)
    if (folder/'generation-ledger.jsonl').exists(): raise FileExistsError('No automatic experiment reruns')
    service = DemoService(manifest['product'], runtime_dir=folder)
    service.load_sync('goal-z')
    pc = service.config['product']
    used = dict(generations=0, searches=0, trajectories=0, probe_generations=0)
    stage, tag = 'probes', {}
    async def generate(messages, tools, **kwargs):
        if used['generations']>=1406 or (stage=='probes' and used['probe_generations']>=54):
            raise RuntimeError('Generation quota exceeded')
        used['generations']+=1
        if stage=='probes':used['probe_generations']+=1
        started=time.monotonic()
        result=await service.wrapped.generate(messages,tools,**kwargs)
        append(folder/'generation-ledger.jsonl',dict(stage=stage,tag=tag,elapsed_seconds=time.monotonic()-started,
            input_messages=[m.model_dump() for m in messages],response=result.model_dump()))
        if any(result.usage.get(k) for k in ('prompt_truncated','lossy_body_trimmed','documents_deleted','history_dropped','context_budget_error')):
            raise RuntimeError('Experiment refuses clipped inputs')
        return result
    report=dict(manifest=manifest,used=used,accepted=False,training=False)
    try:
        cases={r['id']:r for r in rows(folder/'natural.jsonl')}
        probes=[]
        for state in read(folder/'states.json'):
            if not state['comparable']:
                append(folder/'probe-ledger.jsonl',dict(**state,skipped=True));continue
            messages=[Message(**m) for m in state['messages']]
            tag=dict(case_id=state['case_id'],variant=state['variant'],arm='A')
            candidate=await generate(messages,[],temperature=0.,seed=42)
            case=cases[state['case_id']]
            item=dict(case_id=state['case_id'],variant=state['variant'],oracle=state['oracle'],A=dict(answer=candidate.content,**metrics(candidate.content,case)))
            for arm in ('B','C'):
                if candidate.parse_error or candidate.tool_call:
                    item[arm]=dict(skipped=True,reason='No valid answer candidate',normalized_exact_match=0);continue
                prepared,_=review_messages(service.client,messages,candidate,ARMS[arm])
                tag=dict(case_id=state['case_id'],variant=state['variant'],arm=arm)
                reviewed=await generate(prepared,[],temperature=0.,seed=42)
                from mini_agent_rl.model.final_review import actual_messages
                from mini_agent_rl.model.grounded_action_audit import visible_documents
                visible=visible_documents([m.model_dump() for m in actual_messages(service.client,prepared,'probe_review_audit')])
                audit=validate_review(reviewed,visible,ARMS[arm])
                selected=reviewed if audit['valid'] else candidate
                item[arm]=dict(answer=selected.content,audit=audit,raw=reviewed.raw_text,**metrics(selected.content,case))
            probes.append(item);append(folder/'probe-ledger.jsonl',item)
            print(json.dumps(dict(stage='probes',case_id=state['case_id'],variant=state['variant'],A=item['A']['normalized_exact_match'],C=item.get('C',{}).get('normalized_exact_match'))),flush=True)
        repairs=sum(r['variant']=='original_hybrid' and not r['A']['normalized_exact_match'] and r.get('C',{}).get('normalized_exact_match')==1 and r['C'].get('audit',{}).get('valid') for r in probes)
        report['probe_repairs']=repairs
        if repairs<2:
            report['stop_reason']='Probe gate failed: relation review repaired fewer than two original hybrid errors'
            return
        async def run_set(filename,arms,label):
            nonlocal stage,tag
            stage=label;values=[]
            class Registry:
                def __init__(self):
                    self.registry=ToolRegistry();self.registry.register(service.search_tool('goal-z-dev','hybrid-rrf-v1'))
                def definitions(self):return self.registry.definitions()
                async def execute(self,*args):
                    if used['searches']>=888:raise RuntimeError('Search quota exceeded')
                    used['searches']+=1
                    return await self.registry.execute(*args)
            wrapped=SimpleNamespace(base=service.client,generate=generate)
            for i,case in enumerate(rows(folder/filename)):
                order=arms[i%len(arms):]+arms[:i%len(arms)]
                for arm in order:
                    if used['trajectories']>=296:raise RuntimeError('Trajectory quota exceeded')
                    used['trajectories']+=1;tag=dict(case_id=case['id'],arm=arm)
                    pipeline=None if arm=='A' else FinalReview(ARMS[arm],generate,service.client)
                    value=await q.rollout(case['question'],wrapped,Registry(),sample=False,seed=42,product_config=pc,decision_pipeline=pipeline)
                    if value['model_calls']>(4 if arm=='A' else 5) or value['searches']>3:raise RuntimeError('Per-case budget violated')
                    found={d['title'] for s in value['trace'] for d in s.get('observation',{}).get('result',[])}
                    value=dict(case_id=case['id'],arm=arm,stage=stage,reference=case['answer'],all_support_seen=int(set(case['support_titles']).issubset(found)),**metrics(value['answer'],case),**value)
                    values.append(value);append(folder/'trajectory-ledger.jsonl',value)
                print(json.dumps(dict(stage=stage,completed=i+1,total=len(rows(folder/filename)))),flush=True)
            return summarize(values)
        report['natural']=await run_set('natural.jsonl',['A','B','C'],'natural')
        scores=report['natural']
        def no_new_bad(candidate,baseline):return all(candidate['reasons'].get(k,0)<=baseline['reasons'].get(k,0) for k in ('invalid_action','repeated_action'))
        if scores['C']['correct']<scores['A']['correct']+2 or scores['C']['correct']<scores['B']['correct']+1 or not no_new_bad(scores['C'],scores['A']):
            report['stop_reason']='Natural gate failed; regression stages not run';return
        report['development']=await run_set('development.jsonl',['A','C'],'development')
        report['constructed']=await run_set('constructed.jsonl',['A','C'],'constructed')
        report['accepted']=all(report[group]['C']['correct']>=report[group]['A']['correct'] and no_new_bad(report[group]['C'],report[group]['A']) for group in ('development','constructed'))
        report['stop_reason']='All stages passed' if report['accepted'] else 'Regression gate failed'
    except Exception as exc:
        write(folder/'failure.json',dict(stage=stage,error=repr(exc),used=used));report['stop_reason']='Infrastructure or integrity failure';raise
    finally:
        report['complete']=not (folder/'failure.json').exists()
        write(folder/'report.json',report)
        service.release_model()
        print(json.dumps(dict(report=str(folder/'report.json'),accepted=report['accepted'],stop_reason=report.get('stop_reason'),used=used)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.run:asyncio.run(experiment(args.run))
    else:prepare()
