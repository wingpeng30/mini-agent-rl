"""Guarded fixed-internal v112 evaluation; semantic eligibility stays pending."""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path

from mini_agent_rl.domain import Message, Task
from mini_agent_rl.evaluation import answer_metrics
from mini_agent_rl.evidence_sft_v112 import validate_dataset, validate_source_path
from mini_agent_rl.tools.search import _BM25Index
try:
    from .build_v112 import ROOT, OUT, read, norm, digest, question
    from .train_v112 import sha256, SOURCE_SHA256, check_budget
    from .run_v111_baseline import model_config, tree_fingerprint, RUN_CONFIG, check_infrastructure_error
except ImportError:
    from build_v112 import ROOT, OUT, read, norm, digest, question
    from train_v112 import sha256, SOURCE_SHA256, check_budget
    from run_v111_baseline import model_config, tree_fingerprint, RUN_CONFIG, check_infrastructure_error


def require_new(path, protected=()):
    path = validate_source_path(path).resolve()
    if path.exists():
        raise ValueError('output_already_exists')
    for other in protected:
        other = Path(other).resolve()
        if path == other or path in other.parents or other in path.parents:
            raise ValueError('output_overlaps_input')
    return path


def load_inputs(gate_path, train_path, validation_path):
    paths = [validate_source_path(p) for p in (gate_path, train_path, validation_path)]
    gate = json.loads(paths[0].read_text(encoding='utf-8'))
    if gate.get('version') != 'v112' or gate.get('approved') is not True or gate.get('source_adapter_sha256') != SOURCE_SHA256:
        raise ValueError('approved_v112_data_gate_required')
    for name, path in zip(('train', 'validation'), paths[1:]):
        if gate.get('artifacts', {}).get(name, {}).get('sha256') != sha256(path):
            raise ValueError('data_hash_mismatch:'+name)
    check_budget(gate, paths[0])
    train, states = read(paths[1]), read(paths[2])
    checked = validate_dataset(train, states)
    if not checked['passed']:
        raise ValueError('data_gate_recheck_failed')
    manifest = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    ids = manifest['internal_ids']
    if len(ids) != 50 or len(set(ids)) != 50 or {r['task_id'] for r in states} != set(ids):
        raise ValueError('fixed_50_internal_coverage_required')
    if any(r['split'] != 'validation' for r in states):
        raise ValueError('validation_states_only')
    source = ROOT/'data/hotpot-agent-v041/train.jsonl'
    if sha256(source) != manifest['source_sha256']:
        raise ValueError('source_train_changed')
    by = {r['metadata']['task_id']: r for r in read(source)}
    if any(row['question'] != question(by[row['task_id']]) for row in states):
        raise ValueError('state_question_differs_from_frozen_source')
    tasks = [Task(id=tid, question=question(by[tid]), answer=by[tid]['metadata']['gold_answer'],
                  metadata=by[tid]['metadata']) for tid in ids]
    corpus_path = OUT/'train-corpus.json'
    corpus = json.loads(corpus_path.read_text(encoding='utf-8'))
    hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(p)
              for p in [*paths, source, OUT/'manifest.json', corpus_path]}
    return states, tasks, corpus, hashes


def normalize_prompt_tokens(value):
    """Accept only a flat token list or one explicitly batched token list.

    Token ID zero is valid. Booleans, negative IDs, empty inputs, tensors,
    multiple batches and deeper nesting are rejected rather than coerced.
    """
    if not isinstance(value,list) or not value:
        raise ValueError('invalid_prompt_token_shape')
    if len(value)==1 and isinstance(value[0],list):
        value=value[0]
    if not value or any(type(token) is not int or token<0 for token in value):
        raise ValueError('invalid_prompt_token_type_or_shape')
    return value


def score_state(row, response, index):
    if normalize_prompt_tokens(response.prompt_token_ids) != normalize_prompt_tokens(row['prompt_token_ids']):
        raise ValueError('prompt_token_replay_mismatch:'+row['state_hash'])
    legal = not response.parse_error and not response.usage.get('output_truncated', False)
    actual = None
    if response.tool_calls:
        call = response.tool_calls[0]
        query = call.arguments.get('query')
        legal = legal and len(response.tool_calls) == 1 and call.name == 'search' and isinstance(query, str) and bool(query.strip())
        if legal:
            actual = {'action': 'search', 'value': query}
    elif response.content.strip() and legal:
        actual = {'action': 'answer', 'value': response.content}
    else:
        legal = False
    target = row['target']
    action_match = bool(actual and actual['action'] == target['action'])
    metrics = answer_metrics(actual['value'] if actual else '', target['value'])
    previous = []
    for message in row['messages'][:-1]:
        if message['role'] == 'assistant':
            action = json.loads(message['content'])
            if action.get('action') == 'search':
                previous.append(norm(action['query']))
    hits, new = [], []
    repeated = bool(actual and actual['action'] == 'search' and norm(actual['value']) in previous)
    if actual and actual['action'] == 'search':
        hits = index.search(actual['value'], top_k=3)
        old = {d['id'] for d in row['evidence']}
        new = [d for d in hits if d['id'] not in old]
    return {'state_hash': row['state_hash'], 'task_id': row['task_id'], 'target': target,
            'raw_output': response.raw_text, 'parse_error': response.parse_error,
            'response': response.model_dump(), 'actual_action': actual, 'protocol_valid': bool(legal),
            'action_match': action_match, 'normalized_target_match': action_match and bool(metrics['normalized_exact_match']),
            'target_token_f1': metrics['token_f1'] if action_match else 0,
            'repeated_search': repeated, 'search_legal_nonrepeat': bool(legal and not repeated and actual and actual['action']=='search'),
            'retrieval_results': hits, 'new_retrieval_evidence': new,
            'semantic_audit': 'pending'}


def aggregate_states(rows):
    n = len(rows)
    answers=[r for r in rows if r['target']['action']=='answer']
    searches=[r for r in rows if r['target']['action']=='search']
    return {'count': n, 'protocol_rate': sum(r['protocol_valid'] for r in rows)/(n or 1),
            'action_match_rate': sum(r['action_match'] for r in rows)/(n or 1),
            'normalized_target_match_rate': sum(r['normalized_target_match'] for r in rows)/(n or 1),
            'answer_count':len(answers),'search_count':len(searches),
            'supported_answer_correct_rate':sum(r['normalized_target_match'] for r in answers)/(len(answers) or 1),
            'search_action_rate':sum(r['action_match'] for r in searches)/(len(searches) or 1),
            'search_legal_nonrepeat_rate':sum(r['search_legal_nonrepeat'] for r in searches)/(len(searches) or 1),
            'search_new_evidence_rate':sum(bool(r['new_retrieval_evidence']) for r in searches)/(len(searches) or 1),
            'semantic_audit': 'pending'}


def compare_reports(base_tasks, new_tasks, new_states):
    for report, mode in ((base_tasks,'tasks'),(new_tasks,'tasks'),(new_states,'states')):
        if report.get('mode') != mode or report.get('complete_coverage') is not True:
            raise ValueError('complete_evaluation_required')
    if base_tasks['fingerprint']['adapter_files'].get('adapter_model.safetensors') != SOURCE_SHA256:
        raise ValueError('baseline_must_be_frozen_v101')
    for report in (new_tasks,new_states):
        if report['fingerprint']['data'] != base_tasks['fingerprint']['data']:
            raise ValueError('different_data_fingerprints')
        for key in ('runner_config','model_files','code'):
            if report['fingerprint'][key] != base_tasks['fingerprint'][key]:
                raise ValueError('different_protocol:'+key)
        left, right = dict(report['fingerprint']['model_config']), dict(base_tasks['fingerprint']['model_config'])
        left.pop('adapter_path',None);right.pop('adapter_path',None)
        if left != right:
            raise ValueError('different_model_config')
    if new_tasks['fingerprint']['adapter_files'] != new_states['fingerprint']['adapter_files']:
        raise ValueError('candidate_adapter_mismatch')
    state_ids=[r['state_hash'] for r in new_states['rows']]
    if len(state_ids)!=len(set(state_ids)) or set(state_ids)!=set(new_states['expected_ids']):
        raise ValueError('incomplete_state_coverage')
    b = {r['task_id']:r for r in base_tasks['rows']};n = {r['task_id']:r for r in new_tasks['rows']}
    expected=set(base_tasks['expected_ids'])
    if len(expected)!=50 or set(b)!=expected or set(n)!=expected or len(b)!=len(base_tasks['rows']) or len(n)!=len(new_tasks['rows']):
        raise ValueError('paired_fixed_50_required')
    correct={k for k,r in b.items() if r['normalized_exact_match']==1}
    kept=sum(n[k]['normalized_exact_match']==1 for k in correct)
    preservation=kept/len(correct) if correct else None
    gains=sum(n[k]['normalized_exact_match']==1 and b[k]['normalized_exact_match']==0 for k in b)
    losses=sum(n[k]['normalized_exact_match']==0 and b[k]['normalized_exact_match']==1 for k in b)
    state=aggregate_states(new_states['rows'])
    base_f1=sum(r['token_f1'] for r in b.values())/50
    candidate_f1=sum(r['token_f1'] for r in n.values())/50
    valid_terminations={'answer','repeated_action','max_steps'}
    base_task_protocol=sum(r.get('termination') in valid_terminations for r in b.values())/50
    candidate_task_protocol=sum(r.get('termination') in valid_terminations for r in n.values())/50
    gates={'supported_answer_correct_90':state['answer_count']>0 and state['supported_answer_correct_rate']>=.9,
           'search_action_90':state['search_count']>0 and state['search_action_rate']>=.9,
           'protocol_98':state['protocol_rate']>=.98,
           'task_protocol_98':candidate_task_protocol>=.98,
           'correct_preservation_95':preservation is not None and preservation>=.95,
           'net_improvement_3':gains-losses>=3,'f1_nondecrease':candidate_f1>=base_f1}
    return {'version':'v112','base_em':sum(r['normalized_exact_match'] for r in b.values())/50,
            'candidate_em':sum(r['normalized_exact_match'] for r in n.values())/50,
            'base_f1':base_f1,'candidate_f1':candidate_f1,
            'base_task_protocol_rate':base_task_protocol,'candidate_task_protocol_rate':candidate_task_protocol,
            'base_correct':len(correct),'preserved':kept,'preservation':preservation,'gains':gains,'losses':losses,
            'net_improvement':gains-losses,'state_metrics':state,'numeric_gates':gates,
            'semantic_audit':'pending','internal_eligible':False}


async def evaluate(args, client_factory=None):
    from mini_agent_rl.model.local import TransformersModelClient
    from mini_agent_rl.cli import default_reward
    from mini_agent_rl.runtime import AgentRunner
    from mini_agent_rl.storage import SQLiteStore
    from mini_agent_rl.tools import ToolRegistry, SearchTool
    model_path=validate_source_path(args.model_path)
    adapter_path=validate_source_path(args.adapter_path)
    protected=[model_path,adapter_path,args.gate_report,args.train_data,args.validation_data,OUT]
    output=require_new(args.output,protected)
    db=require_new(str(output)+'.sqlite',protected)
    states,tasks,corpus,data_hashes=load_inputs(args.gate_report,args.train_data,args.validation_data)
    adapter=adapter_path
    if sha256(adapter/'adapter_model.safetensors') != args.expected_adapter_sha256:
        raise ValueError('adapter_hash_mismatch')
    config=model_config(model_path,adapter_path)
    code=sorted((ROOT/'src/mini_agent_rl').rglob('*.py'))+[Path(__file__),ROOT/'scripts/build_v112.py',ROOT/'scripts/train_v112.py',ROOT/'scripts/run_v111_baseline.py']
    fp={'data':data_hashes,'runner_config':RUN_CONFIG,'model_config':config.model_dump(),
        'model_files':tree_fingerprint(model_path),'adapter_files':tree_fingerprint(adapter),
        'code':{str(p.relative_to(ROOT)):sha256(p) for p in code}}
    report={'version':'v112','mode':args.mode,'fingerprint':fp,'rows':[],
            'expected_ids':[r['state_hash'] for r in states] if args.mode=='states' else [t.id for t in tasks],
            'complete_coverage':False,'semantic_audit':'pending','internal_eligible':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    def write():
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write()
    client=(client_factory or TransformersModelClient)(config)
    registry=ToolRegistry();registry.register(SearchTool(corpus,'bm25-v2'))
    try:
        if args.mode=='states':
            index=_BM25Index(corpus)
            for row in states:
                response=await client.generate([Message(**m) for m in row['messages'][:-1]],registry.definitions(),temperature=0,seed=42)
                report['rows'].append(score_state(row,response,index));write()
            report['metrics']=aggregate_states(report['rows'])
        else:
            store=SQLiteStore(str(db))
            try:
                runner=AgentRunner(client,registry,default_reward(),store,environment_fingerprint=fp,**RUN_CONFIG)
                for task in tasks:
                    group=await runner.run_group(task,1)
                    rollout=store.get_rollout(group.rollout_ids[0]);check_infrastructure_error(rollout)
                    report['rows'].append({'task_id':task.id,'rollout_id':rollout.id,'final_answer':rollout.final_answer,
                        'termination':rollout.termination_reason,**answer_metrics(rollout.final_answer,task.answer),
                        'transitions':[t.model_dump() for t in store.get_transitions(rollout.id)]});write()
            finally:
                store.close()
        if tree_fingerprint(adapter)!=fp['adapter_files']:
            raise ValueError('adapter_changed_during_evaluation')
        for path, expected in data_hashes.items():
            candidate=Path(path)
            if not candidate.is_absolute():
                candidate=ROOT/candidate
            if sha256(candidate)!=expected:
                raise ValueError('data_changed_during_evaluation')
        report['complete_coverage']=len(report['rows'])==len(report['expected_ids'])
    finally:
        write()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run');run.add_argument('--mode',choices=('states','tasks'),required=True)
    for name in ('model-path','adapter-path','expected-adapter-sha256','gate-report','train-data','validation-data','output'):
        run.add_argument('--'+name,required=True)
    compare=sub.add_parser('compare')
    for name in ('base-tasks','candidate-tasks','candidate-states','output'):
        compare.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if args.command=='run':
        asyncio.run(evaluate(args))
    else:
        files=[validate_source_path(p) for p in (args.base_tasks,args.candidate_tasks,args.candidate_states)]
        path=require_new(args.output,files)
        result=compare_reports(*[json.loads(Path(p).read_text(encoding='utf-8')) for p in files])
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
