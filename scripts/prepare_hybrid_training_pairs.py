"""Build eligible paired states; semantic approval is a separate required gate."""
import argparse
import json
from pathlib import Path
from mini_agent_rl.domain import Message,ToolCall
from mini_agent_rl.prompts import system_prompt
from mini_agent_rl.retrieval.local import LocalRetrieval,build_index,sha
from run_local_relation_review import cpu_client
from run_hybrid_adaptation import read,rows,write,append

def prepare(folder):
    cfg=read(folder/'retrieval-source.json')
    cfg.update(corpus=str(folder/'corpus-train.json'),corpus_sha256=sha(folder/'corpus-train.json'),
        index_dir=str(folder/'training-index'))
    build_index(cfg)
    engine=LocalRetrieval(cfg)
    settings=read(folder/'settings.json');client=cpu_client(settings['product'],settings['base_model'])
    tokenizer=client._processor.tokenizer
    end=tokenizer.convert_tokens_to_ids('<|im_end|>')
    train=rows(folder/'selected-train.jsonl');evaluation=rows(folder/'cases.jsonl')
    if len(train)!=128 or {c['id'] for c in train}&{c['id'] for c in evaluation}:
        raise RuntimeError('Train/evaluation identity overlap')
    if {c['question'].strip().casefold() for c in train}&{c['question'].strip().casefold() for c in evaluation}:
        raise RuntimeError('Train/evaluation question overlap')
    output=folder/'training-pair-candidates.jsonl'
    if output.exists():raise FileExistsError(output)
    count=0
    for case in train:
        teacher=case.get('teacher')
        if not teacher or len(teacher['queries'])!=2:continue
        arms={}; eligible=True
        for arm,mode in (('A','bm25-v2'),('B','hybrid-rrf-v1')):
            messages=[Message(role='system',content=system_prompt(False,'direct-v090')),Message(role='user',content=case['question'])]
            states=[];seen=set();observed=set()
            for round_index,query in enumerate(teacher['queries']):
                docs=engine.search(query,mode);seen.add(' '.join(query.casefold().split()))
                observed.update(d['title'] for d in docs)
                obs=dict(ok=True,result=docs,searched_queries=sorted(seen))
                call=ToolCall(name='search',arguments={'query':query})
                messages.extend([Message(role='assistant',content=json.dumps(dict(action='search',query=query),ensure_ascii=False),tool_calls=[call]),
                    Message(role='tool',content=json.dumps(obs,ensure_ascii=False),tool_call_id=call.id)])
                if round_index==0:
                    if not any(teacher['bridge'].casefold() in d['text'].casefold() for d in docs):eligible=False
                    if set(teacher['support_titles']).issubset(observed):eligible=False
                    target=dict(action='search',query=teacher['queries'][1]);kind='search'
                else:
                    if not set(teacher['support_titles']).issubset(observed):eligible=False
                    target=dict(action='answer',answer=case['answer']);kind='answer'
                inputs=client._inputs(messages,'training-pair-candidate')
                states.append(dict(kind=kind,target=target,prompt_token_ids=inputs['input_ids'][0].tolist(),
                    prompt_attention_mask=inputs['attention_mask'][0].tolist(),
                    action_token_ids=tokenizer.encode(json.dumps(target,ensure_ascii=False),add_special_tokens=False)+[end],
                    messages=[m.model_dump() for m in messages]))
            arms[arm]=states
        if eligible:
            append(output,dict(case=case,arms=arms,semantic_review_status='pending',
                evidence_gate='Both views contain bridge; final views contain both labeled documents. Semantic entailment not yet established.'))
            count+=1
    write(folder/'training-pair-preparation.json',dict(eligible_questions=count,candidate_states=count*2,
        required_pairs=dict(answer=32,search=32),train_eval_ids_disjoint=True,train_eval_questions_disjoint=True,
        semantic_approval_required=True,training_started=False,training_index_sha256=engine.dense_index().manifest_sha256))
    print(json.dumps(read(folder/'training-pair-preparation.json')),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);prepare(parser.parse_args().folder)
