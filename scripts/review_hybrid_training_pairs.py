"""Freeze 32 body-reviewed questions, with answer/search paired states each."""
import argparse
import json
from pathlib import Path
import re
from run_hybrid_adaptation import read,rows,write,append
from mini_agent_rl.retrieval.local import sha

# Reviewed in source order, before inspecting any candidate training results.
# Excluded cases include false independent-statehood premises, missing birthplace
# relations, unrelated people and ambiguous manufacturer/performer selection.
SELECTED=[1,2,7,8,9,10,14,15,17,18,21,22,25,26,27,29,30,32,33,35,36,37,38,41,42,43,44,46,47,48,49,50]

def docs(state):
    return {d['id']:d for m in state['messages'] if m['role']=='tool' for d in json.loads(m['content'])['result']}

def sentence(text,needle):
    found=[s for s in re.split(r'(?<![A-Z]\.)(?<=[.!?])\s+',text) if needle.casefold() in s.casefold()]
    if not found:raise RuntimeError('Reviewed relation quote absent: '+needle)
    return found[0]

def approve(folder):
    candidates=rows(folder/'training-pair-candidates.jsonl');output=folder/'approved-training-pairs.jsonl'
    if output.exists():raise FileExistsError(output)
    approved=[]
    for index in SELECTED:
        c=candidates[index];case=c['case'];teacher=case['teacher'];arms=c['arms']
        final_a,final_b=docs(arms['A'][1]),docs(arms['B'][1])
        common={k:d for k,d in final_a.items() if k in final_b and d['title'] in teacher['support_titles']}
        first_a,first_b=docs(arms['A'][0]),docs(arms['B'][0])
        bridge=next((d for k,d in common.items() if k in first_a and k in first_b and teacher['bridge'].casefold() in d['text'].casefold()),None)
        if bridge is None:raise RuntimeError('Bridge relation missing from one first view: '+case['id'])
        # Necessary final quote must come from the other relation's document.
        gold=case['answer']
        final=next((d for d in common.values() if d['id']!=bridge['id'] and gold.casefold() in d['text'].casefold()),None)
        if final is None:raise RuntimeError('Final body quote missing: '+case['id'])
        bridge_quote=sentence(bridge['text'],teacher['bridge']);final_quote=sentence(final['text'],gold)
        for arm in ('A','B'):
            first=docs(arms[arm][0]);full=docs(arms[arm][1])
            if final_quote in '\n'.join(d['text'] for d in first.values()):
                raise RuntimeError('Search state already contains final relation: '+case['id'])
            if not all(quote in '\n'.join(d['text'] for d in full.values()) for quote in (bridge_quote,final_quote)):
                raise RuntimeError('Final relations absent from visible input')
        for state_index,kind in ((0,'search'),(1,'answer')):
            approved.append(dict(pair_id=case['id']+':'+kind,case_id=case['id'],kind=kind,question=case['question'],
                semantic_approved=True,reviewer='assistant body review; no candidate output used',
                approval_note='Bridge identifies the entity needed by the second relation; final quote supports the frozen answer. Search prefix lacks the final necessary quote.',
                necessary_quotes=[dict(document_id=bridge['id'],quote=bridge_quote)]+([dict(document_id=final['id'],quote=final_quote)] if kind=='answer' else []),
                missing_final_quote=final_quote if kind=='search' else None,A=arms['A'][state_index],B=arms['B'][state_index]))
    result=approved
    if len(result)!=64 or len({r['case_id'] for r in result})!=32:raise RuntimeError('Approved quota mismatch')
    for item in result:append(output,item)
    write(folder/'training-pairs-freeze.json',dict(approved_pairs=64,answer_pairs=32,search_pairs=32,distinct_questions=32,
        source_sha256=sha(folder/'training-pair-candidates.jsonl'),pairs_sha256=sha(output),review_script_sha256=sha(__file__),
        same_supervised_action=True,all_necessary_quotes_visible=True,final_necessary_quote_absent_from_search_prefixes=True,
        original_trajectory_count=128,updates_per_arm=128,seed=42,auxiliary_repetitions=2,
        evaluation_questions_excluded=True,loss_weights=dict(original=.75,new=.25)))
    print(json.dumps(read(folder/'training-pairs-freeze.json')),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);approve(parser.parse_args().folder)
