"""Bind the written body review to immutable local traces; no labels changed."""
import argparse
from collections import Counter
import re
from pathlib import Path
from run_hybrid_adaptation import read, rows, write, append
from mini_agent_rl.retrieval.local import sha

def finalize(folder):
    material=rows(folder/'audit-material.jsonl')
    review_path=Path(__file__).with_name('hybrid_error_review.json')
    review=read(review_path)
    if len(material)!=64 or sorted(r[0] for r in review['reviews'])!=list(range(64)):
        raise ValueError('Review coverage mismatch')
    destination=folder/'error-review.jsonl'
    if destination.exists():raise FileExistsError(destination)
    for index,category,note,eligible in review['reviews']:
        item=material[index];raw=item['raw'];visible={}
        for step_id,step in enumerate(raw['trace']):
            for doc in step.get('observation',{}).get('result',[]):
                visible.setdefault(doc['id'],dict(doc=doc,first_visible_step=step_id))
        excerpts=[]
        for doc in item['labeled_support_documents']:
            text=doc['text']; sentences=re.split(r'(?<=[.!?])\s+',text)
            gold=item['case']['answer'].casefold()
            selected=[s for s in sentences if gold in s.casefold()] or sentences[:1]
            excerpts.append(dict(document_id=doc['id'],title=doc['title'],quote=' '.join(selected),
                actually_seen=doc['id'] in visible,first_visible_step=visible.get(doc['id'],{}).get('first_visible_step')))
        if eligible and not all(e['actually_seen'] for e in excerpts):raise ValueError('Complete evidence flag without visible supports')
        repeated=next((i for i,s in enumerate(raw['trace']) if s.get('repeated')),None)
        flagged=1 if category=='bridge_entity' and len(raw['trace'])>1 else repeated if category=='repeated_search' and repeated is not None else len(raw['trace'])-1
        append(destination,dict(case_id=item['case']['id'],category=category,note=note,
            question=item['case']['question'],gold=item['case']['answer'],answer=item['run']['answer'],
            first_flagged_step=flagged,first_flagged_step_is_causal_proof=False,
            complete_evidence_relation_or_repeat_candidate=eligible,necessary_excerpts=excerpts,
            source_material_sha256=sha(folder/'audit-material.jsonl'),review_scope=review['scope'],
            scoring_changed=False))
    result=rows(destination)
    write(folder/'error-review-summary.json',dict(cases=len(result),categories=dict(Counter(r['category'] for r in result)),
        complete_evidence_relation_or_repeat_candidates=sum(r['complete_evidence_relation_or_repeat_candidate'] for r in result),
        review_source_sha256=sha(review_path),review_ledger_sha256=sha(destination),
        scope=review['scope'],original_scores_unchanged=True,remote_training_started=False))
    print(read(folder/'error-review-summary.json'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);finalize(parser.parse_args().folder)
