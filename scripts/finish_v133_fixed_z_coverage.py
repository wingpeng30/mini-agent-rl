"""Read-only adjudication of frozen evidence diagnostics; no retrieval or generation."""
import json
from collections import Counter
from pathlib import Path
import run_v133_fixed_z_coverage as run

def positions(ctx, proof):
    result=[]
    for d in ctx['documents']:
        if str(d.get('document_id',d.get('id')))!=proof['document_id'] or proof['quote'] not in d['text']:continue
        escaped=json.dumps(proof['quote'],ensure_ascii=False)[1:-1]
        start,end=d['body_input_char_span'];a=ctx['actual_prompt'].find(escaped,start,end)
        assert a>=0
        b=a+len(escaped);indices=[i for i,(s,e) in enumerate(ctx['offset_mapping']) if s<b and e>a]
        assert indices
        result.append(dict(proof,input_char_span=[a,b],input_token_span=[min(indices),max(indices)+1],document_body_span=d['body_input_token_span']))
    return result

def main():
    run.guard();p=run.OUT
    plans=run.rows(p/'query-freeze.jsonl');reviews=run.rows(p/'coverage-reviewed.jsonl')
    groups={x['case_id']:[e for e in reviews if e['case_id']==x['case_id']] for x in plans}
    branches=run.rows(p/'branch-audit.jsonl');historical=run.rows(p/'historical-ranking.jsonl')
    packets={x['case_id']:x for x in run.rows(run.PRIOR/'diagnostic-review-packets.jsonl')}
    ranks=[]
    for x in historical+[dict(case_id=b['case_id'],kind='frozen_'+b['arm'],query=b['query'],ranking=b['ranking']) for b in branches]:
        indexed={d['id']:d for d in x['ranking']['ranked']}
        ranks.append(dict(x,edge_support_ranks=[dict(edge_index=e['edge_index'],status=e['status'],
            supports=[dict(document_id=q['document_id'],**{k:indexed.get(q['document_id'],{}).get(k) for k in ['rank','score']}) for q in e['proofs']]) for e in groups[x['case_id']]],
            model_document_limit=3,deeper_documents_delivered=False))
    run.save('support-ranks.jsonl',ranks,True)
    old=[]
    for b in run.rows(run.PRIOR/'budget-reconstruction.jsonl'):
        plan=next(x for x in plans if x['case_id']==b['case_id']);packet=packets[b['case_id']]
        prefix=packet['actual_calls'][b['anchor_call']]
        # All 53 queries were read by the lead; they contain only question entities
        # and this single approved bridge, except the fifth-duke query also embeds
        # the fourth duke. Require that bridge in the old actual body, not a title.
        bridge=plan['subject'];query=run.norm(b['query'])
        uses_bridge=run.norm(bridge) in query and run.norm(bridge) not in run.norm(plan['question'])
        binding=plan['binding'];grounded=not uses_bridge or any(binding.get('quote','__not_a_quote__') in d['text'] for d in prefix['documents'])
        es=groups[b['case_id']];covered=all(e['status']=='supported' for e in es);ambiguous=any(e['critical_ambiguity'] for e in es)
        rawdocs=prefix['documents']+b['result']
        old.append(dict(case_id=b['case_id'],branch_id=b['branch_id'],query=b['query'],grounded=grounded,
            grounding_reason='Question entities or approved binding present in this prefix body' if grounded else 'Named bridge lacks approved relation in this old prefix',
            legal_budget=b['legal_budget'],repeated=b['repeated'],calls_if_answer=b['calls_if_answer'],searches_if_answer=b['searches_if_answer'],
            source_chain_supported=covered,critical_ambiguity=ambiguous,
            complete_chain_returned=covered and all(run.support_visible(e,rawdocs) for e in es),
            complete_chain_retained=covered and all(run.support_visible(e,b['documents']) for e in es),
            visible_edge_proofs=[[hit for q in e['proofs'] for hit in positions(b,q)] for e in es],
            input_token_count=b['actual_input_tokens'],context_usage=b['context_usage'],product_success_counted=False))
    run.save('historical-branch-audit.jsonl',old,True)
    exact=[]
    for b in branches:
        exact.append(dict(branch_id=b['branch_id'],case_id=b['case_id'],visible_edge_proofs=
            [[hit for q in e['proofs'] for hit in positions(b['answer_context'],q)] for e in groups[b['case_id']]]))
    run.save('exact-input-quotes.jsonl',exact,True)
    gpu=run.rows(p/'gpu-next-actions.jsonl') if (p/'gpu-next-actions.jsonl').exists() else []
    judged=[]
    for x in gpu:
        # Source-based equivalence, scoped by the reviewed terminal triple, not gold.
        terminal=groups[x['case_id']][-1]['proofs'][0]['triple'][-1]
        resp=x['response'];answer=resp['content'].strip()
        if resp['tool_calls']:outcome='search_after_sufficient_evidence'
        elif run.norm(answer)==run.norm(terminal):outcome='sufficient_correct_answer'
        else:outcome='requires_manual_action_review'
        assert outcome!='requires_manual_action_review',(answer,terminal)
        judged.append(dict(x,outcome=outcome,source_answer=terminal))
    run.save('gpu-action-review.jsonl',judged,True)
    # Explicit lead judgments of original queries; these observations are separate
    # from causal repair claims, which require a supported unambiguous full chain.
    mistakes={
        'Ashford United team':'Wrong team; visible Ward body identifies Nottingham Forest.',
        'Sarah spouse after death of Hagar':'Reverses Hagar and Sarah roles.',
        'John II of France birthplace':'Changes target from Bernard VII to another generation/person.',
        'Elizabeth Frances Ingram birthplace':'Wrong spouse entity; source wife is Frances, husband Charles.',
        'Henry VIII the Sparrow Duke of Głogów father':'Requests father instead of death place.',
        'Duke Władysław of Opole father':'Requests another ancestor, not target father death place.',
        'Thomas Powys, 1st Baron Lilford birthplace':'Changes father to grandfather.',
        'Albrecht, Duke of Württemberg birthday':'Father role substituted for mother Margarete.',
        'Annapurna Devi mother':'Requests another mother rather than Annapurna birthplace.',
        'Johnny Depp':'Actor is visible for Wonka, not spoiled character Veruca.',
        'Adalbert I, Margrave of Tuscany father':'Requests next ancestor, not Adalbert I birthplace.',
        'Boniface II Margrave of Tuscany birthplace':'Changes requested father to grandfather.'}
    query_observations=[dict(case_id=x['case_id'],query=x['query'],issue=mistakes.get(x['query']),
        causal_failure_attribution='not established by query text alone') for x in historical if x['kind']=='original']
    run.save('original-query-review.jsonl',query_observations,True)
    ledger=[]
    for plan in plans:
        cid=plan['case_id'];es=groups[cid];bs=[b for b in branches if b['case_id']==cid]
        unresolved=any(e['status']!='supported' or e['critical_ambiguity'] for e in es)
        context=any(b['legal_budget'] and b['complete_chain_returned'] and not b['complete_chain_retained'] and not b['critical_ambiguity'] for b in bs)
        decisions=[g for g in judged if g['case_id']==cid]
        query_repair=not unresolved and any(x['issue'] for x in query_observations if x['case_id']==cid) and any(b['eligible_for_gpu'] for b in bs)
        retrieval=False
        for b in bs:
            if unresolved or not b['legal_budget']:continue
            idx={d['id']:d for d in b['ranking']['ranked']}
            retrieval|=any(e['proofs'] and all(idx.get(q['document_id'],{}).get('rank',99999)>3 for q in e['proofs']) and not run.support_visible(e,b['original_prefix_documents']) for e in es)
        decision_failure=any(g['outcome']!='sufficient_correct_answer' for g in decisions)
        primary='coverage_or_question_basis' if unresolved else 'context_retention' if context else 'retrieval' if retrieval else 'answer_decision' if decision_failure else 'conditional_repaired'
        ledger.append(dict(case_id=cid,question=plan['question'],split=plan['split'],anchor_call=plan['anchor_call'],target_binding=plan['binding'],
            required_edges=es,primary_block=primary,coexisting_verified_mechanisms=dict(query_generation=query_repair,retrieval=retrieval,context_retention=context,answer_decision=decision_failure),
            query_observations=[x for x in query_observations if x['case_id']==cid],branches=[dict(branch_id=b['branch_id'],legal=b['legal_budget'],
            new_necessary_edges=b['new_necessary_edges'],complete_returned=b['complete_chain_returned'],complete_retained=b['complete_chain_retained'],gpu_eligible=b['eligible_for_gpu']) for b in bs],
            gpu_outcomes=[g['outcome'] for g in decisions],diagnostic_only=True,training_allowed=False,product_success_counted=False))
    run.save('diagnostic-ledger.jsonl',ledger,True)
    assessment=run.read(p/'cpu-assessment.json');counts={k:sum(x['coexisting_verified_mechanisms'][k] for x in ledger) for k in ['context_retention','retrieval','query_generation','answer_decision']}
    eligible=assessment['coverage_unconfirmed_or_ambiguous']
    route='evidence_coverage_and_question_basis' if eligible>=12 else next((k for k in sorted(counts,key=lambda k:-counts[k]) if counts[k]>=4),'insufficient_mechanism_evidence')
    summary=dict(cpu=run.read(p/'cpu-summary.json'),assessment=assessment,gpu=run.read(p/'gpu-summary.json'),
        primary_blocks=dict(Counter(x['primary_block'] for x in ledger)),verified_mechanism_cases=counts,
        gpu_outcomes=dict(Counter(x['outcome'] for x in judged)),historical_legal=sum(x['grounded'] and x['legal_budget'] for x in old),
        historical_branches=len(old),unique_next_route=route,training_authorized_or_started=False,
        absence_claim=False,review_limit='Full primary and targeted relevant bodies; other candidate titles only. No exhaustive corpus absence audit.')
    run.save('decision.json',summary)
    assert len(ledger)==24 and len(reviews)==48 and len(old)==53 and len(branches)==48
    assert summary['cpu']['new_queries']<=48 and summary['gpu']['calls']<=24
    todo=run.rows(p/'gpu-list.jsonl');assert len(gpu)==len(todo)
    for g,t in zip(gpu,todo):assert g['response']['prompt_token_ids']==t['prompt_token_ids']
    for b in branches:assert len(b['result'])<=3 and b['answer_context']['actual_input_tokens']<=2048
    run.guard()
    run.save('verification.json',dict(original_token_replay_calls=run.read(p/'freeze.json')['token_replayed_calls'],
        rank_top3_exact=True,exact_quote_positions=True,branch_isolation=True,gpu_actual_ids_exact=True,
        identity_and_frozen_hashes=True,training_updates=0,evaluation_training_isolated=True,
        caveat='Semantic unconfirmed decisions are bounded human review, not an exhaustive negative.'))
    run.save('delivery-manifest.json',dict(sha256={f.name:run.sha(f) for f in p.iterdir() if f.is_file()},finisher_sha256=run.sha(Path(__file__))))
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
