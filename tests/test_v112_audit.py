import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_v112 import check, packet, ProtocolFailure
from finalize_v112 import audit_citations, unique, verified_api_key
from mini_agent_rl.evidence_sft_v112 import compute_candidate_hash, validate_row
from test_evidence_sft_v112 import row


def state(action='answer'):
    return {'state_hash':'s','question':'What is A?', 'target':{'action':action,'value':'B'},
            'evidence':[{'id':'current','title':'A','text':'A is B.'}],
            'retrieval_results':[{'id':'future','title':'C','text':'C is D.'}],
            'messages':[{'role':'user','content':'What is A?'}]}


def review():
    return {'state_hash':'s','approved':True,'reason':'A is explicitly B.',
            'risk_labels':['relation'],'citations':[{'document_id':'current','quote':'A is B.'}],
            'missing_fact':''}


def test_future_evidence_cannot_validate_answer_quote():
    r=review();r['citations']=[{'document_id':'future','quote':'C is D.'}]
    with pytest.raises(ProtocolFailure,match='nonverbatim'):
        check({'reviews':[r]},[state()])
    assert not audit_citations(r,state())


def test_search_requires_explicit_gap_and_new_evidence():
    r=review();r['citations']=[{'document_id':'future','quote':'C is D.'}]
    with pytest.raises(ProtocolFailure,match='missing_gap'):
        check({'reviews':[r]},[state('search')])
    r['missing_fact']='Need the identity of C'
    assert audit_citations(r,state('search'))
    s=state('search');s['messages'].append({'role':'assistant','content':'{"action":"search","query":"B"}'})
    s['messages'].append({'role':'assistant','content':'{"action":"search","query":"B"}'})
    assert not audit_citations(r,s)


def test_blind_packet_has_no_labels_or_gold():
    s=state();s.update(metadata={'gold_answer':'secret'},provisional_risks=['time'],reviews=[review()],split='validation')
    p=packet(s)
    assert not any(x in p for x in ('metadata','reviews','split','provisional_risks'))
    assert 'secret' not in str(p)


def test_duplicate_review_cannot_overwrite():
    with pytest.raises(ValueError,match='duplicate'):
        unique([review(),review()])


def test_review_bound_to_action_not_just_state():
    r=row(1);validate_row(r)
    r['target']['value']='new unsupported answer'
    r['messages'][-1]['content']='{"action":"answer","answer":"new unsupported answer"}'
    with pytest.raises(ValueError,match='dual_approved'):
        validate_row(r)


def test_api_verdict_requires_matching_completed_request_and_response():
    s=state();r=review();p=packet(s)
    provenance={'s':[('settled-key',p,r)]}
    imported={**r,'kind':'api','reviewer':'reviewer','attempts':[],'first_pass_valid':True}
    assert verified_api_key(imported,s,provenance)=='settled-key'
    tampered=copy.deepcopy(imported);tampered['approved']=False
    assert verified_api_key(tampered,s,provenance) is None
    changed=copy.deepcopy(s);changed['target']['value']='another answer'
    assert verified_api_key(imported,changed,provenance) is None


def test_search_citation_must_point_to_new_evidence():
    s=state('search');r=review();r['missing_fact']='Need C'
    assert not audit_citations(r,s)  # Uncited future distractor is not enough.
    r['citations'].append({'document_id':'future','quote':'C is D.'})
    assert audit_citations(r,s)  # Current context plus cited new fact is allowed.
