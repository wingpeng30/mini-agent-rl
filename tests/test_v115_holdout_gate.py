import hashlib
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from authorize_v115_independent100 import authorize

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,sort_keys=True),encoding='utf-8')
    return Path(path)

def setup(tmp_path, *, numeric=True, semantic=True):
    numeric_gates={k:numeric for k in ('protocol_98','supported_answer_90','insufficient_evidence_search_90',
        'correct_preservation_95','net_gain_at_least_3','f1_non_decrease')}
    comparison=dump(tmp_path/'comparison.json',{'version':'v115','internal_only':True,'independent100_opened':False,
        'final_test_accessed':False,'numeric_gates':numeric_gates})
    packet=dump(tmp_path/'packet-manifest.json',{'blinded':True,'gold_in_packets':False,'scores_in_packets':False,
        'model_identity_in_packets':False,'changed_task_cases':2})
    review=dump(tmp_path/'review.json',{'packet_manifest_sha256':hashlib.sha256(packet.read_bytes()).hexdigest(),
        'reviewers':2,'independent_reviews':True,'state_case_count':100,'reviewed_state_cases_by_each':{'a':100,'b':100},
        'changed_task_case_count':2,'reviewed_changed_tasks_by_each':{'a':2,'b':2},
        'unresolved_disputes':0 if semantic else 1,'confirmed_semantic_regressions':0,
        'support_claim':'no_unresolved_support_issue' if semantic else 'unresolved'})
    return comparison,review,packet

def test_independent100_is_blocked_when_any_internal_numeric_gate_fails(tmp_path):
    comparison,review,packet=setup(tmp_path,numeric=False)
    out=tmp_path/'authorization.json'
    with pytest.raises(ValueError,match='internal_numeric_gate_failed'):
        authorize(comparison,review,packet,out)
    assert not out.exists()

def test_independent100_is_blocked_on_any_semantic_dispute(tmp_path):
    comparison,review,packet=setup(tmp_path,semantic=False)
    with pytest.raises(ValueError,match='semantic_gate_failed'):
        authorize(comparison,review,packet,tmp_path/'authorization.json')

def test_independent100_authorization_requires_two_complete_blind_reviews(tmp_path):
    comparison,review,packet=setup(tmp_path)
    out=tmp_path/'authorization.json'
    result=authorize(comparison,review,packet,out)
    assert result['authorized'] is True
    assert result['independent100_opened_at_authorization'] is False
    assert result['final_test_accessed'] is False
