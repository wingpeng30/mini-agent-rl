import json
from types import SimpleNamespace
from mini_agent_rl.domain import Message
from run_hybrid_adaptation import change_view, prefix, metrics

def test_score_free_changes_only_visible_scores():
    payload={'ok':True,'result':[{'id':'a','title':'A','text':'Body','score':.03}], 'searched_queries':['a']}
    messages=[Message(role='tool',content=json.dumps(payload))]
    changed=change_view(messages,'score-free-v1')
    assert json.loads(messages[0].content)==payload
    expected=dict(payload,result=[{'id':'a','title':'A','text':'Body'}])
    assert json.loads(changed[0].content)==expected

def test_repeated_final_search_prefix_reconstructs_before_action():
    raw={'question':'Q','trace':[{'query':'A','observation':{'ok':True,'result':[], 'searched_queries':['a'],'latency_ms':2}},
        {'query':'A','repeated':True}]}
    messages=prefix(raw,1)
    assert len(messages)==4
    assert 'latency_ms' not in json.loads(messages[-1].content)
    assert json.loads(messages[-1].content)['searched_queries']==['a']

def test_bm25_order_preserves_documents_and_scores():
    docs=[{'id':'b','score':.04,'text':'B'},{'id':'a','score':.03,'text':'A'},{'id':'c','score':.02,'text':'C'}]
    class BM:
        def search(self,query,k):return [{'id':'a'},{'id':'b'}]
    engine=SimpleNamespace(bm25=BM(),documents=[1,2,3])
    messages=[Message(role='assistant',content='{"action":"search","query":"Q"}'),Message(role='tool',content=json.dumps({'result':docs}))]
    changed=json.loads(change_view(messages,'bm25-order',engine)[1].content)['result']
    assert [d['id'] for d in changed]==['a','b','c']
    assert sorted(changed,key=lambda d:d['id'])==sorted(docs,key=lambda d:d['id'])

def test_frozen_aliases_scoring():
    assert metrics('alias',{'answer':'gold','answer_aliases':['alias']})['normalized_exact_match']==1
    assert metrics(None,{'answer':'gold'})['normalized_exact_match']==0

def test_stage_gate_requires_all_strata_and_control_gain():
    from continue_hybrid_adaptation import qualifies
    base={'natural':40,'groups':{'development':25,'natural_supplement':15,'constructed_singlehop':18},'invalid':0,'repeated':8}
    winner={'natural':43,'groups':{'development':26,'natural_supplement':17,'constructed_singlehop':18},'invalid':0,'repeated':8}
    assert qualifies(winner,base)
    assert not qualifies(dict(winner,groups=dict(winner['groups'],constructed_singlehop=17)),base)
    assert not qualifies(dict(winner,invalid=1),base)
    assert not qualifies(dict(winner,repeated=9),base)
    assert not qualifies(winner,base,control=dict(base,natural=42))
    assert qualifies(winner,base,control=dict(base,natural=41))

def test_paired_report_matches_questions_and_is_order_independent():
    from report_hybrid_adaptation import paired
    baseline=[{'case_id':'a','normalized_exact_match':0},{'case_id':'b','normalized_exact_match':1},
              {'case_id':'c','normalized_exact_match':1}]
    candidate=[{'case_id':'c','normalized_exact_match':1},{'case_id':'a','normalized_exact_match':1},
               {'case_id':'b','normalized_exact_match':0}]
    result=paired(baseline,candidate)
    assert result['pairs']==3 and result['normalized_em_delta']==0
    assert result['improved']==['a'] and result['regressed']==['b'] and result['unchanged']==1
    assert result==paired(baseline[::-1],candidate[::-1])
    assert result['bootstrap95'][0]<=0<=result['bootstrap95'][1]

def test_group_report_small_hand_calculated_sample():
    from report_hybrid_adaptation import aggregate
    data=[dict(normalized_exact_match=1,strict_exact_match=1,token_f1=1,both_support_seen=True,
        invalid_actions=0,repeated_actions=0,model_calls=3,searches=2,input_tokens=10,output_tokens=5,elapsed_seconds=4,reason='answer'),
        dict(normalized_exact_match=0,strict_exact_match=0,token_f1=.5,both_support_seen=False,
        invalid_actions=1,repeated_actions=0,model_calls=4,searches=3,input_tokens=20,output_tokens=7,elapsed_seconds=6,reason='max_steps')]
    result=aggregate(data)
    assert result['correct']==1 and result['count']==2 and result['mean_f1']==.75
    assert result['calls']==7 and result['searches']==5 and result['seconds']==10
    assert result['all_support']==1 and result['input_tokens']==30 and result['output_tokens']==12
