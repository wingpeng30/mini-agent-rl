import copy
import pytest
from mini_agent_rl.domain import ModelResponse, ToolCall
from mini_agent_rl.tools.search import _BM25Index
from scripts.evaluate_v112 import score_state, compare_reports, require_new, SOURCE_SHA256, normalize_prompt_tokens


def state():
    return {'state_hash':'s','task_id':'t','prompt_token_ids':[1,2],
            'messages':[{'role':'user','content':'Q'},{'role':'assistant','content':'{"action":"search","query":"old"}'},
                        {'role':'assistant','content':'{"action":"answer","answer":"A"}'}],
            'target':{'action':'answer','value':'The Answer'},'evidence':[]}


def test_prompt_mismatch_aborts():
    with pytest.raises(ValueError,match='prompt_token_replay'):
        score_state(state(),ModelResponse(content='A',prompt_token_ids=[9]),_BM25Index([]))


def test_single_batch_saved_prompt_compares_actual_token_sequence():
    row=state();row['prompt_token_ids']=[[1,2]]
    result=score_state(row,ModelResponse(content='answer',prompt_token_ids=[1,2]),_BM25Index([]))
    assert result['normalized_target_match']
    assert row['prompt_token_ids']==[[1,2]]
    with pytest.raises(ValueError,match='prompt_token_replay_mismatch'):
        score_state(row,ModelResponse(content='answer',prompt_token_ids=[1,3]),_BM25Index([]))


@pytest.mark.parametrize('value',[None,[],[[]],[[1],[2]],[[[1]]],[1,[2]],[True],[1.0],['1'],[-1],(1,2)])
def test_invalid_token_shapes_and_types_rejected(value):
    with pytest.raises(ValueError,match='prompt_token'):
        normalize_prompt_tokens(value)


def test_zero_is_valid_token_id():
    assert normalize_prompt_tokens([[0,1]])==[0,1]


def test_answer_normalization_and_raw_output():
    result=score_state(state(),ModelResponse(content='answer',raw_text='{"action":"answer","answer":"answer"}',
                                           prompt_token_ids=[1,2]),_BM25Index([]))
    assert result['normalized_target_match'] and result['protocol_valid']
    assert result['raw_output'] and result['semantic_audit']=='pending'


def test_parse_failure_stays_in_denominator():
    result=score_state(state(),ModelResponse(parse_error='bad json',raw_text='???',prompt_token_ids=[1,2]),_BM25Index([]))
    assert not result['protocol_valid'] and not result['action_match']


def test_repeated_search_and_new_documents():
    response=ModelResponse(tool_calls=[ToolCall(name='search',arguments={'query':'old'})],prompt_token_ids=[1,2])
    result=score_state(state(),response,_BM25Index([{'id':'d','title':'old','text':'old evidence'}]))
    assert result['repeated_search'] and not result['search_legal_nonrepeat']
    assert result['new_retrieval_evidence'][0]['id']=='d'


def reports():
    fp={'data':{'train':'hash'},'runner_config':{'max_steps':3},'model_files':{'w':'hash'},
        'code':{'eval':'hash'},'model_config':{'adapter_path':'base','temperature':0},
        'adapter_files':{'adapter_model.safetensors':SOURCE_SHA256}}
    ids=[str(i) for i in range(50)]
    base={'mode':'tasks','complete_coverage':True,'expected_ids':ids,'fingerprint':fp,
          'rows':[{'task_id':s,'termination':'answer','normalized_exact_match':int(int(s)<20),'token_f1':int(int(s)<20)} for s in ids]}
    new=copy.deepcopy(base)
    for row in new['rows'][:23]:
        row['normalized_exact_match']=row['token_f1']=1
    states={'mode':'states','complete_coverage':True,'fingerprint':copy.deepcopy(fp),
            'expected_ids':[str(i) for i in range(100)],
            'rows':[{'state_hash':str(i),'target':{'action':'answer' if i<50 else 'search'},
                     'protocol_valid':True,'action_match':True,'normalized_target_match':True,
                     'search_legal_nonrepeat':True,'new_retrieval_evidence':[{'id':'new'}]} for i in range(100)]}
    return base,new,states


def test_numeric_success_cannot_bypass_semantic_audit():
    result=compare_reports(*reports())
    assert all(result['numeric_gates'].values())
    assert result['net_improvement']==3 and result['preservation']==1
    assert result['semantic_audit']=='pending' and result['internal_eligible'] is False


def test_missing_question_or_protocol_change_rejected():
    base,new,states=reports();new['rows'].pop()
    with pytest.raises(ValueError,match='fixed_50'):
        compare_reports(base,new,states)
    base,new,states=reports();new['fingerprint']['runner_config']['max_steps']=4
    with pytest.raises(ValueError,match='different_protocol'):
        compare_reports(base,new,states)


def test_answer_type_alone_does_not_pass_answer_accuracy_gate():
    base,new,states=reports()
    for row in states['rows'][:50]:
        row['normalized_target_match']=False
    assert not compare_reports(base,new,states)['numeric_gates']['supported_answer_correct_90']


def test_output_cannot_overlap_protected_input(tmp_path):
    source=tmp_path/'source';source.mkdir()
    with pytest.raises(ValueError,match='overlaps'):
        require_new(source/'output.json',[source])


def test_baseline_cannot_be_arbitrary_adapter():
    base,new,states=reports()
    base['fingerprint']['adapter_files']['adapter_model.safetensors']='other'
    with pytest.raises(ValueError,match='frozen_v101'):
        compare_reports(base,new,states)


def test_search_scoring_matches_runtime_top_three():
    response=ModelResponse(tool_calls=[ToolCall(name='search',arguments={'query':'new'})],prompt_token_ids=[1,2])
    result=score_state(state(),response,_BM25Index([{'id':str(i),'title':'new','text':'new evidence'} for i in range(5)]))
    assert len(result['retrieval_results'])==3


def test_forbidden_output_path_rejected_before_creation():
    with pytest.raises(ValueError):
        require_new('data/final-test/evaluation.json')


def test_task_protocol_gate_independent_of_state_protocol():
    base,new,states=reports()
    for row in new['rows'][-2:]:
        row['termination']='invalid_action'
    result=compare_reports(base,new,states)
    assert result['numeric_gates']['protocol_98'] is True
    assert result['numeric_gates']['task_protocol_98'] is False
    assert result['base_task_protocol_rate']==1
    assert result['candidate_task_protocol_rate']==.96
