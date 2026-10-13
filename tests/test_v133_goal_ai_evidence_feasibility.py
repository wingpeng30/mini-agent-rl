import importlib.util
import io
import json
from pathlib import Path

P=Path(__file__).resolve().parents[1]/'scripts/run_v133_goal_ai_evidence_feasibility.py'
spec=importlib.util.spec_from_file_location('goal_ai',P)
ai=importlib.util.module_from_spec(spec);spec.loader.exec_module(ai)

def test_streaming_array_preserves_rows_and_unicode():
    values=[{'id':i,'text':'实体—关系 '+('x'*70000)} for i in range(3)]
    assert list(ai.array_rows(io.StringIO(json.dumps(values,ensure_ascii=False))))==values

def test_incomplete_stream_fails_instead_of_silently_accepting():
    import pytest
    with pytest.raises(ValueError):list(ai.array_rows(io.StringIO('[{"id":1')))

def test_query_candidates_do_not_introduce_hidden_answer():
    q=ai.query_candidates('Director X','education','Where did Director X study?')
    assert len(q)<=3 and all('NYU' not in x for x in q)

def test_replay_compares_text_not_just_document_title():
    a=[dict(id='1',title='Director X',text='Screenwriter Y attended NYU')]
    b=[dict(id='1',title='Director X',text='Director X attended NYU')]
    assert ai.clean_docs(a)!=ai.clean_docs(b)

def test_failure_latch_blocks_another_stage_before_reading_inputs(tmp_path,monkeypatch):
    import pytest
    monkeypatch.setattr(ai,'OUT',tmp_path)
    (tmp_path/'failure.json').write_text('{}',encoding='utf8')
    with pytest.raises(RuntimeError,match='Failure latch'):ai.guard()
