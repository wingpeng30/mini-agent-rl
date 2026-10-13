import pytest
from goal_au_reranker import window_starts,stable_order,FeasibilityBound
def test_windows_include_end_without_silent_truncation():
    assert window_starts(100,384)==[0]
    assert window_starts(700,384)==[0,256,316]
    assert window_starts(2176,384)==[0,256,512,768,1024,1280,1536,1792]
    with pytest.raises(FeasibilityBound):window_starts(2300,384)
def test_ties_preserve_bm25_rank_and_not_document_name():
    docs=[{'id':'z'},{'id':'a'},{'id':'b'}]
    assert stable_order(docs,[1.,1.,2.])==[2,0,1]
    with pytest.raises(RuntimeError):stable_order(docs,[1.,float('nan'),2.])
