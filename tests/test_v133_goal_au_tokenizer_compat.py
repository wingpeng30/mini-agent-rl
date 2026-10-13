from pathlib import Path
import pytest
from transformers import AutoTokenizer
from goal_au_reranker_v2 import pair_from_ids,window_starts
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(str(ROOT/'runs/v133-goal-au-frozen-reranking-v2/reranker'),local_files_only=True)
@pytest.mark.parametrize('query,doc',[
 ('Who was born here?','a body'),('王的父亲?','Father: 王. Date: 1922-09-03.'),
 ('"role" and relation','Braces { nested } and newline\nsecond line.'),('', '')])
def test_declared_template_matches_official_callable(tokenizer,query,doc):
    q=tokenizer.encode(query,add_special_tokens=False);d=tokenizer.encode(doc,add_special_tokens=False)
    manual=pair_from_ids(tokenizer,q,d)
    # Use the official backend's explicit pair even for empty B. The high-level
    # callable treats an empty second string as an omitted pair; real title+
    # body windows are nonempty, while legacy token-ID pair assembly is explicit.
    official=tokenizer.backend_tokenizer.encode(query,pair=doc,add_special_tokens=True)
    assert manual['input_ids']==official.ids
    assert manual['attention_mask']==official.attention_mask
    if doc:
        assert manual['input_ids']==tokenizer(query,doc,add_special_tokens=True,truncation=False)['input_ids']
    assert manual['input_ids'][1:1+len(q)]==q
def test_window_body_tokens_and_boundaries_preserved(tokenizer):
    doc=tokenizer.encode('Words repeat in a long document. '*200,add_special_tokens=False)
    q=tokenizer.encode('birthplace?',add_special_tokens=False)
    starts=window_starts(len(doc),384,maximum=20)
    for start in starts:
        body=doc[start:start+384];pair=pair_from_ids(tokenizer,q,body)
        assert pair['input_ids'][len(q)+3:-1]==body
        assert len(pair['input_ids'])==len(q)+len(body)+4
