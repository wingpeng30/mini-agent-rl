import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_v133_fixed_z_coverage as c

def test_external_identity_asset_is_absolute():
 external=c.ROOT.parent/'qwen_08b/model.safetensors'
 assert c.asset_key(external)==str(external)
 assert c.asset_key(c.ROOT/'src/mini_agent_rl/prompts.py')=='src/mini_agent_rl/prompts.py'

def test_title_or_answer_hit_is_not_body_proof():
 edge={'proofs':[{'document_id':'a','quote':'born in Chicago'}]}
 docs=[{'id':'b','title':'born in Chicago','text':'born in Chicago'}, {'id':'a','title':'born in Chicago','text':'from Chicago'}]
 assert c.support_visible(edge,docs)==[]

def test_actual_body_fragment_survives_or_fails():
 edge={'proofs':[{'document_id':'a','quote':'born in Chicago'}]}
 assert c.support_visible(edge,[{'id':'a','text':'X was born in Chicago, Illinois.'}])
 assert not c.support_visible(edge,[{'id':'a','text':'X was born'}])

def test_query_specs_no_embedded_hidden_terminal_answers():
 assert len(c.SPECS)==24 and len({s[0] for s in c.SPECS})==24
 for _,subject,role,quote,a,b in c.SPECS:
  assert subject and role and a and b
  assert '13 May 1870' not in a+b and 'Rosaline' not in a+b
  assert quote is None or subject.casefold() in quote.casefold()
