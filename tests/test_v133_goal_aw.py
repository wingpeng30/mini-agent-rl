import asyncio,json
from mini_agent_rl.domain import Message,ModelResponse
from mini_agent_rl.model.decision_pipeline import DecisionConfig,DecisionPipeline,audit_reader

def test_two_auxiliary_limit_and_no_history_mutation():
    calls=[]
    async def generate(messages,tools,**kwargs):
        calls.append(list(messages))
        return ModelResponse(content='ok',raw_text='{"action":"answer","answer":"ok"}',usage={})
    pipe=DecisionPipeline(DecisionConfig(mode='draft'),generate)
    messages=[Message(role='system',content='original'),Message(role='user',content='question'),
              Message(role='tool',content='{"ok":true,"result":[]}')]
    async def run():
        for _ in range(4):await pipe(messages,[])
    asyncio.run(run())
    assert pipe.model_calls==6 and pipe.auxiliary_calls==2
    assert len(messages)==3 and calls[0]==messages
    assert len(calls[1])==4 and len(calls[-1])==3

def test_quote_body_only_and_empty():
    m=[Message(role='tool',content=json.dumps({'ok':True,'result':[{'id':'doc','title':'Title only','text':'Alice is Bob’s daughter.'}]}))]
    obj=dict(bridge='Alice',document_id='doc',quote='Alice is Bob’s daughter.',next_relation='birthplace')
    assert audit_reader(json.dumps(obj),m)['valid']
    obj['quote']='Title only'
    assert not audit_reader(json.dumps(obj),m)['valid']
    assert audit_reader(json.dumps({k:'' for k in obj}),m)['empty']

def test_reader_final_instruction_matches_draft():
    from mini_agent_rl.model.decision_pipeline import FINAL_INSTRUCTION
    assert 'unverified' in FINAL_INSTRUCTION
    assert not audit_reader('{"action":"answer","answer":"hidden"}',[])['valid']
