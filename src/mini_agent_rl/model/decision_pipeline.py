"""Optional, bounded auxiliary decisions; never executes auxiliary actions."""
import json
from dataclasses import dataclass
from ..domain import Message

READER_PROMPT = ('Output exactly one JSON object with string fields bridge, document_id, quote, next_relation. '
 'Resolve only the first entity or fact needed by the original question from visible document bodies. '
 'bridge is that value; quote is one exact body quote proving it; next_relation states what must be checked next. '
 'Keep roles, direction, time and scope. Do not answer the final question. '
 'If unsupported, use empty strings for all four fields. Do not invent names absent from the question or bodies.')
FINAL_INSTRUCTION = ('The auxiliary output below is a generated, unverified suggestion, not evidence. '
 'Answer the original question or search using the original visible document bodies. '
 'Output exactly one ordinary search/answer JSON action. Auxiliary output: ')

@dataclass(frozen=True)
class DecisionConfig:
    mode: str = 'none'
    max_auxiliary_calls: int = 2
    max_model_calls: int = 6
    def __post_init__(self):
        if self.mode not in ('none','draft','reader'):raise ValueError('Unknown decision mode')
        if self.max_auxiliary_calls!=2 or self.max_model_calls!=6:raise ValueError('Frozen AW budgets')

def audit_reader(raw,messages):
    try:
        obj=json.loads(raw)
        keys=('bridge','document_id','quote','next_relation')
        if not isinstance(obj,dict) or set(obj)!=set(keys) or not all(isinstance(obj[k],str) for k in keys):
            return dict(valid=False,reason='invalid reader fields')
        if all(obj[k]=='' for k in keys):return dict(valid=True,empty=True,result=obj)
        if not all(obj[k] for k in keys):return dict(valid=False,reason='partly empty reader')
        from .grounded_action_audit import visible_documents
        matches=[d for d in visible_documents([m.model_dump() if hasattr(m,'model_dump') else m for m in messages])
                 if d['document_id']==obj['document_id'] and obj['quote'] in d['text']]
        if not matches:return dict(valid=False,reason='quote not in currently visible document body')
        return dict(valid=True,empty=False,result=obj,semantic_support_not_automatically_proven=True)
    except (ValueError,TypeError,KeyError):return dict(valid=False,reason='invalid reader JSON')

class DecisionPipeline:
    def __init__(self,config,generate,auxiliary_generate=None):
        self.config=config;self.generate=generate;self.auxiliary_generate=auxiliary_generate or generate
        self.auxiliary_calls=0;self.model_calls=0
    async def __call__(self,messages,tools,**kwargs):
        auxiliary=[]
        eligible=self.config.mode!='none' and any(m.role=='tool' for m in messages) and self.auxiliary_calls<2
        if eligible:
            copied=list(messages)
            if self.config.mode=='reader':copied[0]=Message(role='system',content=READER_PROMPT)
            aux=await self._call(self.auxiliary_generate,copied,tools,**kwargs)
            self.auxiliary_calls+=1
            audit=audit_reader(aux.raw_text,messages) if self.config.mode=='reader' else dict(valid=not bool(aux.parse_error),reason=aux.parse_error)
            auxiliary.append(dict(raw=aux.raw_text,prompt_token_ids=aux.prompt_token_ids,action_token_ids=aux.action_token_ids,
                                  usage=aux.usage,audit=audit,ordinary_action_parse_error=aux.parse_error))
            payload=dict(status='unverified' if audit['valid'] else 'invalid',output=aux.raw_text if audit['valid'] else None)
            copied=list(messages)+[Message(role='user',content=FINAL_INSTRUCTION+json.dumps(payload,ensure_ascii=False))]
        else:copied=messages
        final=await self._call(self.generate,copied,tools,**kwargs)
        return final.model_copy(update={'usage':dict(final.usage,decision_pipeline=dict(mode=self.config.mode,auxiliary=auxiliary,
             auxiliary_calls=self.auxiliary_calls,actual_model_calls=self.model_calls))})
    async def _call(self,generate,messages,tools,**kwargs):
        if self.model_calls>=self.config.max_model_calls:raise RuntimeError('Actual model call budget exceeded')
        self.model_calls+=1
        return await generate(messages,tools,**kwargs)
