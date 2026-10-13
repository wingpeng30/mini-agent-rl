"""Isolated AU BM25-top20 reranker. Original model-facing bodies unchanged."""
from __future__ import annotations
import math,time

class FeasibilityBound(Exception):pass

def window_starts(length,width,stride=256,maximum=8):
    if width<=0 or stride>width:raise FeasibilityBound('document title leaves insufficient window capacity')
    if length<=width:return [0]
    starts=list(range(0,length-width+1,stride))
    if starts[-1]!=length-width:starts.append(length-width)
    if len(starts)>maximum:raise FeasibilityBound('more than eight document windows')
    return starts

def stable_order(candidates,scores):
    if len(candidates)!=len(scores) or not all(math.isfinite(s) for s in scores):raise RuntimeError('reranking score failure')
    return sorted(range(len(candidates)),key=lambda i:(-scores[i],i,str(candidates[i]['id'])))

class FrozenReranker:
    def __init__(self,path,tokenizer_only=False):
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=None;self.journal=[]
        if not tokenizer_only:
            import torch
            if torch.cuda.device_count()!=1:raise RuntimeError('AU single GPU required')
            self.model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,torch_dtype=torch.bfloat16).to('cuda').eval()
    def pairs(self,query,candidates):
        t=self.tokenizer;q=t.encode(query,add_special_tokens=False)
        if len(q)>64:raise FeasibilityBound('query exceeds64 reranker tokens')
        pairs=[];mapping=[];spans=[]
        for rank,doc in enumerate(candidates):
            prefix=t.encode(doc['title']+'\n',add_special_tokens=False);body=t.encode(doc['text'],add_special_tokens=False)
            width=384-len(prefix);starts=window_starts(len(body),width)
            for start in starts:
                tokens=prefix+body[start:start+width]
                pair=t.prepare_for_model(q,tokens,add_special_tokens=True,truncation=False,return_attention_mask=True)
                if len(pair['input_ids'])>512:raise FeasibilityBound('pair exceeds model window')
                pairs.append(pair);mapping.append(rank);spans.append(dict(rank=rank+1,document_id=doc['id'],body_token_span=[start,min(len(body),start+width)]))
        return pairs,mapping,spans
    def rank(self,query,candidates):
        import torch
        pairs,mapping,spans=self.pairs(query,candidates);values=[];start=time.monotonic()
        with torch.inference_mode():
            for batch_start in range(0,len(pairs),8):
                batch=self.tokenizer.pad(pairs[batch_start:batch_start+8],padding=True,return_tensors='pt').to('cuda')
                scores=self.model(**batch).logits.reshape(-1).float()
                if not torch.isfinite(scores).all():raise RuntimeError('AU nonfinite crossencoder')
                values.extend(scores.cpu().tolist())
        document_scores=[max(v for v,m in zip(values,mapping) if m==i) for i in range(len(candidates))]
        order=stable_order(candidates,document_scores)
        record=dict(query=query,bm25_ids=[d['id'] for d in candidates],reranked_ids=[candidates[i]['id'] for i in order],
            document_scores=document_scores,window_scores=values,window_spans=spans,elapsed_seconds=time.monotonic()-start)
        self.journal.append(record)
        # Original dictionaries: full bodies and original BM25 score. Auxiliary scores are never injected.
        return [candidates[i] for i in order[:3]],record
