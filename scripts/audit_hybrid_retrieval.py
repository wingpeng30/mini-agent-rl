"""Independent scalar check of dense mapping, max aggregation and RRF."""
import argparse
import json
from pathlib import Path
import numpy as np
from mini_agent_rl.retrieval.local import LocalRetrieval, Encoder, sha
from run_hybrid_adaptation import read, rows, write

def audit(folder):
    engine=LocalRetrieval()
    dense=engine.dense_index()
    if dense.encoder is None:dense.encoder=Encoder(engine.cfg)
    docs={d['id']:d for d in engine.documents}
    if any(m['document_id'] not in docs for m in dense.mapping):raise ValueError('Unknown chunk ID')
    queries=sorted({s['query'] for x in read(folder/'states.json') for s in x['messages']
        if s['role']=='assistant' for s in [json.loads(s['content'])] if s.get('action')=='search'})
    values=[]
    for query in queries:
        vector=dense.encoder.query(query)
        # Scalar rows independently reduce chunk scores by document; avoid production fuse.
        maximum={}
        for mapping,v in zip(dense.mapping,dense.vectors):
            score=float(np.dot(v,vector));cid=mapping['document_id']
            maximum[cid]=max(score,maximum.get(cid,float('-inf')))
        dense_ids=sorted(maximum,key=lambda cid:(-maximum[cid],cid))[:20]
        actual_dense=dense.search(query,20)
        if dense_ids!=[d['id'] for d in actual_dense]:raise ValueError('Dense rank mismatch')
        score_delta=max(abs(maximum[d['id']]-d['score']) for d in actual_dense)
        # Published scores round to 6 decimals; scalar and matrix float32
        # accumulation can fall on opposite sides of a rounding boundary.
        if score_delta>2e-6:raise ValueError('Dense score mismatch: '+str(score_delta))
        sparse=engine.bm25.search(query,20);fusion={}
        for ids in ([d['id'] for d in sparse],dense_ids):
            for rank,cid in enumerate(ids,1):fusion[cid]=fusion.get(cid,0)+1/(60+rank)
        ordered=sorted(fusion,key=lambda cid:(-fusion[cid],cid))[:3]
        actual=engine.search(query,'hybrid-rrf-v1')
        if ordered!=[d['id'] for d in actual]:raise ValueError('RRF rank mismatch')
        if any(round(fusion[d['id']],8)!=d['score'] for d in actual):raise ValueError('RRF score mismatch')
        if any(d['title']!=docs[d['id']]['title'] or d['text']!=docs[d['id']]['text'] for d in actual):raise ValueError('Document mapping mismatch')
        values.append(dict(query=query,dense_ids=dense_ids,sparse_ids=[d['id'] for d in sparse],top3=ordered,
            dense_scalar_vs_published_max_absolute_error=score_delta,passed=True))
    write(folder/'retrieval-audit.json',dict(passed=True,queries=len(values),checks=values,
        index_sha256=dense.manifest_sha256,max_norm_error=float(np.max(np.abs(np.linalg.norm(dense.vectors,axis=1)-1)))))
    print(json.dumps({'passed':True,'queries':len(values)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);audit(parser.parse_args().folder)
