"""Bounded provenance audit of the existing 24 diagnostic questions. CPU only."""
import argparse,io,json,sys,zipfile
from pathlib import Path
import run_v133_fixed_z_coverage as prior
import prepare_v133_goal_z_multihop_data as source

ROOT=prior.ROOT
OUT=ROOT/'runs/v133-goal-ai-coverage-source-v1'

def save(name,value,jl=False):
    with (OUT/name).open('x',encoding='utf8') as f:
        if jl:
            for row in value:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        else:json.dump(value,f,ensure_ascii=False,indent=2)

def prepare():
    prior.guard();assert not OUT.exists()
    plans=prior.rows(prior.OUT/'query-freeze.jsonl');assert len(plans)==24
    packets=prior.rows(prior.OUT/'coverage-review-packets.jsonl');assert len(packets)==48
    files=[Path(__file__),source.MUSIQUE,source.WIKI,ROOT/'scripts/prepare_v133_goal_z_multihop_data.py',
        prior.OUT/'query-freeze.jsonl',prior.OUT/'coverage-review-packets.jsonl',prior.OUT/'coverage-reviewed.jsonl']
    files+=list({ROOT/q['corpus'] for q in plans})
    OUT.mkdir()
    save('freeze.json',dict(case_ids=[q['case_id'] for q in plans],hashes={str(p):prior.sha(p) for p in files},
        source_members={'musique':'data/musique_ans_v1.0_dev.jsonl','2wiki':'dev.json'},
        caps=dict(cases=24,candidates_per_edge=12,source_support_documents=48,new_queries=0,gpu_calls=0,training_updates=0),
        audit_question_changes=False,corpus_changes=False,training_allowed=False,new_confirmation_access=False))
    targets={q['case_id']:q for q in plans};found={};scanned={}
    for kind,path,member in [('musique',source.MUSIQUE,'data/musique_ans_v1.0_dev.jsonl'),('2wiki',source.WIKI,'dev.json')]:
        with zipfile.ZipFile(path) as archive,archive.open(member) as stream:
            if kind=='musique':records=(json.loads(line) for line in stream if line.strip())
            else:records=prior.old.ai.array_rows(io.TextIOWrapper(stream,encoding='utf8'))
            count=0
            for raw in records:
                count+=1;cid=kind+':'+str(raw.get('id',raw.get('_id')))
                if cid not in targets:continue
                assert cid not in found
                canonical=source.canonical(kind,raw)
                supports=[d for d in canonical['docs'] if d['support']]
                found[cid]=dict(case_id=cid,source_archive=str(path),member=member,record_number=count,
                    original_question=canonical['question'],original_answer=canonical['answer'],
                    question_exact=canonical['question']==targets[cid]['question'],
                    official_chain=canonical['chain'],supporting_facts=canonical.get('supporting_facts'),
                    support_documents=supports,diagnostic_only=True,training_allowed=False)
            scanned[kind]=count
    save('official-source-packets.jsonl',[found.get(q['case_id'],dict(case_id=q['case_id'],source_located=False)) for q in plans],True)
    mappings=[]
    for q in plans:
        corpus=prior.read(ROOT/q['corpus'])
        for d in found.get(q['case_id'],{}).get('support_documents',[]):
            title_hits=[c for c in corpus if source.norm(c['title'])==source.norm(d['title'])]
            mappings.append(dict(case_id=q['case_id'],corpus=q['corpus'],official_title=d['title'],official_text=d['text'],
                official_text_sha256=prior.old.ai.key(d['text']),exact_body_ids=[c['id'] for c in title_hits if c['text']==d['text']],
                same_title_documents=[dict(document_id=c['id'],title=c['title'],text=c['text'],text_exact=c['text']==d['text']) for c in title_hits],
                title_present=bool(title_hits),source_body_preserved=any(c['text']==d['text'] for c in title_hits)))
    save('source-corpus-mapping.jsonl',mappings,True)
    # Full frozen candidate bodies, unchanged; each edge retains its original cap.
    save('candidate-fulltext-packets.jsonl',packets,True)
    save('prepare-summary.json',dict(cases=24,source_located=len(found),support_documents=len(mappings),
        exact_body_preserved=sum(m['source_body_preserved'] for m in mappings),members_scanned=scanned,
        candidate_edges=48,unique_candidate_documents=len({(x['corpus'],d['id']) for x in packets for d in x['candidate_documents']}),
        new_queries=0,gpu_calls=0,updates=0))
    print(json.dumps(prior.read(OUT/'prepare-summary.json')))

if __name__=='__main__':
    try:prepare()
    except Exception as e:
        if OUT.exists() and not (OUT/'failure.json').exists():save('failure.json',dict(type=type(e).__name__,message=str(e),automatic_retry=False))
        raise
