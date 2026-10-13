"""One bounded CPU isolation pass. No model-based selection or pool expansion."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import prepare_v133_goal_z_multihop_data as source
import curate_v133_goal_z_multihop as curator
from prepare_v133_goal_ah import ROOT, OUT, rows, sha, stable, write_once

DEST=OUT/'sealed-confirmation-v1'

def digest_text(text):
    return hashlib.sha256(source.norm(text).encode()).hexdigest()

def fingerprints(text):
    return {digest_text(s) for s in re.split(r'[.!?]\s+',text) if len(source.norm(s).split())>=12}

def case_entities(c):
    entities={source.norm(d['title']) for d in c.get('docs',[]) if d.get('support')}
    entities.update(source.norm(s) for s in c.get('support_titles',[]))
    if c.get('source')=='musique':
        entities.update(source.norm(x.get('answer','')) for x in c.get('chain',[])[:-1])
    elif c.get('source')=='2wiki':
        entities.update(source.norm(str(v)) for x in c.get('chain',[]) for v in [x[0],x[2]])
    return {s for s in entities if len(s)>=3}

def registry():
    ids=set();questions=set();titles=set();entities=set();passages=set();sentences=set()
    files={}; excluded=[]
    def walk(x):
        if isinstance(x,list):
            for v in x:walk(v)
        elif isinstance(x,dict):
            for key in ['id','case_id','_id']:
                if isinstance(x.get(key),(str,int)):
                    s=str(x[key]);ids.add(s);ids.add(s.split(':',1)[-1])
            ids.update(str(v) for v in x.get('source_ids',[]) if isinstance(v,(str,int)))
            if isinstance(x.get('question'),str):questions.add(source.w.qhash(x['question']))
            if isinstance(x.get('title'),str):titles.add(source.norm(x['title']))
            for key in ['text','paragraph_text']:
                if isinstance(x.get(key),str):
                    passages.add(digest_text(x[key]));sentences.update(fingerprints(x[key]))
            if x.get('source') in ['musique','2wiki'] and isinstance(x.get('chain'),list):entities.update(case_entities(x))
            for k,v in x.items():
                if k not in ['prompt_token_ids','actual_input_token_ids','action_token_ids','offset_mapping','old_logprobs']:walk(v)
    for folder in sorted((ROOT/'runs').glob('v133-goal-*')):
        if folder==OUT:continue
        # Test temporary directories contain no experiment data and may be unreadable.
        for directory,dirs,names in __import__('os').walk(folder):
            dirs[:]=[d for d in dirs if not any(s in d.casefold() for s in ['pytest','__pycache__','.git'])]
            for name in sorted(names):
                path=Path(directory)/name
                if path.suffix not in ['.json','.jsonl']:continue
                if 'sealed' in name.casefold():
                    excluded.append(str(path.relative_to(ROOT)));continue
                text=path.read_text(encoding='utf-8-sig')
                try:
                    values=[json.loads(s) for s in text.splitlines() if s.strip()] if path.suffix=='.jsonl' else [json.loads(text)]
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f'Exposure registry cannot parse {path}') from exc
                for v in values:walk(v)
                files[str(path.relative_to(ROOT))]=sha(path)
    return ids,questions,titles,entities,passages,sentences,files,excluded

def main():
    if DEST.exists():raise FileExistsError('One-time confirmation isolation pass already exists')
    DEST.mkdir()
    ids,questions,titles,entities,passages,sentences,files,excluded=registry()
    write_once(OUT/'exposure-registry.json',dict(files=files,excluded_named_sealed_files=excluded,
        excluded_test_temporary_dirs=True,counts=dict(ids=len(ids),questions=len(questions),titles=len(titles),
            entities=len(entities),passages=len(passages),sentences=len(sentences)),
        includes_all_historical_candidate_pools=True,not_a_training_source=True))
    accepted=[];reject=Counter();counts={}
    for name,loader in [('musique',source.load_musique),('2wiki',source.load_wiki)]:
        ranked=sorted((v for v in loader('dev') if source.source_eligible(name,v)),
                      key=lambda v:stable('confirmation:'+name+':'+str(v.get('id',v.get('_id')))))[:256]
        pool=[source.canonical(name,v) for v in ranked]
        write_once(DEST/f'candidate-pool-{name}.jsonl',pool,jsonl=True)
        count=0
        for c in pool:
            reason=source.audit_case(c,(ids,questions,titles))
            supports=[d for d in c['docs'] if d['support']]
            if not reason and case_entities(c)&entities:reason='historical_entity_chain'
            if not reason and any(digest_text(d['text']) in passages or fingerprints(d['text'])&sentences for d in supports):reason='historical_support_passage'
            if reason:reject[reason]+=1;continue
            accepted.append(c);count+=1
            ids.update(c['source_ids']);questions.add(source.w.qhash(c['question']))
            entities.update(case_entities(c));titles.update(source.norm(d['title']) for d in supports)
            if count==32:break
        counts[name]=count
    # Isolation is not an entailment audit; no claim of a ready-to-use confirmation.
    manifest=dict(counts=counts,required=dict(musique=32,**{'2wiki':32}),pool_cap_per_source=256,
        one_time_pass=True,rejections=dict(reject),confirmation_accessed_by_models=False,
        exposure_registry_sha256=sha(OUT/'exposure-registry.json'),semantic_source_review='pending',
        status='awaiting_source_quality_audit' if counts=={'musique':32,'2wiki':32} else 'stopped_isolation_shortfall',
        oracle_or_model_filtering=False,pool_expansion_allowed=False)
    write_once(DEST/'isolated-candidates.jsonl',accepted,jsonl=True)
    if len(accepted)==64:
        # Historical support titles are already excluded above. Build with historical
        # corpus titles, but allow the newly selected support titles.
        historical_titles=set(titles)-{source.norm(d['title']) for c in accepted for d in c['docs'] if d['support']}
        corpus,_=curator.build_corpus('confirm',accepted,historical_titles,historical_titles)
        write_once(DEST/'corpus.json',corpus)
        manifest['corpus_sha256']=sha(DEST/'corpus.json')
    write_once(OUT/'confirmation-isolation-manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False))

if __name__=='__main__':main()
