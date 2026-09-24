"""Freeze training-only questions and construct unapproved evidence/action candidates."""
from __future__ import annotations
import argparse, hashlib, json, re, sqlite3, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/evidence-sft-v112'

def normalize_prompt_tokens(value):
    """Unbatch one prompt without accepting malformed token IDs or shapes."""
    if isinstance(value, dict):
        value = value.get('input_ids')
    if not isinstance(value, list) or not value:
        raise ValueError('invalid_prompt_token_shape')
    if len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not value or any(type(token) is not int or token < 0 for token in value):
        raise ValueError('invalid_prompt_token_type_or_shape')
    return value

def render_prompt(processor, messages):
    data = [{'role': m.role, 'content': m.content} for m in messages]
    return normalize_prompt_tokens(processor.apply_chat_template(
        data, tokenize=True, add_generation_prompt=True, enable_thinking=False))
def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def all_candidates():
    rows=read(OUT/'candidates.jsonl')
    for path in sorted(OUT.glob('supplement-*.jsonl')):
        rows.extend(read(path))
    if len({r['state_hash'] for r in rows})!=len(rows):raise ValueError('duplicate_supplement_state')
    return rows
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2)+'\n'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise RuntimeError('refuse_to_change_frozen_artifact:'+str(path))
    path.write_text(text, encoding='utf-8')
def write_rows(path, rows):
    text = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise RuntimeError('refuse_to_change_frozen_rows')
    path.write_text(text, encoding='utf-8')
def norm(q):
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in unicodedata.normalize('NFKC',q).casefold()).split())
def question(row):
    return next(m['content'] for m in row['messages'] if m['role']=='user')
def provisional_risks(q):
    risks=['relation']
    if re.search(r'\b(year|when|born|older|earlier|first|century|date|founded|started|established)\b',q,re.I): risks.append('time')
    if re.search(r'\b(long|length|height|tall|mile|km|kilometer|metre|meter|population|area|number|how many|how much|larger)\b',q,re.I): risks.append('measurement')
    if re.search(r'\b(both|same|common|which|what|known|type|kind)\b',q,re.I): risks.append('scope')
    return risks
def ledger_snapshot():
    path=ROOT/'runs/v110-api.sqlite'
    if not path.is_file(): raise RuntimeError('authoritative_ledger_missing')
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        rows=db.execute('select * from calls order by key').fetchall()
        stats=db.execute('select status,count(*),sum(charged) from calls group by status').fetchall()
    if any(s!='complete' for s,n,c in stats): raise RuntimeError('unresolved_calls')
    return {'calls':len(rows),'rows_sha256':digest(rows),'charged_cny':sum(c for s,n,c in stats),'remaining_cny':60-sum(c for s,n,c in stats),'authoritative_ledger':'runs/v110-api.sqlite','actual_account_balance_queried':False}
def build(model_path):
    from transformers import AutoProcessor
    from mini_agent_rl.domain import Message
    from mini_agent_rl.model.context import budget_messages
    from mini_agent_rl.prompts import DIRECT_SYSTEM
    from mini_agent_rl.answer_verifier import visible_documents
    from mini_agent_rl.tools.search import _BM25Index
    from mini_agent_rl.training.sft import encode_assistant_only
    source=ROOT/'data/hotpot-agent-v041/train.jsonl'
    rows=read(source)
    old=[]
    for split in ('train','validation'): old+=read(ROOT/f'data/thinking-sft-v090/{split}.jsonl')
    oldids={r['metadata']['task_id'] for r in old}
    oldquestions={norm(question(r)) for r in old}
    diagnostic_rows=read(ROOT/'backups/v110-train-inputs.jsonl')
    diagnostics={r['task_id'] for r in diagnostic_rows}
    diagnostic_questions={norm(r['question']) for r in diagnostic_rows}
    eligible=[r for r in rows if r['metadata']['task_id'] not in oldids|diagnostics and norm(question(r)) not in oldquestions|diagnostic_questions]
    eligible.sort(key=lambda r: hashlib.sha256(('v112-internal:'+r['metadata']['task_id']).encode()).hexdigest())
    internal={r['metadata']['task_id'] for r in eligible[:50]}
    if len(internal)!=50: raise RuntimeError('not_enough_internal_questions')
    internal_q={norm(question(r)) for r in rows if r['metadata']['task_id'] in internal}
    blocked={'hotpot-5a7e02d55542991319bc941f'}
    report=json.loads((ROOT/'reports/v111-train-R-evidence-review.json').read_text(encoding='utf-8'))
    blocked.update(r['task_id'] for r in report['entries'] if r['classification']=='uncertain')
    report=json.loads((ROOT/'reports/v111-retained-evidence-review.json').read_text(encoding='utf-8'))
    blocked.update(r['task_id'] for r in report['items'] if r['classification']=='shared_genre_scope_risk')
    training=[r for r in rows if r['metadata']['task_id'] not in internal|diagnostics|blocked and norm(question(r)) not in internal_q]
    current=ledger_snapshot()
    frozen=OUT/'manifest.json'
    snapshot=json.loads(frozen.read_text(encoding='utf-8'))['opening_ledger'] if frozen.exists() else current
    manifest={'version':'v112','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_questions':len(rows),
      'eligible_internal_questions':len(eligible),'internal_ids':sorted(internal),'training_ids':sorted(r['metadata']['task_id'] for r in training),
      'diagnostic_only_ids':sorted(diagnostics),'quarantine_ids':sorted(blocked),'internal_claim':'current-run question isolation, not historically unseen',
      'opening_ledger':snapshot,'new_api_cap_cny':12,'cumulative_cap_cny':39.600696,
      'training_gate':{'states_min':300,'states_max':400,'questions_min':150},'validation_gate':{'states_min':100,'questions':50},
      'min_answer_fraction':.3,'min_nonempty_search_fraction':.3,'max_empty_search_fraction':.2,
      'risk_min_train':20,'risk_min_validation':5,'final_test_accessed':False,'independent100_accessed':False}
    save(OUT/'manifest.json',manifest)
    corpus={}
    for row in rows:
        for d in visible_documents(row['messages']):corpus[d['id']]=d
    corpus=list(corpus.values());save(OUT/'train-corpus.json',corpus);index=_BM25Index(corpus)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    def render(msgs):
        return render_prompt(processor, msgs)
    candidates=[];rejected=[]
    selected=training+[r for r in rows if r['metadata']['task_id'] in internal]
    for row in selected:
        tid=row['metadata']['task_id'];q=question(row);messages=row['messages']
        # Original states retain exact history. An additional reversed partial
        # state removes a whole document; its proposed search remains untrusted.
        variants=[]
        for i,m in enumerate(messages):
            if m['role']=='assistant' and any(x['role']=='tool' for x in messages[:i]):
                variants.append(('recorded-'+str(i),messages[:i],json.loads(m['content'])))
        docs=visible_documents(messages)
        if len(docs)==2:
            prefix=[{'role':'system','content':DIRECT_SYSTEM},{'role':'user','content':q},
              {'role':'assistant','content':json.dumps({'action':'search','query':q},ensure_ascii=False)},
              {'role':'tool','content':json.dumps({'ok':True,'result':[docs[1]]},ensure_ascii=False)}]
            variants.append(('removed-first-document',prefix,{'action':'search','query':docs[0]['title']}))
        for kind,prefix,action in variants:
            try:
                prefix=[dict(m) for m in prefix];prefix[0]={'role':'system','content':DIRECT_SYSTEM}
                result=budget_messages([Message(**m) for m in prefix],2048,lambda ms:len(render(ms)),lambda s:len(processor.tokenizer.encode(s,add_special_tokens=False)),observation_encoding='compact-v3')
                actual=[{'role':m.role,'content':m.content} for m in result.messages]
                evidence=visible_documents(actual)
                target={'role':'assistant','content':json.dumps(action,ensure_ascii=False,separators=(',',':'))}
                full=actual+[target];encoded=encode_assistant_only(processor,full,2176,len(actual),False)
                if sum(x!=-100 for x in encoded['labels'])>128:raise ValueError('target_exceeds_128')
                state=digest({'question':norm(q),'messages':actual})
                hits=index.search(action['query'],top_k=5) if action['action']=='search' else []
                item={'task_id':tid,'question':q,'messages':full,'target_assistant_index':len(actual),'thinking_enabled':False,
                  'target':{'action':action['action'],'value':action.get('answer',action.get('query'))},
                  'step':sum(m['role']=='assistant' for m in actual),'risk_labels':[], 'provisional_risks':provisional_risks(q),
                  'evidence':evidence,'evidence_hash':digest(evidence),'state_hash':state,'reviews':[],
                  'split':'validation' if tid in internal else 'train','variant':kind,'prompt_token_ids':render(result.messages),
                  'prompt_tokens':len(render(result.messages)),'target_tokens':sum(x!=-100 for x in encoded['labels']),
                  'retrieval_results':[{k:d[k] for k in ('id','title','text')} for d in hits],
                  'metadata':{'source':'data/hotpot-agent-v041/train.jsonl','source_kind':'legacy_support_trajectory_unapproved','gold_answer':row['metadata']['gold_answer']}}
                candidates.append(item)
            except (ValueError,RuntimeError) as exc: rejected.append({'task_id':tid,'variant':kind,'reason':str(exc)})
    unique={}
    for r in candidates:
        if r['state_hash'] in unique and r['target']!=unique[r['state_hash']]['target']:raise RuntimeError('conflicting_state_target')
        unique.setdefault(r['state_hash'],r)
    candidates=list(unique.values())
    # Coverage-led deterministic priority, never quality selected on internal outcomes.
    candidates.sort(key=lambda r:(r['split']!='validation','measurement' not in r['provisional_risks'],'time' not in r['provisional_risks'],hashlib.sha256(('v112-order:'+r['task_id']).encode()).hexdigest(),r['variant']))
    write_rows(OUT/'candidates.jsonl',candidates)
    save(ROOT/'reports/v112-material-inventory.json',{'manifest_sha256':digest(manifest),'candidate_states':len(candidates),
      'splits':{s:{'states':sum(r['split']==s for r in candidates),'questions':len({r['task_id'] for r in candidates if r['split']==s}),
      'provisional_risk_counts':{k:sum(r['split']==s and k in r['provisional_risks'] for r in candidates) for k in ('relation','measurement','time','scope')}} for s in ('train','validation')},
      'preflight_rejections':rejected,'accepted_states':0,'training_permitted':False,'note':'Candidates require dual evidence review; legacy oracle-title queries are not presumed justified.'})
    print(json.dumps({'candidate_states':len(candidates),'internal_questions':len(internal),'encoding_rejections':len(rejected)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-path',default='D:/qwen_08b');a=p.parse_args();build(a.model_path)
