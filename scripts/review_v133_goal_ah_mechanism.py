"""Main-agent source-bound scoring of the frozen 80 mechanism calls (CPU only)."""
import json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'runs/v133-goal-ah-entity-relation'
LABELS=['baseline','42-A','43-A','42-B','43-B']
REASONS=[
'Director Dean Parisot studied NYU; correct supported answer.',
'Birth Pedro Leopoldo is in Minas Gerais; Minsk Region is a wrong answer despite complete evidence.',
'Performer Matthieu Chedid has father Louis Chedid; correct.',
'Father Francesco I explicitly died 1397; correct.',
'Need Abraham’s marriage after Sarah’s death. Only 42-B preserves the right deceased person; other queries incorrectly condition on Hagar’s death.',
'Sir James’s parent James Hamilton is already explicitly visible. All queries correctly request James Hamilton’s place of death, not Sir James’s own death.',
'Fifth duke’s father is fourth duke; all queries target the fourth duke’s death date. Retrieved proof gives year 1575; no full calendar date, so partial precision only.',
'Father Casimir is already bound. All queries ask Casimir’s father instead of his birthplace, although CPU retrieval incidentally supplies his Krakow birthplace.',
'All queries target Moby Grape artist record label correctly. Retrieved album-label paragraphs do not prove artist affiliation.',
'Farid Khan is Sher Shah’s birth name, not birth location; all models prematurely answer.',
'Palm Springs is a retirement location, not death location; all models prematurely answer.',
'Only baseline queries Go-Daigo’s birthplace. Seed42 models query court locations Yoshino/Kyoto; seed43 models answer Kyoto without birthplace evidence.',
'All queries correctly request Poplarville county; CPU retrieved paragraph explicitly identifies Pearl River County.',
'All queries add the unrelated UEFA-policy season 2015–16 to the FA Cup win search. Scope is unsupported; CPU retrieval supplies no team win-year proof.',
'Frances’s husband is Charles Tupper. All queries instead target unrelated Elizabeth Frances Ingram’s birthplace.',
'All queries target chief judge of Gloria’s birthplace, either by alias or Delta State. Returned Rivers State/other judge paragraphs do not supply Delta State proof.',
]

def write_once(path,data):
    data=json.dumps(data,ensure_ascii=False,indent=2)+'\n'
    if path.exists():assert path.read_text(encoding='utf8')==data
    else:path.write_text(data,encoding='utf8')

def main():
    probes=list(map(json.loads,(OUT/'mechanism-probes.jsonl').read_text(encoding='utf8').splitlines()))
    actual={}
    for p in OUT.glob('actual-inputs-*-baseline.jsonl'):
        for row in map(json.loads,p.read_text(encoding='utf8').splitlines()):actual[row['case_id']]=row
    all_reviews=[];summary={}
    for label in LABELS:
        raw=list(map(json.loads,(OUT/f'mechanism-{label}.jsonl').read_text(encoding='utf8').splitlines()))
        correct={0,2,3,5,6,8,12,15}
        if label=='baseline':correct.add(11)
        if label=='42-B':correct.add(4)
        for i,(probe,row) in enumerate(zip(probes,raw)):
            assert probe['case_id']==row['case_id'] and probe['prompt_token_ids']==row['prompt_token_ids']
            call=actual[row['case_id']]['calls'][probe['call_index']]
            assert call['prompt_token_ids']==row['prompt_token_ids']
            useful=i in {5,7,12} or (i==4 and label=='42-B')
            partial=i==6
            all_reviews.append(dict(model=label,probe_index=i,case_id=row['case_id'],category=probe['category'],
                correct_intent_or_supported_answer=i in correct,useful_cpu_proof=useful,
                partial_precision_cpu_proof=partial,product_success_inclusion=False,
                actual_prompt_sha256=call['prompt_sha256'],call_index=call['call_index'],
                parsed_output={k:row.get(k) for k in ['answer','query','parse_error']},
                necessary_missing_relation=([['James Hamilton, 4th Duke of Hamilton','deathplace','?']] if i==5 else probe['missing_relations']),
                visible_source=[dict(document_id=d['document_id'],quote=d['text'],input_char_span=d['input_char_span'],input_token_span=d['input_token_span'],span_scope='document JSON block') for d in call['documents']],
                cpu_retrieved_source=row.get('cpu_retrieved',[]),reason=REASONS[i],reviewer='main agent'))
        summary[label]=dict(correct=len(correct),total=16,useful_cpu_proof=sum(r['useful_cpu_proof'] for r in all_reviews if r['model']==label),partial_precision_cpu_proof=1)
    write_once(OUT/'mechanism-source-adjudication-v1.json',dict(reviews=all_reviews,summary=summary,
        B_over_A={str(seed):summary[f'{seed}-B']['correct']-summary[f'{seed}-A']['correct'] for seed in [42,43]},
        route_qualifies=False,confirmation_allowed=False,reason='Seed43 B=A on semantic actions and CPU results, below frozen +1 requirement.',
        metadata_correction=dict(probe_index=5,old=probes[5]['missing_relations'],new=[['James Hamilton, 4th Duke of Hamilton','deathplace','?']],reason='Actual frozen input already names father. Inputs, runs, gold and expected search action unchanged.')))
    print(json.dumps(summary))

if __name__=='__main__':main()
