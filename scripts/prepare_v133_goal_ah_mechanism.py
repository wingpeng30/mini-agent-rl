"""Freeze sixteen existing development decision inputs; never training data."""
from prepare_v133_goal_ah import OUT, ROOT, rows, documents, write_once, sha

# Every incomplete context has at least one legal remaining search. Categories
# describe traps in the visible context, not a factually-false label for absence.
SPECS=[
 ('complete','114313dc',-1,[],['New York University','Tisch School of the Arts']),
 ('complete','829358_',-1,[],['Minas Gerais']),
 ('complete','478978_',-1,[],['Louis Chedid']),
 ('complete','ee323392',-1,[],['1397']),
 ('entity','468778_',1,[['Abraham','married after Sarah death','?']],[]),
 ('entity','76403934',1,[['Sir James Abercrombie 1st Baronet','father','?']],[]),
 ('entity','f299dd60',1,[['Juan de la Cerda 4th Duke','date of death','?']],[]),
 ('entity','f27e0130',1,[['Casimir I the Restorer','birthplace','?']],[]),
 ('relation','235471_',1,[['Moby Grape','artist label affiliation','?']],[]),
 ('relation','df13ddf4',2,[['Sher Shah Suri','birthplace','?']],[]),
 ('relation','66772eb8',2,[['Harry W. Gerstad','deathplace','?']],[]),
 ('relation','69af3b6c',2,[['Emperor Go-Daigo','birthplace','?']],[]),
 ('missing','506035_',1,[['Larkin I. Smith birthplace','county','?']],[]),
 ('missing','337223_',1,[['Mark Crossley team','FA Cup win year','?']],[]),
 ('missing','475070_',1,[['Frances Tupper spouse','birthplace','?']],[]),
 ('missing','862994_',1,[['Gloria Kemasuode birthplace','chief judge','?']],[]),
]

def main():
    source=[r for r in rows(OUT/'cpu-eval-contexts.jsonl') if r['split']=='development']
    probes=[]
    for category,key,index,missing,answers in SPECS:
        matches=[r for r in source if key in r['case_id']]
        if len(matches)!=1:raise ValueError('Ambiguous probe source')
        r=matches[0];c=r['calls'][index];used=c['call_index']
        if category!='complete' and used>=3:raise ValueError('No legal search budget')
        probes.append(dict(case_id=r['case_id'],category=category,question=r['question'],
            call_index=c['call_index'],prompt_token_ids=c['prompt_token_ids'],
            prompt_attention_mask=[1]*len(c['prompt_token_ids']),actual_prompt_sha256=c['prompt_sha256'],
            visible_documents=documents(c['actual_prompt']),remaining_searches=3-used,
            expected_action='answer' if category=='complete' else 'search',
            accepted_answers=answers,missing_relations=missing,scoring='source-bound lead review; search must target needed entity and relation; CPU retrieval returns useful proof tracked separately',
            training_use_forbidden=True,oracle_evidence_added=False))
    write_once(OUT/'mechanism-probes.jsonl',probes,jsonl=True)
    write_once(OUT/'mechanism-freeze.json',dict(probes=16,categories={c:4 for c in ['complete','entity','relation','missing']},
        model_calls=80,models=['baseline','42-A','43-A','42-B','43-B'],
        sha256=sha(OUT/'mechanism-probes.jsonl'),product_score_inclusion=False,
        data_source_sha256=sha(OUT/'cpu-eval-contexts.jsonl'),oracle_evidence_added=0))

if __name__=='__main__':main()
