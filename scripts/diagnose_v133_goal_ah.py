"""Versioned CPU diagnostic sidecars; preserves every original AG assessment."""
import json
from collections import Counter
from prepare_v133_goal_ah import OUT, AG, documents, rows, sha, write_once

# Source-bound adjudications. Other rows retain inherited categories explicitly.
SPEC = {
'114313dc': ('evidence_sufficient', [('Home Fries','director','Dean Parisot','0b6acd92a259b53aa00ef44e'),('Dean Parisot','education','New York University','20b615033862244da30fef38')], [], 'Director biography explicitly supplies NYU education; writer education alone is insufficient.', 'success'),
'76403934': ('missing_link', [('Sir James Abercrombie','locale','Edinburgh','56f3b855da277556ca88965f')], [['Sir James Abercrombie','father','?'],['father','place of death','?']], 'Edinburgh describes the subject, not the father death location. Old Tony Parker rationale is unrelated.', None),
'468778_': ('entity_mismatch', [('Abraham','firstborn','Ishmael','30425a692981eb963b039466')], [['Abraham','married after Sarah death','?']], 'No Keturah sentence is visible. Ishmael is a son, not the requested later spouse.', None),
'455392_': ('relation_scope_mismatch', [('Desert Studies Center','operator consortium campus count','7','681b09ed753882bb523aa493'),('Southern California','regional CSU campuses','12','7c2fc69a4a3822a7f28edd94')], [['California State University','total campus count','?']], 'Seven is an operator subset; twelve is a regional subset, not CSU system total.', None),
'46376_': ('necessary_evidence_not_retrieved', [('Amy Adams','among highest paid actresses','world','d40aee213156684305f9bd79')], [['question target','highest paid person identity','?'],['target product','two promoted features','?']], 'Highest paid actress does not identify the question target. Retrieval and entity binding both fail.', None),
'485221_': ('entity_mismatch', [('Coffee Swamp','located on','Washington Island Wisconsin','aacf7b1c49e6628fc6af516f'),('Washington Island Michigan','located in','Lake Superior','5b0dffb9a0dfc242946008a6')], [['Washington Island Wisconsin','body of water','?']], 'Michigan and Wisconsin islands are different entities.', None),
'132472_': ('relation_direction_mismatch', [('British Aerospace Dynamics','offered','ALARM','ad1a6167e9015d33d22ad7c1'),('BAC','later became','British Aerospace Dynamics','dc70aec5137d0d60b59bf945')], [], 'Requested predecessor is BAC, not the successor named in the answer.', None),
'358582_': ('missing_link', [('Thrill of a Lifetime album','released by','Capitol Records','3983626058309afd9090d950')], [['King Kobra','artist record label','?']], 'Album release label alone does not establish artist affiliation.', None),
'569582_': ('missing_link', [('Kinfolk album','released through','Derrty Entertainment and Universal Records','33fd33ac57870494db3d602b')], [['Ali & Gipp','signed to','?']], 'Co-release is not an artist signing relation.', None),
'635187_': ('missing_link', [('Lago District','province','Niassa','9ceef37c6dd07c6a14a93f3e'),('Niassa Reserve','located in','Cabo Delgado and Niassa','7f70b4e8fbe0c9a9c265b135')], [['Niassa Province','shares province boundary','?']], 'Reserve membership in two provinces does not prove a shared boundary.', None),
'364489_': ('missing_qualification', [('Bamboo Among the Oaks','publisher','Minnesota Historical Society Press','c6e345e14507be204d22f7be'),('Minnesota Historical Society','headquarters','Minnesota History Center','4009b46c78842fd70e041e0f')], [['Press','organization/HQ binding','?'],['History Center','exact requested location qualification','?']], 'Parent society and publisher press are not explicitly bound; near downtown requires location qualification.', None),
'829358_': ('evidence_sufficient_answer_wrong', [('Darley Ramon Torres','birthplace','Pedro Leopoldo','66b9d6622ad49cead48a46dd'),('Pedro Leopoldo','state','Minas Gerais','27701255ad2329b137d47657')], [], 'Both relations are visible; Minsk Region is the wrong answer.', None),
'805610_': ('ambiguity', [('A Stitch for Time','director','Nigel Noble','609779ac759af4fe58b292cf'),('Close Harmony','director and 54th Academy film award','Nigel Noble / Documentary Short','357e2f40d7033b39e66f895d')], [['Nigel Noble','personal award attribution','?']], 'Question asks director, not A Stitch film. Film award versus personal attribution is ambiguous; preserve AG category.', None),
}

def main():
    inputs=rows(OUT/'cpu-eval-contexts.jsonl')
    inherited={r['case_id']:r for r in rows(AG/'correction-ledger-final-v2.jsonl') if r['arm']=='A'}
    cmap={r['case_id']:r['review_id'] for r in rows(AG/'confirmation-review-map.jsonl') if r['model']=='baseline'}
    creview={r['review_id']:r for r in rows(AG/'confirmation-review-final.jsonl')}
    result=[]; corrections=[]
    for x in inputs:
        prior=inherited[x['case_id']] if x['split']=='development' else creview[cmap[x['case_id']]]
        final=x['calls'][-1]; ds=documents(final['actual_prompt'])
        r=dict(case_id=x['case_id'],split=x['split'],question=x['question'],answer=x['answer'],
               inherited_category=prior['category'],inherited_reason=prior.get('reason'),
               scoring_version='AH CPU sidecar v1; inherited AG categories except explicit corrections',
               input_counts=[len(c['prompt_token_ids']) for c in x['calls']],
               actual_final_prompt_sha256=final['prompt_sha256'],
               visible_document_ids=[d['document_id'] for d in ds],
               review_scope='input-integrity only; semantic category inherited',diagnosis=None)
        spec=next((v for k,v in SPEC.items() if k in x['case_id']),None)
        if spec:
            axis,facts,missing,reason,newcat=spec; quotes=[]
            for entity,rel,obj,did in facts:
                d=next(d for d in ds if d['document_id']==did)
                # Full paragraph is the actual source, with exact input positions.
                escaped=json.dumps(d['text'],ensure_ascii=False)[1:-1]
                start=final['actual_prompt'].find(escaped,d['input_char_span'][0],d['input_char_span'][1])
                if start<0:raise ValueError('Source paragraph not found in actual input')
                end=start+len(escaped); offsets=final['offset_mapping']
                tokens=[i for i,(a,b) in enumerate(offsets) if b>start and a<end]
                quotes.append(dict(triple=[entity,rel,obj],document_id=did,reference=d['reference'],
                    quote=d['text'],call_index=final['call_index'],input_char_span=[start,end],
                    input_token_span=[min(tokens),max(tokens)+1]))
            r.update(review_scope='source-bound targeted semantic review',diagnosis=axis,
                     actual_visible_quotes=quotes,missing_relations=missing,review_reason=reason)
            if newcat:r['corrected_category']=newcat
            if any(k in x['case_id'] for k in ['114313dc','76403934','468778_','455392_']):
                corrections.append(dict(case_id=x['case_id'],old_category=prior['category'],
                    new_category=newcat or prior['category'],reason=reason,evidence=quotes,
                    source_ledger_sha256=sha(AG/('correction-ledger-final-v2.jsonl' if x['split']=='development' else 'confirmation-review-final.jsonl'))))
        result.append(r)
    write_once(OUT/'diagnostic-ledger-v1.jsonl',result,jsonl=True)
    write_once(OUT/'adjudication-corrections-v1.jsonl',corrections,jsonl=True)
    write_once(OUT/'diagnostic-summary.json',dict(rows=len(result),semantic_targeted=sum(r['diagnosis'] is not None for r in result),
        inherited_semantic_rows=sum(r['diagnosis'] is None for r in result),
        diagnoses=dict(Counter(r['diagnosis'] for r in result if r['diagnosis'])),
        cpu_only=True,oracle_evidence_added=0,real_search_success_not_remeasured=True))

if __name__=='__main__':main()
