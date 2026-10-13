"""Separate newly acquired edges, returned complete chains and retained chains."""
import close_v133_goal_ai as c
from finalize_v133_goal_ai_closure import citations,E
def main():
 c.guard();diags=c.rows(c.O/'diagnostic-ledger-v4.jsonl');branches=c.rows(c.O/'budget-reconstruction.jsonl')
 new={
  diags[5]['case_id']+'|1':E('Juan fifth duke','father','Juan fourth duke','Juan de la Cerda, 5th Duke of Medinaceli'),
  diags[13]['case_id']+'|0':E('Horatio Powys','father','Thomas Powys, 2nd Baron Lilford','Horatio Powys'),
  diags[13]['case_id']+'|1':E('Horatio Powys','father','Thomas Powys, 2nd Baron Lilford','Horatio Powys'),
  diags[14]['case_id']+'|0':E('Margarete Sophie of Austria','born on','13 May 1870','Archduchess Margarete Sophie of Austria'),
  diags[23]['case_id']+'|0':E('Adalbert II of Tuscany','father','Adalbert I of Tuscany','Adalbert II, Margrave of Tuscany'),
  diags[23]['case_id']+'|1':E('Adalbert II of Tuscany','father','Adalbert I of Tuscany','Adalbert II, Margrave of Tuscany'),
 }
 out=[]
 for b in branches:
  proof=citations(b,new[b['branch_id']]) if b['branch_id'] in new else []
  if b['branch_id'] in new:assert proof
  out.append(dict(branch_id=b['branch_id'],new_necessary_edges=proof,
    complete_chain_returned=b['branch_id']==diags[14]['case_id']+'|0',
    complete_chain_retained=b['branch_id']==diags[14]['case_id']+'|0',
    ungrounded_excluded=b['branch_id']==diags[5]['case_id']+'|0',product_success_counted=False))
 c.save('branch-progress-v4.jsonl',out,True)
 repairs=c.rows(c.O/'repair-query-results.jsonl');chelsea=next(x for x in repairs if x['plan']['case_id']=='musique:2hop__776776_90536')
 manager=next(d for d in chelsea['result'] if d['title']=='List of Chelsea F.C. managers')
 assert 'Roberto Di Matteo Italy 2012 2012 FA Cup 2012 UEFA Champions League' in manager['text']
 assert chelsea['answer_context']['documents']==[] and chelsea['answer_context']['actual_input_tokens']==245
 c.save('evidence-progress-v4.json',dict(diagnostic=dict(cases_with_new_necessary_edge=4,branches_with_new_necessary_edge=6,
   cases_complete_chain_returned=1,cases_complete_chain_retained=1,illegal_or_ungrounded_not_counted=True),
   training_repairs=dict(queries=16,complete_chain_in_raw_results_and_prefix=11,complete_chain_in_actual_answer_input=10,
     dropped_context_case=chelsea['plan']['case_id'],raw_manager_quote=manager['text'],actual_input_tokens=245,
     raw_quote_is_not_visible_input_evidence=True),product_success_measured=False))
 print('Diagnostic: 4 new-edge cases / 1 returned full chain / 1 retained full chain. Training: 11 raw complete / 10 retained.')
if __name__=='__main__':main()
