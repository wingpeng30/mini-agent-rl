"""Add explicit necessary edges and seven independent axes; preserve v3 reviews."""
import close_v133_goal_ai as c
from finalize_v133_goal_ai_closure import T

N=[
 [['Gloria Kemasuode','born in','Delta State'],['Delta State','chief judge at source time','?']],
 [['?agency','decides OTC/prescription marketing','drugs'],['?agency','food-safety system','?']],
 [['Jamie Ward','member of relevant team','?team'],['?team','FA Cup won in','?year']],
 [['Mark Crossley','member of relevant team','?team'],['?team','FA Cup won in','?year']],
 [['Hagar','spouse','?person'],['?person','married after Sarah died','?']],
 [['Juan fifth duke','father','?father'],['?father','died on','?date']],
 [['Balance-toi','performer','Tony Parker'],['Tony Parker','works at requested time','?workplace']],
 [['Holding Out for a Hero Shrek 2 version','performer','?person'],['?person','spouse','?']],
 [['St. Matthew Island','located in sea','?sea'],['?sea','named for','?person']],
 [['Bonne of Armagnac','father','Bernard VII'],['Bernard VII','born in','?place']],
 [['Frances Tupper','spouse','Charles Tupper'],['Charles Tupper','born in','?place']],
 [['Wenceslaus of Krosno','father','Henry VIII the Sparrow'],['Henry VIII the Sparrow','died in','?place']],
 [['Axiliad','director','Witold Leszczyński'],['Witold Leszczyński','born in','?place']],
 [['Horatio Powys','father','?father'],['?father','born in','?place']],
 [['Philipp Albrecht','mother','Margarete Sophie of Austria'],['Margarete Sophie of Austria','born on','?date']],
 [['Shubhendra Shankar','mother','Annapurna Devi'],['Annapurna Devi','born in','?place']],
 [['Stefan Vladislav','father','Stefan the First-Crowned'],['Stefan the First-Crowned','nationality','?']],
 [['Charlie and the Chocolate Factory','spoilt character','?character'],['?character','played by','?actor']],
 [['Rose Tint My World','composer',"Richard O'Brien"],["Richard O'Brien",'spouse','?']],
 [['Day of the Assassin','director','Brian Trenchard-Smith'],['Brian Trenchard-Smith','born in','?place']],
 [['Hickory Grove Estates','located in county','Oktibbeha County'],['Oktibbeha County','county seat','?']],
 [['Desert Studies Center','part of institution','California State University'],['California State University whole system','total campuses','?count']],
 [['Jujhar Singh','father','Guru Gobind Singh'],['Guru Gobind Singh','father','?']],
 [['Adalbert II of Tuscany','father','?father'],['?father','born in','?place']],
]
def axes(kinds):
 return dict(necessary_evidence_not_retrieved='missing_edge' in kinds,
   entity_or_role_mismatch='entity_mismatch' in kinds,
   relationship_type_direction_scope_mismatch='relation_mismatch' in kinds,
   incomplete_chain='missing_edge' in kinds,necessary_qualifier_missing='missing_qualifier' in kinds,
   sufficient_evidence_but_wrong_answer='sufficient_but_wrong' in kinds,
   question_label_or_evidence_ambiguity='ambiguity' in kinds)
def main():
 c.guard();train=c.rows(c.O/'train-audit-ledger-v3.jsonl');diag=c.rows(c.O/'diagnostic-ledger-v3.jsonl')
 overrides={1:[['Vilsandi','located in','Baltic Sea'],['?major Russian city','borders','Baltic Sea']],
  11:[['Joe McLaughlin','played for','Chelsea'],['Chelsea Champions League winning team','manager','?']],
  12:[['Luis Villafañe','country','?country'],['Virgin Islands Pipeline in ?country','named after','?']],
  15:[['I Need You','artist','?artist'],['Something For by ?artist','written for','?']],
  17:[['David Friesen','born in','Tacoma'],['Tacoma','located in county','?']],
  26:[['Elbert, West Virginia','located in','McDowell County, West Virginia'],['McDowell County, West Virginia','shares border with county','?']],
  28:[['Kodungattu','director','Joshiy'],['Joshiy','born in','?']],
  29:[['Maria Nowakowska','born in city','?city'],['Battle named after ?city','ended on','?date']]}
 for i,p in enumerate(train):
  p['required_edges']=overrides.get(i,[e[:3] for e in T[i][3]])
  p['reviewed_state']='post-repair actual input' if p['status']=='recovery' else 'original last actual input'
  p['failure_axes']=axes(p['error_types']);p['review_version']=4
  p['observed_relationships_are_not_necessary_relationships']=True
 for i,p in enumerate(diag):
  p['required_edges']=N[i];p['reviewed_visible_relations']=[z['triple'] for z in p['edge_citations']]
  p['failure_axes']=axes(p['error_types']);p['review_version']=4
  p['missing_relation_proven_absent_from_entire_corpus']=False
 c.save('train-audit-ledger-v4.jsonl',train,True);c.save('diagnostic-ledger-v4.jsonl',diag,True)
 print('Annotated 32 training and 24 diagnostic cases; no scores or semantic decisions changed.')
if __name__=='__main__':main()
