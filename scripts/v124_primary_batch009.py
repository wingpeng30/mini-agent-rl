"""Primary case examination of original training rows 81–90."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (81, "Which other occupation", "retain_candidate", [],
  "The film names Kristy McNichol in the described role. The query copies that pre-action relation and retrieves her biography, which explicitly adds singer as another occupation. This is the search counterpart of earlier answer row 2.", None,
  [("before_action", "Summer of My German Soldier (film)", "It stars Kristy McNichol as a Jewish-American girl"), ("retrieved_after_search", "Kristy McNichol", "is an American actress and singer")]),
 (82, "What 2012 American musical", "retain_candidate", [],
  "The series article identifies Nashville and R. J. Cutler as producer. Its discography article says the singing-character actors perform their own vocals. Both question clauses agree on the same show; Smash is a distractor.", None,
  [("before_action", "Nashville (2012 TV series)", "produced by R. J. Cutler"), ("before_action", "Nashville discography", "All of the actors who portray singing characters perform their own vocals")]),
 (83, "What is the organization", "retain_candidate", ["question_created_by_figer_not_explicit", "question_typo_midfielderm"],
  "Figer's biography already names his organisation and says it represents Zé Roberto, but does not identify him as the Palmeiras left wing back. The query retrieves Zé Roberto's biography and fills that player-role relation. The question's 'created by' wording is stronger than the documented organisation/co-ownership relation and needs answer-state scrutiny; the typo is separate.", None,
  [("before_action", "Juan Figer", "His organisation, MJF Publicidade e Promoções S/C Ltda"), ("before_action", "Juan Figer", "Zé Roberto"), ("retrieved_after_search", "Zé Roberto", "plays for Palmeiras as a left wing back or as a midfielder")]),
 (84, "What kinds of citizenship", "retain_candidate", ["citizenship_time_snapshot"],
  "The film article identifies Steven Seagal as a star. His biography states he holds American, Russian, and Serbian citizenship, matching the stored list. Citizenship can change, so this remains a source-snapshot answer.", None,
  [("before_action", "Kill Switch (2008 film)", "starring Steven Seagal"), ("before_action", "Steven Seagal", "holds American, Russian, and Serbian citizenship")]),
 (85, " Yuji Nomi", "retain_candidate", ["question_grammar"],
  "Nomi's biography directly pairs Whisper of the Heart with The Cat Returns; the film article independently names Nomi's music credit. The awkward question grammar does not alter the answer.", None,
  [("before_action", "Yuji Nomi", "His work includes the Studio Ghibli films \"Whisper of the Heart\" and \"The Cat Returns\""), ("before_action", "The Cat Returns", "with music by Yuji Nomi")]),
 (86, "What English playwright", "retain_candidate", ["authorship_is_proposed_not_established"],
  "The play article names Jonson and Dekker only as proposed authors, exactly as the question qualifies. The grounded query retrieves Jonson's biography with 1572 birth and English playwright identity. The search supports identification without asserting certain authorship.", None,
  [("before_action", "The Tragical History of Guy Earl of Warwick", "Ben Jonson and Thomas Dekker have been proposed"), ("retrieved_after_search", "Ben Jonson", "c. 11 June 1572 – 6 August 1637) was an English playwright")]),
 (87, "Does the Hokkaido", "retain_candidate", [],
  "The Segugio Italiano article directly states it comes in both short-haired and wire-haired varieties. No negative claim about Hokkaido is needed to answer the either-or question.", None,
  [("before_action", "Segugio Italiano", "It comes in both short-haired and wire-haired varieties")]),
 (88, "What is Jayson Anthony Paul", "retain_candidate", [],
  "The JTG biography gives Jayson Anthony Paul's ring name; the Cryme Tyme article identifies JTG as a member. The answer is supported despite an unrelated similarly worded musician result.", None,
  [("before_action", "JTG", "Jayson Anthony Paul (born December 10, 1984), better known by the ring name JTG"), ("before_action", "Cryme Tyme", "consisting of JTG and Shad Gaspard")]),
 (89, "Pomponius Grekeinos", "retain_candidate", ["query_verifies_missing_person_link"],
  "The prior Catacombs article establishes the underground burial-place category but not Pomponius. Querying that visible place retrieves a Pomponius biography explicitly tying his inscription to the Catacombs of Rome. This verifies the missing person-place link.", None,
  [("before_action", "Catacombs of Rome", "underground burial places under Rome"), ("retrieved_after_search", "Pomponius Grekeinos", "commemorated by an inscription in the Catacombs of Rome")]),
 (90, "Rodelinda and La vida breve", "retain_candidate", [],
  "Rodelinda is identified as opera seria in the initial document. The query names the other work from the question and retrieves La vida breve as an opera. Their supported common category is opera; this does not imply La vida breve is opera seria.", None,
  [("before_action", "Rodelinda (opera)", "is an opera seria in three acts"), ("retrieved_after_search", "La vida breve", "is an opera in two acts and four scenes")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-009.jsonl")
