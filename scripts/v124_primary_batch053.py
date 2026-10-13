"""Primary examination of v122 internal candidate states 71–80."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (71, "Voruta was the capital city", "repair_candidate", ["question_overstates_uncertain_historical_claim", "answer_entity_visible_before_search"],
  "The Voruta article says it may have been a capital and that its exact location is unknown. The question converts this into a definite 'was' statement. The Kingdom of Lithuania article clarifies the monarchy but cannot remove the original uncertainty.", None,
  [("before_action", "Voruta", "may have been the capital city"), ("before_action", "Voruta", "exact location is unknown"), ("retrieved_after_search", "Kingdom of Lithuania", "was a Lithuanian monarchy")]),
 (72, "Voruta was the capital city", "repair_candidate", ["question_overstates_uncertain_historical_claim"],
  "Kingdom of Lithuania is mentioned in the source and is a monarchy, but Voruta's capital status is explicitly uncertain. A valid question must ask which monarchy Voruta may have served, not state the disputed claim as fact.", None,
  [("before_action", "Voruta", "may have been the capital city of the Grand Duchy of Lithuania and the Kingdom of Lithuania"), ("before_action", "Kingdom of Lithuania", "was a Lithuanian monarchy")]),
 (73, "What is the date of birth of the person", "retain_candidate", ["query_copies_visible_relation"],
  "R. J. Harris's article explicitly identifies Tom Cole as the incumbent he challenged in 2010. The copied challenge sentence retrieves Cole's biography and birth date.", None,
  [("before_action", "R. J. Harris", "challenged incumbent Tom Cole"), ("retrieved_after_search", "Tom Cole", "born April 28, 1949")]),
 (74, "What is the date of birth of the person", "retain_candidate", [],
  "The challenged incumbent is Tom Cole; his biography gives April 28, 1949. The answer is supported and does not confuse Harris's own birth date.", None,
  [("before_action", "R. J. Harris", "challenged incumbent Tom Cole"), ("before_action", "Tom Cole", "born April 28, 1949")]),
 (75, '"Thodi Der" is a song', "retain_candidate", ["query_copies_visible_film_relation"],
  "The song article names Half Girlfriend as its film. Searching that sentence retrieves the film article naming Mohit Suri as director.", None,
  [("before_action", "Thodi Der", "Bollywood film Half Girlfriend"), ("retrieved_after_search", "Half Girlfriend (film)", "directed by Mohit Suri")]),
 (76, '"Thodi Der" is a song', "retain_candidate", [],
  "Thodi Der belongs to Half Girlfriend's soundtrack; that film's article identifies Mohit Suri as director. The answer is supported.", None,
  [("before_action", "Thodi Der", "Bollywood film Half Girlfriend"), ("before_action", "Half Girlfriend (film)", "directed by Mohit Suri")]),
 (77, " Kevin Daniels had a part", "retain_candidate", ["query_copies_visible_role_context"],
  "Daniels's biography identifies Ladder 49 via his Don Miller role. The query copies that visible role sentence and retrieves Ladder 49's article naming director Jay Russell.", None,
  [("before_action", "Kevin Daniels", "movie Ladder 49"), ("retrieved_after_search", "Ladder 49", "directed by Jay Russell")]),
 (78, " Kevin Daniels had a part", "retain_candidate", [],
  "Daniels played Don Miller in Ladder 49; the film article identifies it as a 2004 American drama directed by Jay Russell. The answer is supported.", None,
  [("before_action", "Kevin Daniels", "movie Ladder 49"), ("before_action", "Ladder 49", "2004 American drama film, directed by Jay Russell")]),
 (79, "What rugby game featured James Harris", "retain_candidate", ["answer_title_already_visible", "search_verifies_event_identity"],
  "James Harris's biography already names the 2012 and 2013 Varsity Matches. Copying that line retrieves The Varsity Match article, confirming this is the Oxford-Cambridge rugby fixture.", None,
  [("before_action", "James Harris (rugby player)", "2012 and 2013 Varsity Matches"), ("retrieved_after_search", "The Varsity Match", "annual rugby union fixture played between the universities of Oxford and Cambridge")]),
 (80, "What rugby game featured James Harris", "retain_candidate", [],
  "Harris's rugby biography states he played in the 2012 and 2013 Varsity Matches, and the event article confirms its rugby identity. The answer is supported.", None,
  [("before_action", "James Harris (rugby player)", "2012 and 2013 Varsity Matches"), ("before_action", "The Varsity Match", "annual rugby union fixture")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-053.jsonl", cohort="v122_internal_candidates")
