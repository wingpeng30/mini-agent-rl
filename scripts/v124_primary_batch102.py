"""Primary examination of v123 internal candidate states 11–20."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (11, '"Thodi Der" is a song', "retain_candidate", ["query_uses_visible_film", "retrieval_adds_director"],
  "The initial song article explicitly places Thodi Der on the Half Girlfriend soundtrack. Searching that visible film link retrieves its article, which names Mohit Suri as director.", None,
  [("before_action", "Thodi Der", "soundtrack of the Bollywood film Half Girlfriend"), ("retrieved_after_search", "Half Girlfriend (film)", "directed by Mohit Suri")]),
 (12, '"Thodi Der" is a song', "retain_candidate", [],
  "The soundtrack-to-film relation and Half Girlfriend's director credit form a supported two-document answer chain for Mohit Suri.", None,
  [("before_action", "Thodi Der", "Bollywood film Half Girlfriend"), ("before_action", "Half Girlfriend (film)", "directed by Mohit Suri")]),
 (13, " Kevin Daniels had a part in the 2004 American drama", "retain_candidate", ["query_uses_visible_ladder49_role", "retrieval_adds_director"],
  "Kevin Daniels's initial biography identifies his Ladder 49 role. The search quotes that visible role and retrieves the film article, which identifies it as a 2004 American drama directed by Jay Russell.", None,
  [("before_action", "Kevin Daniels", "in the movie Ladder 49"), ("retrieved_after_search", "Ladder 49", "2004 American drama film, directed by Jay Russell")]),
 (14, " Kevin Daniels had a part in the 2004 American drama", "retain_candidate", [],
  "The actor's biography connects Kevin Daniels to Ladder 49, and that film's retrieved article directly credits Jay Russell as director. The answer is supported.", None,
  [("before_action", "Kevin Daniels", "in the movie Ladder 49"), ("before_action", "Ladder 49", "directed by Jay Russell")]),
 (15, "Empress Bianca is the first novel", "repair_candidate", ["which_day_question_vs_full_date_answer", "retrieval_adds_birth_date"],
  "The initial article names Lady Colin Campbell as author of Empress Bianca. Retrieval supplies her full birth date, 17 August 1949. The question asks 'on which day', while the frozen target is a full date; the intended granularity should be stated explicitly before using the pair.",
  "Ask for her full date of birth, or change the answer to day and month if that was intended; replay.",
  [("before_action", "Empress Bianca", "the first novel by Lady Colin Campbell"), ("retrieved_after_search", "Lady Colin Campbell", "born 17 August 1949")]),
 (16, "Empress Bianca is the first novel", "repair_candidate", ["paired_question_date_granularity_mismatch"],
  "17 August 1949 is the supported full date for Lady Colin Campbell. Because the question says 'which day' rather than 'what full date', the frozen answer's year may exceed the requested unit; clarify target granularity before admission.",
  "Use a full-date question for this exact answer and recheck the pair.",
  [("before_action", "Empress Bianca", "first novel by Lady Colin Campbell"), ("before_action", "Lady Colin Campbell", "born 17 August 1949")]),
 (17, "Rebel Eats was produced by the chef", "repair_candidate", ["from_state_inferred_from_birthplace", "query_uses_visible_producer"],
  "The initial Rebel Eats article names producer Justin Warner, and retrieval calls him a chef born in Hagerstown, Maryland. That supports his birth state; 'from what state' is less precise and can imply residence or upbringing not established here.",
  "Ask in which state producer-chef Justin Warner was born, then replay.",
  [("before_action", "Rebel Eats", "produced by Justin Warner"), ("retrieved_after_search", "Justin Warner", "born in Hagerstown, Maryland")]),
 (18, "Rebel Eats was produced by the chef", "repair_candidate", ["paired_from_state_birth_state_conflation"],
  "Maryland follows from Justin Warner's reported Hagerstown birthplace. The original wording says he is 'from' that state, a broader claim than the excerpt proves; the answer needs a birth-state question.",
  "Bind Maryland to birthplace in a revised question and replay.",
  [("before_action", "Rebel Eats", "produced by Justin Warner"), ("before_action", "Justin Warner", "born in Hagerstown, Maryland")]),
 (19, "How many BAFTA awards did the actress", "repair_candidate", ["award_count_snapshot_unanchored", "retrieval_adds_count"],
  "The initial Run article identifies Olivia Colman as Carol. Retrieval says she had won three BAFTA Awards in that source snapshot, but the question gives no date or source boundary for an accumulating award count.",
  "Ask how many BAFTA Awards the cited biography reported by its snapshot date, or use a fixed award list; replay.",
  [("before_action", "Run (TV series)", "Carol (Olivia Colman)"), ("retrieved_after_search", "Olivia Colman", "won three BAFTA Awards")]),
 (20, "How many BAFTA awards did the actress", "repair_candidate", ["paired_unanchored_award_tally"],
  "Three is present in Olivia Colman's retrieved biography, and the Run article identifies her as Carol. The count is a time-dependent tally, so the undated question cannot safely use it as a timeless answer.",
  "Tie the count to the source snapshot or a specified time, then replay.",
  [("before_action", "Run (TV series)", "Carol (Olivia Colman)"), ("before_action", "Olivia Colman", "won three BAFTA Awards")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-102.jsonl", cohort="v123_internal_candidates")
