"""Primary examination of v122 internal candidate states 201–210."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (201, "Burnt by the Sun contains", "retain_candidate", ["query_uses_visible_cast_relation"],
  "The initial Burnt by the Sun article names Nadezhda Mikhalkova in the cast. The search follows that visible relation and retrieves her biography, which confirms she is a Russian actress and gives her fuller name.", None,
  [("before_action", "Burnt by the Sun", "daughter Nadezhda Mikhalkova"), ("retrieved_after_search", "Nadezhda Mikhalkova", "is a Russian actress")]),
 (202, "Burnt by the Sun contains", "retain_candidate", [],
  "Nadezhda Mikhalkova is named in the film cast, and her biography identifies her as Russian and expands the name to Nadezhda Nikitichna Mikhalkova. The answer is supported.", None,
  [("before_action", "Burnt by the Sun", "daughter Nadezhda Mikhalkova"), ("before_action", "Nadezhda Mikhalkova", "Nadezhda Nikitichna Mikhalkova"), ("before_action", "Nadezhda Mikhalkova", "is a Russian actress")]),
 (203, "What is the nationality of the writer", "retain_candidate", ["query_copies_visible_writer_relation"],
  "The Crook and Ladder episode article names Bill Odenkirk as writer. The search retrieves his biography, which calls him an American comedy writer.", None,
  [("before_action", "Crook and Ladder", "written by Bill Odenkirk"), ("retrieved_after_search", "Bill Odenkirk", "is an American comedy writer")]),
 (204, "What is the nationality of the writer", "retain_candidate", [],
  "Bill Odenkirk wrote Crook and Ladder and is identified as American in his biography. The answer is supported.", None,
  [("before_action", "Crook and Ladder", "written by Bill Odenkirk"), ("before_action", "Bill Odenkirk", "is an American comedy writer")]),
 (205, "What was the nationality of the person who played Ivan", "repair_candidate", ["query_targets_original_version_not_requested_final_version"],
  "The initial Maid of Pskov article already says the third and final version was made famous by Chaliapin as Ivan. The proposed query copies the original-version premiere sentence instead, which is unrelated to the missing Chaliapin nationality. Retrieval happens to bring in his biography and the answer is supported, but the search action is misdirected.",
  "Search for Chaliapin from the visible final-version role sentence, then replay the paired answer.",
  [("before_action", "The Maid of Pskov", "third and final version was completed in 1892"), ("before_action", "The Maid of Pskov", "famous by Chaliapin in the role of Ivan the Terrible"), ("retrieved_after_search", "Feodor Chaliapin", "was a Russian opera singer")]),
 (206, "What was the nationality of the person who played Ivan", "repair_candidate", ["paired_query_targets_original_version"],
  "Russian is supported for Feodor Chaliapin, and the opera article links him to Ivan in the final version. The preceding query nevertheless targeted the original-version premiere, so the pair needs a corrected search.",
  "Retain Russian only after replaying a Chaliapin-targeted query and reviewing the new pair.",
  [("before_action", "The Maid of Pskov", "famous by Chaliapin in the role of Ivan the Terrible"), ("before_action", "Feodor Chaliapin", "was a Russian opera singer")]),
 (207, "What is the parent company of the company founded", "retain_candidate", ["query_uses_visible_skater_identity"],
  "The initial Rodney Mullen article exactly matches the influential street skater clue. Searching his name retrieves Tensor Trucks, whose article says Mullen founded it and identifies Dwindle Distribution as its parent.", None,
  [("before_action", "Rodney Mullen", "most influential street skater in the history of the sport"), ("retrieved_after_search", "Tensor Trucks", "founded and designed by professional skateboarder, Rodney Mullen"), ("retrieved_after_search", "Tensor Trucks", "parent company is Dwindle Distribution")]),
 (208, "What is the parent company of the company founded", "retain_candidate", [],
  "The skater is Rodney Mullen; Tensor Trucks is named as the company he founded, and its article gives Dwindle Distribution as parent. The answer is supported by the same entity chain.", None,
  [("before_action", "Rodney Mullen", "most influential street skater in the history of the sport"), ("before_action", "Tensor Trucks", "founded and designed by professional skateboarder, Rodney Mullen"), ("before_action", "Tensor Trucks", "parent company is Dwindle Distribution")]),
 (209, "In which year the female co-founder", "repair_candidate", ["question_missing_was_auxiliary", "search_verifies_cofounder_clue"],
  "Stevie Case's birth year is visible initially; searching her name retrieves Monkeystone Games and verifies that she was a founder. The evidence chain is sound, but 'In which year the female co-founder ... born?' is ungrammatical and requires a revised question.",
  "Ask 'In which year was Monkeystone Games co-founder Stevie Case born?' and replay.",
  [("before_action", "Stevie Case", "born September 7, 1976"), ("retrieved_after_search", "Monkeystone Games", "founded by John Romero, Tom Hall, Stevie Case, and Brian Moon")]),
 (210, "In which year the female co-founder", "repair_candidate", ["paired_question_missing_was_auxiliary"],
  "The 1976 birth year and Stevie Case's co-founder status are both supported. The original question is syntactically incomplete and needs repair before use.",
  "Retain 1976 in a grammatical, replayed pair.",
  [("before_action", "Stevie Case", "born September 7, 1976"), ("before_action", "Monkeystone Games", "founded by John Romero, Tom Hall, Stevie Case, and Brian Moon")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-066.jsonl", cohort="v122_internal_candidates")
