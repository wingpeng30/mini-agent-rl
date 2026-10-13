"""Primary case examination of original training rows 261–270."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (261, "\"The Simpsons\"' fourteenth season", "repair_candidate", ["duplicate_training_action"],
  "Third same-query Simpsons state after rows 259–260. It returns the same episode air date already attached to answer row 185, with only distractors changed.",
  "Preserve original, select at most one justified search example for the question, and check pair replay.",
  [("before_action", "The Simpsons (season 14)", "The other episode, \"How I Spent My Strummer Vacation\", was run by Mike Scully"), ("retrieved_after_search", "How I Spent My Strummer Vacation", "first aired on the Fox network in the United States on November 10, 2002")]),
 (262, "Which Toad the Wet Sprocket album", "retain_candidate", [],
  "The album articles give Coil's 1997 release and New Constellation's October 2013 release. Coil therefore debuted earlier.", None,
  [("before_action", "Coil (album)", "released in 1997"), ("before_action", "New Constellation", "released October 15, 2013")]),
 (263, "In which German federal State", "retain_candidate", ["query_copies_visible_context", "duplicate_training_action_group"],
  "Westerburg's biographical line places his birth at Cologne; the copied sentence retrieves the Cologne article naming North Rhine-Westphalia. The same question/query/result recurs three times.", None,
  [("before_action", "Gerhard Westerburg", "Gerhard Westerburg (Cologne c.1490"), ("retrieved_after_search", "Cologne", "German federal State of North Rhine-Westphalia")]),
 (264, "In which German federal State", "repair_candidate", ["duplicate_training_action"],
  "Same Cologne birthplace, query and federal-state result as row 263 with fewer distractors. It adds no independent target.",
  "Deduplicate after checking the paired answer state at row 267.",
  [("before_action", "Gerhard Westerburg", "Gerhard Westerburg (Cologne c.1490"), ("retrieved_after_search", "Cologne", "German federal State of North Rhine-Westphalia")]),
 (265, "In which German federal State", "repair_candidate", ["duplicate_training_action"],
  "Third same-query copy yielding Cologne's North Rhine-Westphalia state. The different unrelated distractors do not establish a distinct reasoning path.",
  "Preserve original but select one justified representative after pair review.",
  [("before_action", "Gerhard Westerburg", "Gerhard Westerburg (Cologne c.1490"), ("retrieved_after_search", "Cologne", "German federal State of North Rhine-Westphalia")]),
 (266, "In which German federal State", "repair_candidate", ["duplicate_training_action"],
  "Fourth instance of the same Westerburg/Cologne sentence query and retrieved state fact, effectively duplicating row 263's supervision.",
  "Keep source row unchanged and remove redundant copies only in a separately validated export.",
  [("before_action", "Gerhard Westerburg", "Gerhard Westerburg (Cologne c.1490"), ("retrieved_after_search", "Cologne", "German federal State of North Rhine-Westphalia")]),
 (267, "In which German federal State", "retain_candidate", [],
  "The visible Westerburg biography starts with Cologne as birthplace; the separate Cologne article explicitly identifies North Rhine-Westphalia as its German federal state. The Lower Saxony mention in Westerburg's line refers to his death place, not birth place.", None,
  [("before_action", "Gerhard Westerburg", "Gerhard Westerburg (Cologne c.1490"), ("before_action", "Cologne", "German federal State of North Rhine-Westphalia")]),
 (268, "When is American actress who starred", "retain_candidate", ["query_copies_cast_list", "duplicate_training_action_group"],
  "The film excerpt lists Jennifer Coolidge among its stars; searching the cast sentence retrieves her biography, identifying her as an American actress born August 28, 1963. This is the first of three repeated searches.", None,
  [("before_action", "Gentlemen Broncos", "Jennifer Coolidge"), ("retrieved_after_search", "Jennifer Coolidge", "born August 28, 1963"), ("retrieved_after_search", "Jennifer Coolidge", "an American actress and comedian")]),
 (269, "When is American actress who starred", "repair_candidate", ["duplicate_training_action"],
  "Same question and cast-sentence query as row 268, retrieving the same Coolidge birth date despite different distractors.",
  "Deduplicate after verifying the paired answer state and preserve original input for audit.",
  [("before_action", "Gentlemen Broncos", "Jennifer Coolidge"), ("retrieved_after_search", "Jennifer Coolidge", "born August 28, 1963")]),
 (270, "When is American actress who starred", "repair_candidate", ["duplicate_training_action"],
  "Third cast-sentence query copy with identical Coolidge result. No new independent birth-date supervision is present.",
  "Keep frozen row but do not count the duplicate as a new training target in a corrected export.",
  [("before_action", "Gentlemen Broncos", "Jennifer Coolidge"), ("retrieved_after_search", "Jennifer Coolidge", "born August 28, 1963")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-027.jsonl")
