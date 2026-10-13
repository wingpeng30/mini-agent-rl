"""Primary case examination of original training rows 241–250."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (241, "Which band was formed earlier, Marilyn Manson", "retain_candidate", ["person_band_name_collision", "query_biases_known_entity", "duplicate_training_action_group"],
  "The pre-action band article gives Marilyn Manson's 1989 formation and distinguishes it from the same-named singer. Although the query emphasizes the already-known band, retrieval supplies Walk the Moon's 2006 start, completing the comparison.", None,
  [("before_action", "Marilyn Manson (band)", "in 1989"), ("retrieved_after_search", "Walk the Moon", "started the band in 2006")]),
 (242, "Which band was formed earlier, Marilyn Manson", "retain_candidate", ["person_band_name_collision"],
  "Walk the Moon's 2006 start is known before the action. The query emphasizes that band but retrieves the Marilyn Manson band article with the missing 1989 formation year; the singer's similarly named article must not substitute for the band evidence.", None,
  [("before_action", "Walk the Moon", "started the band in 2006"), ("retrieved_after_search", "Marilyn Manson (band)", "in 1989")]),
 (243, "Which band was formed earlier, Marilyn Manson", "repair_candidate", ["duplicate_training_action", "person_band_name_collision"],
  "This repeats row 241's question, query, known Marilyn Manson band year, and retrieved Walk the Moon year with a different irrelevant distractor. It is not a separate decision target.",
  "Keep the frozen state for audit and deduplicate the two same-path searches after answer-state inspection.",
  [("before_action", "Marilyn Manson (band)", "in 1989"), ("retrieved_after_search", "Walk the Moon", "started the band in 2006")]),
 (244, "Nilay Konar", "retain_candidate", ["surname_only_query"],
  "With no prior evidence, querying the surname from the question retrieves Nilay Konar's biography, which states her Ankara birthplace. A surname-only query is broad but this observed result fills the first relation.", None,
  [("retrieved_after_search", "Nilay Konar", "born August 30, 1980 in Ankara")]),
 (245, "Nilay Konar", "retain_candidate", ["query_uses_candidate_answer"],
  "Before this action, the Ankara article establishes it as Turkey's capital, but no Nilay birthplace evidence is visible. Querying Ankara retrieves Konar's biography and establishes the missing person-place link. The candidate-city query has succeeded in this recorded result.", None,
  [("before_action", "Ankara", "is the capital of the Republic of Turkey"), ("retrieved_after_search", "Nilay Konar", "born August 30, 1980 in Ankara")]),
 (246, "Nilay Konar", "retain_candidate", [],
  "Nilay Konar's biography states her Ankara birthplace and the Ankara article identifies it as Turkey's capital. The final answer is supported by both facts.", None,
  [("before_action", "Nilay Konar", "born August 30, 1980 in Ankara"), ("before_action", "Ankara", "is the capital of the Republic of Turkey")]),
 (247, "The Edge of Seventeen Features", "retain_candidate", ["query_copies_cast_list", "duplicate_training_action_group"],
  "The film article lists Woody Harrelson in its cast. Searching that cast sentence retrieves his biography giving July 23, 1961, so the year is available after the action. This is one representative of four repeated cast-list queries.", None,
  [("before_action", "The Edge of Seventeen", "Woody Harrelson"), ("retrieved_after_search", "Woody Harrelson", "born July 23, 1961")]),
 (248, "The Edge of Seventeen Features", "repair_candidate", ["duplicate_training_action", "query_copies_cast_list"],
  "Same question and exact cast-list query as row 247, returning the same Harrelson birth date. Different distractors do not make a new independent label.",
  "Preserve original row and deduplicate this group after paired answer review.",
  [("before_action", "The Edge of Seventeen", "Woody Harrelson"), ("retrieved_after_search", "Woody Harrelson", "born July 23, 1961")]),
 (249, "The Edge of Seventeen Features", "repair_candidate", ["duplicate_training_action", "query_copies_cast_list"],
  "Third cast-list query copy with the same target, action and retrieval as rows 247–248. No new birth-date supervision is added.",
  "Retain frozen evidence but exclude redundant copies from any corrected training export pending full pair review.",
  [("before_action", "The Edge of Seventeen", "Woody Harrelson"), ("retrieved_after_search", "Woody Harrelson", "born July 23, 1961")]),
 (250, "The Edge of Seventeen Features", "repair_candidate", ["duplicate_training_action", "query_copies_cast_list"],
  "Fourth cast-list query copy; the pre-action film excerpt, query and Harrelson result are the same as row 247.",
  "Keep original for audit, deduplicate and verify all related answer states before selecting any representative.",
  [("before_action", "The Edge of Seventeen", "Woody Harrelson"), ("retrieved_after_search", "Woody Harrelson", "born July 23, 1961")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-025.jsonl")
