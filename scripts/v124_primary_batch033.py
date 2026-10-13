"""Primary case examination of original training rows 321–330."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (321, "What medium-sized bear native", "quarantine_pending_repair", ["same_name_park_ambiguity", "cross_park_evidence_splice"],
  "The pre-action context contains both the New York and Idaho Harriman State Parks. Mary Averell's donation connects to the New York park, while the retrieved black-bear sighting is explicitly the Idaho park. The question omits a state, so combining the New York clue with the Idaho animal record is a cross-entity splice; the answer is not unambiguously supported.",
  "Inspect the official question provenance and paired answer. Keep isolated unless a single park can be specified with a valid same-park animal source; do not silently choose Idaho from search ranking.",
  [("before_action", "Mary Williamson Averell", "donating the land that became Harriman State Park"), ("before_action", "Harriman State Park (Idaho)", "occasional black or grizzly bear"), ("retrieved_after_search", "American black bear", "medium-sized bear native to North America"), ("retrieved_after_search", "Harriman State Park (Idaho)", "occasional black or grizzly bear")]),
 (322, "Who wrote a work that described", "repair_candidate", ["generic_query", "longest_reign_link_not_retrieved"],
  "The query 'history until' retrieves the named work and Voltaire authorship, and the work extends to the age of Louis XIV. This state does not retrieve evidence that Louis XIV is the longest-serving European monarch, so the superlative clue remains unverified at this action boundary.",
  "Check paired states for a valid reign-duration source. If absent, do not count the complete question as supported; test a question-grounded query that establishes both the reign clue and work-author chain.",
  [("retrieved_after_search", "Essai sur les mœurs et l'esprit des nations", "a work by the French writer, historian, and philosopher Voltaire"), ("retrieved_after_search", "Essai sur les mœurs et l'esprit des nations", "until the dawn of the age of Louis XIV")]),
 (323, "Jasminum multipartitum", "retain_candidate", ["query_uses_candidate_country", "duplicate_training_action_group"],
  "The pre-action Zimbabwe excerpt matches the river-boundary country clue, but not jasmine native range. The query uses the candidate country and retrieves the jasmine article explicitly listing Zimbabwe among its native countries. This verifies the missing relation.", None,
  [("before_action", "Zimbabwe", "between the Zambezi and Limpopo Rivers"), ("retrieved_after_search", "Jasminum multipartitum", "native to Mozambique, Zimbabwe, Swaziland and South Africa")]),
 (324, "Jasminum multipartitum", "repair_candidate", ["duplicate_training_action"],
  "Same question, country-candidate query and jasmine native-range evidence as row 323, with a different set of geographic distractors.",
  "Keep original input but deduplicate this path after answer-state check.",
  [("before_action", "Zimbabwe", "between the Zambezi and Limpopo Rivers"), ("retrieved_after_search", "Jasminum multipartitum", "native to Mozambique, Zimbabwe, Swaziland and South Africa")]),
 (325, "Jasminum multipartitum", "retain_candidate", ["query_known_plant_fills_country"],
  "The plant article already lists Zimbabwe as one native country, but the precise river-boundary clue is missing. Querying the plant name retrieves the Zimbabwe article with that geographic description, completing a distinct reverse-evidence path.", None,
  [("before_action", "Jasminum multipartitum", "native to Mozambique, Zimbabwe, Swaziland and South Africa"), ("retrieved_after_search", "Zimbabwe", "between the Zambezi and Limpopo Rivers")]),
 (326, "Which Gujarati city", "repair_candidate", ["generic_query", "retrieval_hit_not_strategy_proof", "ranking_time_unanchored"],
  "From empty context, 'city India' retrieves Surat at rank two with the third-cleanest-city claim, but the query omits Gujarat and the ranking phrase. It is too generic to serve as a reliable target-seeking example; the ranking is also undated in the excerpt.",
  "Determine the source's dated ranking scope if available, then replay a query grounded in the question's Gujarati and third-cleanest clues. Do not assert a present ranking without an as-of source.",
  [("retrieved_after_search", "Surat", "in the Indian state of Gujarat"), ("retrieved_after_search", "Surat", "3rd \"cleanest city of India\"")]),
 (327, "Which actress played Beth Wallace", "retain_candidate", ["query_known_character_verifies_award", "duplicate_training_action_group"],
  "The character excerpt names Kelli McCarty as performer. Searching the known character name retrieves McCarty's biography confirming that she was crowned Miss USA in 1991, completing the second clue.", None,
  [("before_action", "Beth Wallace", "played by Kelli McCarty"), ("retrieved_after_search", "Kelli McCarty", "crowned Miss USA in 1991")]),
 (328, "Which actress played Beth Wallace", "repair_candidate", ["duplicate_training_action"],
  "Same character-name query and Kelli McCarty Miss USA result as row 327 with identical pre-action evidence.",
  "Preserve original, deduplicate after the paired answer state is checked.",
  [("before_action", "Beth Wallace", "played by Kelli McCarty"), ("retrieved_after_search", "Kelli McCarty", "crowned Miss USA in 1991")]),
 (329, "Which actress played Beth Wallace", "retain_candidate", [],
  "The Beth Wallace article explicitly credits Kelli McCarty, and her biography explicitly states the 1991 Miss USA title and repeats the role. The answer is supported.", None,
  [("before_action", "Beth Wallace", "played by Kelli McCarty"), ("before_action", "Kelli McCarty", "crowned Miss USA in 1991")]),
 (330, "Name the 2005 romantic comedy", "repair_candidate", ["generic_query", "screenwriter_link_not_retrieved"],
  "Searching only 'Asking' retrieves The Wedding Date article, which establishes the 2005 romantic comedy and Elizabeth Young novel adaptation. This state does not establish that its writer also wrote What Happens in Vegas; the requested writer link must be checked before treating the full question as supported.",
  "Trace same-question states for both films' writing credits and replay a question-grounded query if missing. Keep this broad-query hit diagnostic until the whole chain is verified.",
  [("retrieved_after_search", "The Wedding Date", "a 2005 American romantic comedy film"), ("retrieved_after_search", "The Wedding Date", "Based on the novel \"Asking for Trouble\" by Elizabeth Young")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-033.jsonl")
