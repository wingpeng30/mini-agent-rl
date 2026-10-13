"""Primary examination of v123 internal candidate states 161–170."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (161, "What is the largest e-commerce company in Vietnam", "repair_candidate", ["query_targets_ascend_distractor", "time_sensitive_largest_claim", "question_typo"],
  "The initial Sendo article calls Lazada the market leader in Vietnam and retrieval lists Lazada's six-country operation. The query instead copies an Ascend Group sentence, which lacks the full asked footprint. 'Largest' is time-sensitive, and the question merges 'inIndonesia'.",
  "Time-bound the market-leader claim, repair the typo, and replay a Lazada-oriented query grounded in Sendo's visible comparison.",
  [("before_action", "Sendo.vn", "market leader Lazada"), ("before_action", "Ascend Group", "headquartered in Bangkok, Thailand"), ("retrieved_after_search", "Lazada Group", "Indonesia, Malaysia, Philippines, Singapore, Thailand, and Vietnam")]),
 (162, "What is the largest e-commerce company in Vietnam", "repair_candidate", ["paired_query_distractor", "unanchored_largest_claim"],
  "Lazada Group is supported as the site named Vietnam's market leader and as operating across the listed countries as of 2014. The original question has no date for a changeable ranking, and the paired search targets another company.",
  "Use a dated wording and fresh paired retrieval review.",
  [("before_action", "Sendo.vn", "market leader Lazada"), ("before_action", "Lazada Group", "As of 2014, Lazada Group operated sites")]),
 (163, "What restaurant did the artist of", "retain_candidate", ["query_uses_visible_artist", "retrieval_adds_restaurant"],
  "The song document identifies Tracy Lawrence as its recording artist. Search on that visible relation retrieves his biography naming Live At Libby's as the restaurant where he started.", None,
  [("before_action", "Runnin' Behind", "recorded by American country music artist Tracy Lawrence"), ("retrieved_after_search", "Tracy Lawrence", "started at a country music restaurant called")]),
 (164, "What restaurant did the artist of", "retain_candidate", ["answer_includes_source_quote_marks"],
  "Tracy Lawrence is the named recording artist, and the biography directly names the restaurant Live At Libby's. Source quote marks in the answer do not alter its identity.", None,
  [("before_action", "Runnin' Behind", "recorded by American country music artist Tracy Lawrence"), ("before_action", "Tracy Lawrence", "restaurant called \"Live At Libby's\"")]),
 (165, "Metropolis includes a commentary", "repair_candidate", ["query_ignores_visible_person_names", "retrieval_adds_chef_identity"],
  "The first-pass Metropolis article already lists Samantha Brown and Andrew Zimmern as commentators. The search query copies the series premiere date instead of testing which commentator is a chef; retrieval happens to add Zimmern's chef role. The answer can be supported, but the search target is misaligned.",
  "Search Andrew Zimmern occupation from the visible commentator list and replay.",
  [("before_action", "Metropolis (documentary series)", "Samantha Brown and Andrew Zimmern"), ("retrieved_after_search", "Andrew Zimmern", "television personality, chef")]),
 (166, "Metropolis includes a commentary", "repair_candidate", ["paired_query_target_mismatch", "answer_supported"],
  "The Metropolis commentary list contains Andrew Zimmern and his biography explicitly identifies him as television personality and chef. The paired query about episode premiere date did not deliberately target that missing attribute.",
  "Re-review with an occupation-directed query.",
  [("before_action", "Metropolis (documentary series)", "Samantha Brown and Andrew Zimmern"), ("before_action", "Andrew Zimmern", "television personality, chef")]),
 (167, "What mockumentary comedy film featured", "repair_candidate", ["answer_visible_before_search", "query_best_in_show_distractor"],
  "The initial Pittsburgh article already lists Craig Kilborn among its featured people. The query appends Best in Show, a different mockumentary, and retrieval only supplies Kilborn's Daily Show identity. That can confirm the target, but the chosen distractor does not aim at the missing host link.",
  "Query Craig Kilborn's original Daily Show host role, then replay the pair.",
  [("before_action", "Pittsburgh (2006 film)", "Conan O'Brien and Craig Kilborn"), ("before_action", "Best in Show (film)", "mockumentary comedy film"), ("retrieved_after_search", "Craig Kilborn", "original host of \"The Daily Show\"")]),
 (168, "What mockumentary comedy film featured", "repair_candidate", ["paired_distractor_query", "answer_supported"],
  "Pittsburgh is a mockumentary comedy featuring Craig Kilborn, whom the retrieved biography calls The Daily Show's original host. The answer is supported, but its paired search inserted another film rather than querying that host relation.",
  "Retain the answer only after a relevant query replay and independent review.",
  [("before_action", "Pittsburgh (2006 film)", "mockumentary comedy film"), ("before_action", "Pittsburgh (2006 film)", "Conan O'Brien and Craig Kilborn"), ("before_action", "Craig Kilborn", "original host of \"The Daily Show\"")]),
 (169, "What was the role of the man", "retain_candidate", ["query_uses_visible_george_ii", "retrieval_adds_empire_title"],
  "The province source names King George II as its namesake; his retrieved article calls him Prince-elector of the Holy Roman Empire. The name-bearing query is a plausible route to the missing title.", None,
  [("before_action", "Politics of Georgia (U.S. state)", "named after King George II"), ("retrieved_after_search", "George II of Great Britain", "Prince-elector of the Holy Roman Empire")]),
 (170, "What was the role of the man", "retain_candidate", [],
  "The same King George II named in the province article is described as Prince-elector of the Holy Roman Empire. The answer supplies the requested role.", None,
  [("before_action", "Politics of Georgia (U.S. state)", "named after King George II"), ("before_action", "George II of Great Britain", "Prince-elector of the Holy Roman Empire")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-117.jsonl", cohort="v123_internal_candidates")
