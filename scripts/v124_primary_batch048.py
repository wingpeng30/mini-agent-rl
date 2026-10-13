"""Primary examination of v122 internal candidate states 21–30."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (21, "Which actor had a role in both", "retain_candidate", ["query_copies_distractor_actor_bio", "retrieval_succeeds_despite_query_drift"],
  "The initial Bridesmaids cast includes Matt Lucas. The query copies Tucker Albrizzi's unrelated television-role sentence, but also includes both film titles and retrieves Matt Lucas's Alice in Wonderland role. The answer is supported; the copied evidence is a poor query choice.", None,
  [("before_action", "Bridesmaids (2011 film)", "Matt Lucas"), ("before_action", "Tucker Albrizzi", "Tyler Duncan on \"Big Time Rush\""), ("retrieved_after_search", "Matt Lucas", "Tweedle Dum and Tweedle Dee in \"Alice in Wonderland\"")]),
 (22, "Which actor had a role in both", "retain_candidate", [],
  "Bridesmaids lists Matt Lucas in a supporting role, and his biography names his Alice in Wonderland parts. The intersection answer is supported despite the paired search's distractor text.", None,
  [("before_action", "Bridesmaids (2011 film)", "Matt Lucas"), ("before_action", "Matt Lucas", "Tweedle Dum and Tweedle Dee in \"Alice in Wonderland\"")]),
 (23, "The Shrewsbury & Newport Canals Trust", "retain_candidate", ["query_copies_visible_context", "multiple_canals_named"],
  "The trust article names the Shrewsbury Canal and Newport Arm of the Shropshire Union Canal. Its copied sentence retrieves the latter canal's article, which calls it a navigable canal in England. The singular wording should be checked before promotion because two restoration targets are named.", None,
  [("before_action", "Shrewsbury &amp; Newport Canals Trust", "Shrewsbury Canal and the Newport Arm of the Shropshire Union Canal"), ("retrieved_after_search", "Shropshire Union Canal", "navigable canal in England")]),
 (24, "The Shrewsbury & Newport Canals Trust", "retain_candidate", ["multiple_canals_named"],
  "The trust works on the Newport Arm of the Shropshire Union Canal, and that canal is explicitly in England. The country answer is supported for that named arm; the question's singular 'a canal' remains imprecise.", None,
  [("before_action", "Shrewsbury &amp; Newport Canals Trust", "Newport Arm of the Shropshire Union Canal"), ("before_action", "Shropshire Union Canal", "navigable canal in England")]),
 (25, "Are both Junta and Carrom", "quarantine_pending_repair", ["query_follows_wrong_person_entity", "negative_claim_unverified"],
  "The question asks about the Junta board game and Carrom. The query appends Thomas Junta, a homicide-case person, because of lexical overlap; retrieval establishes Carrom's child participation but supplies no evidence that the board game is or is not commonly played by children.", None,
  [("before_action", "Junta (game)", "Junta is a board game"), ("before_action", "Thomas Junta", "an American hockey dad"), ("retrieved_after_search", "Carrom", "commonly played by families, including the children")]),
 (26, "Are both Junta and Carrom", "quarantine_pending_repair", ["negative_claim_from_absence", "wrong_entity_query_in_pair"],
  "Carrom is explicitly commonly played by children, while the visible Junta game snippet only describes its theme and rules. Silence about child play does not prove 'no'; the proposed negative answer is unsupported by this evidence.", None,
  [("before_action", "Junta (game)", "Junta is a board game"), ("before_action", "Carrom", "commonly played by families, including the children")]),
 (27, "Where is the United Kingdom's largest", "repair_candidate", ["query_targets_canadian_waterpark", "unanchored_largest_claim"],
  "The initial Wet N Wild snippet says its UK-largest title was subsequently claimed by Sandcastle Waterpark. The query instead repeats the Canadian World Waterpark title; retrieval happens to return Sandcastle's location. The unanchored superlative and off-target search both require repair.", None,
  [("before_action", "Wet N Wild (North Shields)", "a title subsequently claimed by Sandcastle Waterpark"), ("before_action", "World Waterpark", "Edmonton, Alberta, Canada"), ("retrieved_after_search", "Sandcastle Water Park (Blackpool)", "South Beach in Blackpool, Lancashire, England")]),
 (28, "Where is the United Kingdom's largest", "repair_candidate", ["unanchored_largest_claim", "paired_search_off_target"],
  "The returned Sandcastle article locates it in Blackpool, and Wet N Wild's snippet says Sandcastle subsequently claimed the UK-largest title. The title has no as-of date and the paired query targets a Canadian park, so the answer pair needs a date/source-qualified rewrite.", None,
  [("before_action", "Wet N Wild (North Shields)", "a title subsequently claimed by Sandcastle Waterpark"), ("before_action", "Sandcastle Water Park (Blackpool)", "South Beach in Blackpool, Lancashire, England")]),
 (29, "Who both inflated the Florentine Republic", "quarantine_pending_repair", ["false_birth_year_1496_vs_1469", "undefined_inflated_relation", "query_repeats_irrelevant_title"],
  "The question asks for someone born 3 May 1496, but the retrieved Machiavelli biography says 3 May 1469. The source describes military reforms, not 'inflating' the republic. Repeating the reforms title retrieves Machiavelli but cannot fix the false premises.", None,
  [("before_action", "Florentine military reforms", "initiated by Florentine politician and political theorist Niccolò Machiavelli"), ("retrieved_after_search", "Niccolò Machiavelli", "3 May 1469")]),
 (30, "Who both inflated the Florentine Republic", "quarantine_pending_repair", ["answer_contradicts_question_birth_year", "undefined_inflated_relation"],
  "The answer Niccolò Machiavelli contradicts the question's 1496 birth year: his visible biography gives 1469. The alleged 'inflated' relation also has no source meaning. Quarantine rather than treat a near-digit match as a valid answer.", None,
  [("before_action", "Florentine military reforms", "initiated by Florentine politician and political theorist Niccolò Machiavelli"), ("before_action", "Niccolò Machiavelli", "3 May 1469")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-048.jsonl", cohort="v122_internal_candidates")
