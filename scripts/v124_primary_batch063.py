"""Primary examination of v122 internal candidate states 171–180."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (171, '"Burn" and "Walk Away"', "retain_candidate", ["query_copies_visible_film_relation"],
  "The album snippet identifies Reign of Fire as the film whose end credits used the two songs. The search follows that visible title and retrieves the film article naming director Rob Bowman.", None,
  [("before_action", "Resonance (Mad at Gravity album)", "end title credits of the 2002 action film Reign of Fire"), ("retrieved_after_search", "Reign of Fire (film)", "directed by Rob Bowman")]),
 (172, '"Burn" and "Walk Away"', "retain_candidate", [],
  "The songs are linked to Reign of Fire, and the film article explicitly credits Rob Bowman as director. The answer is supported.", None,
  [("before_action", "Resonance (Mad at Gravity album)", "end title credits of the 2002 action film Reign of Fire"), ("before_action", "Reign of Fire (film)", "directed by Rob Bowman")]),
 (173, "When was the inventor of the wobble board", "repair_candidate", ["paired_answer_has_nonanswer_markup"],
  "The wobble board article names Rolf Harris, and the search retrieves his birth date. The paired target includes the literal fragment '(born' and omits a closing parenthesis, which is an answer-extraction defect despite the date being visible.",
  "Use the clean date '30 March 1930' as the answer after replaying the pair and independent review.",
  [("before_action", "Wobble board", "invented and popularized by the Australian musician and artist Rolf Harris"), ("retrieved_after_search", "Rolf Harris", "born 30 March 1930")]),
 (174, "When was the inventor of the wobble board", "repair_candidate", ["answer_extraction_leaks_source_markup"],
  "The frozen answer '(born 30 March 1930' is a copied source fragment rather than a clean date. The intended birth date is supported, but the label must be corrected before use.",
  "Replace only in a separately versioned replayed pair with '30 March 1930'; keep the frozen target unchanged.",
  [("before_action", "Wobble board", "invented and popularized by the Australian musician and artist Rolf Harris"), ("before_action", "Rolf Harris", "born 30 March 1930")]),
 (175, "Who starred in a Northern Irish band's music video", "retain_candidate", ["query_uses_visible_comic_identity"],
  "The initial Andy Dick biography explicitly says he is best known as a comic. Searching his name retrieves the Northern Irish band Ash's Envy song article, which says its video featured him.", None,
  [("before_action", "Andy Dick", "Best known as a comic"), ("retrieved_after_search", "Envy (song)", "Northern Irish band Ash"), ("retrieved_after_search", "Envy (song)", "music video featured Andy Dick")]),
 (176, "Who starred in a Northern Irish band's music video", "retain_candidate", [],
  "The Envy music video article names Andy Dick, and his biography supplies the comic description. The answer follows the same person.", None,
  [("before_action", "Envy (song)", "music video featured Andy Dick"), ("before_action", "Andy Dick", "Best known as a comic")]),
 (177, "What daughter of Catherine of Guise", "quarantine_pending_repair", ["client_alias_not_explicitly_bound", "similar_names_not_identity_proof"],
  "The initial biography names Marie Louise Gonzaga as Catherine of Guise's daughter. The retrieved coiffeur article lists Princess Marie de Gonzague as a client, but these two differently named people are not explicitly equated in the visible evidence. Treating similarity of Gonzaga/Gonzague as a verified identity would risk a same-family splice.",
  "Find an explicit alias or identity source before retaining the pair; otherwise exclude it. Do not infer equivalence from name resemblance alone.",
  [("before_action", "Marie Louise Gonzaga", "Catherine of Guise"), ("retrieved_after_search", "Champagne (coiffeur)", "aristocratic clients included Princess Marie de Gonzague")]),
 (178, "What daughter of Catherine of Guise", "quarantine_pending_repair", ["unverified_client_alias_binding"],
  "Marie Louise Gonzaga is supported as Catherine of Guise's daughter; the client source instead names Princess Marie de Gonzague. The provided text lacks a direct alias statement, so the full daughter-to-client chain is not established to the required standard.",
  "Keep quarantined pending explicit identity evidence and fresh independent review.",
  [("before_action", "Marie Louise Gonzaga", "Catherine of Guise"), ("before_action", "Champagne (coiffeur)", "Princess Marie de Gonzague")]),
 (179, " Ivory Lee Brown is the uncle", "quarantine_pending_repair", ["conflicting_team_snapshots", "unanchored_team_question"],
  "The initial Brown article calls his nephew Adrian Peterson a Minnesota Vikings running back, while the retrieved Peterson biography calls him a New Orleans Saints running back and says he was drafted by the Vikings. The unanchored question does not select a season or source snapshot, so the frozen Saints answer conflicts with the initial team attribution.",
  "Exclude the frozen pair; create a dated Peterson-team question only if authoritative period evidence is available and replayed.",
  [("before_action", "Ivory Lee Brown", "uncle of Minnesota Vikings running back Adrian Peterson"), ("retrieved_after_search", "Adrian Peterson", "running back for the New Orleans Saints")]),
 (180, " Ivory Lee Brown is the uncle", "quarantine_pending_repair", ["answer_conflicts_with_initial_team_snapshot", "unanchored_team_question"],
  "New Orleans Saints matches the retrieved Peterson biography but not the initial Brown article's Minnesota Vikings description. Without a time anchor, choosing one snapshot is arbitrary for the asked team.",
  "Keep quarantined until a dated, consistent question and answer are independently reviewed.",
  [("before_action", "Ivory Lee Brown", "uncle of Minnesota Vikings running back Adrian Peterson"), ("before_action", "Adrian Peterson", "running back for the New Orleans Saints")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-063.jsonl", cohort="v122_internal_candidates")
