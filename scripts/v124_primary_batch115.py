"""Primary examination of v123 internal candidate states 141–150."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (141, "In what country is the city where Larissa Nevierov", "retain_candidate", ["query_uses_visible_birth_city", "retrieval_adds_country"],
  "The initial biography says Nevierov was born in Trieste. The retrieved Trieste article explicitly places that city in northeastern Italy. The query repeats a relevant source sentence, but the missing city-country link is supplied by retrieval.", None,
  [("before_action", "Larissa Nevierov", "born 18 September 1974 in Trieste"), ("retrieved_after_search", "Trieste", "city and seaport in northeastern Italy")]),
 (142, "In what country is the city where Larissa Nevierov", "retain_candidate", [],
  "Birthplace Trieste and Trieste's country Italy are separately visible by answer time. Her Italian nationality alone would not prove her birthplace's country, but the Trieste article does.", None,
  [("before_action", "Larissa Nevierov", "in Trieste"), ("before_action", "Trieste", "northeastern Italy")]),
 (143, "In which city did Sahar Tawfiq study Arabic", "retain_candidate", ["query_uses_visible_university", "retrieval_adds_city"],
  "Her biography connects the requested studies to Al-Azhar University, and the retrieval explicitly places that university in Cairo. The duplicate 'in' at the end of the question is an editorial defect but does not change the referent or answer.", None,
  [("before_action", "Sahar Tawfiq", "studied Arabic language and literature at Al-Azhar University"), ("retrieved_after_search", "Al-Azhar University", "university in Cairo, Egypt")]),
 (144, "In which city did Sahar Tawfiq study Arabic", "retain_candidate", ["minor_question_grammar"],
  "The study-to-university and university-to-Cairo chain is explicit at answer time. Cairo, Egypt answers the city question with a country qualifier; minor grammar should be cleaned before a final export.",
  "Optionally remove the trailing 'in' without changing the evidence chain.",
  [("before_action", "Sahar Tawfiq", "studied Arabic language and literature at Al-Azhar University"), ("before_action", "Al-Azhar University", "university in Cairo, Egypt")]),
 (145, " Hurtigruten was founded in 1893", "retain_candidate", ["query_uses_visible_endpoint", "retrieval_adds_church_area"],
  "The initial Hurtigruten article names Kirkenes as the other endpoint. The Kirkenes retrieval says its main church is in the Haganes area. Both relation edges are present without inferring an area from general town location.", None,
  [("before_action", "Hurtigruten", "between Bergen and Kirkenes"), ("retrieved_after_search", "Kirkenes", "Kirkenes Church, located in the Haganes area")]),
 (146, " Hurtigruten was founded in 1893", "retain_candidate", [],
  "Kirkenes is the stated route endpoint, and its main church is explicitly located in Haganes. The answer names the requested area.", None,
  [("before_action", "Hurtigruten", "between Bergen and Kirkenes"), ("before_action", "Kirkenes", "Kirkenes Church, located in the Haganes area")]),
 (147, "What is both a type of dessert and a dance community", "quarantine_pending_repair", ["dance_name_lexical_conflation", "query_preselects_answer"],
  "The source spells the parents' community 'Meringue' and the dessert article is also Meringue, so literal overlap exists. But the recognized dance name may be a different word, merengue, and the initial context has no parent/community link; the search preselects Meringue. This needs source and intended-relation verification before treating a spelling coincidence as a sound target.",
  "Check the original source and intended dance term; if the relation is merengue, rewrite or discard rather than training a false equivalence.",
  [("before_action", "Meringue", "type of dessert"), ("retrieved_after_search", "Tom Gorrio", "Miami Salsa and Meringue commu")]),
 (148, "What is both a type of dessert and a dance community", "quarantine_pending_repair", ["paired_lexical_conflation"],
  "The answer string Meringue matches both frozen passages, but the dance/community use is suspicious and may reflect a source spelling error rather than a valid same-word concept. The answer should not be approved on exact string matching alone.",
  "Resolve the dance term and review the corrected question-answer pair independently.",
  [("before_action", "Meringue", "type of dessert"), ("before_action", "Tom Gorrio", "Miami Salsa and Meringue commu")]),
 (149, "In which city is the handball club Anne Mette Pedersen", "retain_candidate", ["query_uses_visible_club", "retrieval_adds_city"],
  "The player article names FC Midtjylland Håndbold as her club, and its retrieved article locates it in Ikast. The query uses the visible player-club relation and retrieves the missing city.", None,
  [("before_action", "Anne Mette Pedersen", "player for FC Midtjylland Håndbold"), ("retrieved_after_search", "FC Midtjylland Håndbold", "handball club based in Ikast")]),
 (150, "In which city is the handball club Anne Mette Pedersen", "retain_candidate", [],
  "The club named in Pedersen's initial biography is explicitly based in Ikast in the retrieved club document. The answer is the requested city.", None,
  [("before_action", "Anne Mette Pedersen", "player for FC Midtjylland Håndbold"), ("before_action", "FC Midtjylland Håndbold", "handball club based in Ikast")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-115.jsonl", cohort="v123_internal_candidates")
