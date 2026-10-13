"""Primary examination of exposed historical internal states 41–50."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (41, "What is the birth date of the person", "retain_candidate", ["query_copies_irrelevant_visible_sentence", "exposed_development_set"],
  "The Baldwin article identifies his employer as James McCord. The query copies its opening sentence about Baldwin instead of the employer relation, but the retrieval nevertheless supplies McCord's birth date.", None,
  [("before_action", "Alfred C. Baldwin III", "hired by James McCord"), ("retrieved_after_search", "James W. McCord Jr.", "born June 26, 1924")]),
 (42, "What is the birth date of the person", "retain_candidate", ["exposed_development_set"],
  "Baldwin's article names James McCord as the employer, and McCord's biography gives June 26, 1924. The answer uses McCord's date, not Baldwin's visible 1936 date.", None,
  [("before_action", "Alfred C. Baldwin III", "hired by James McCord"), ("before_action", "James W. McCord Jr.", "born June 26, 1924")]),
 (43, "When was  American actress born", "quarantine_pending_repair", ["query_entity_not_grounded_before_action", "answer_lookahead_risk", "exposed_development_set"],
  "The search names Elizabeth Mitchell even though the question omits her name and none of the initial snippets connects her to both specified films: her snippet mentions The Santa Clause but the Purge title is absent. The returned Purge article later confirms the intersection, but that cannot justify the earlier query. Quarantine the search target pending a source-grounded rewrite.", None,
  [("before_action", "Elizabeth Mitchell", "March 27, 1970"), ("retrieved_after_search", "The Purge: Election Year", "starring Frank Grillo, Elizabeth Mitchell and Mykelti Williamson")]),
 (44, "When was  American actress born", "repair_candidate", ["pair_has_ungrounded_search_target", "exposed_development_set"],
  "The answer date is supported: Mitchell's article gives it, and the post-search Purge article identifies her as a cast member. The pair cannot be reused unchanged because the preceding search chose Mitchell before the second-film relation was visible.", None,
  [("before_action", "Elizabeth Mitchell", "March 27, 1970"), ("before_action", "The Purge: Election Year", "starring Frank Grillo, Elizabeth Mitchell and Mykelti Williamson")]),
 (45, "The Branford Price Millar Library", "quarantine_pending_repair", ["query_entity_not_grounded_before_action", "answer_lookahead_risk", "exposed_development_set"],
  "The query states Portland State University, but the initial retrieved snippets provide only university descriptions and no Branford Price Millar Library relation. The library article appears only after search. This is an answer-key-like target leap rather than a demonstrated grounded hop.", None,
  [("before_action", "Portland State University", "founded in 1946"), ("retrieved_after_search", "Branford Price Millar Library", "library of Portland State University")]),
 (46, "The Branford Price Millar Library", "repair_candidate", ["pair_has_ungrounded_search_target", "exposed_development_set"],
  "The post-search library article establishes Portland State University and its article gives 1946, so the answer is supported at answer time. The preceding search was not justified by initial evidence, making the paired trajectory unsuitable as-is.", None,
  [("before_action", "Branford Price Millar Library", "library of Portland State University"), ("before_action", "Portland State University", "founded in 1946")]),
 (47, "Who was the director of the 2004", "retain_candidate", ["query_uses_question_and_visible_actor", "exposed_development_set"],
  "Jeff Rudom's visible biography names District 13. The query combines the question and visible actor; retrieval yields the 2004 film article identifying director Pierre Morel. The 2009 sequel is a distractor.", None,
  [("before_action", "Jeff Rudom", "films such as \"Revolver\" and \"District 13\""), ("retrieved_after_search", "District 13", "directed by Pierre Morel")]),
 (48, "Who was the director of the 2004", "retain_candidate", ["exposed_development_set"],
  "Rudom's filmography lists District 13; its article matches the 2004 Luc Besson description and names Pierre Morel as director. The answer does not confuse the 2009 sequel's director.", None,
  [("before_action", "Jeff Rudom", "\"District 13\""), ("before_action", "District 13", "directed by Pierre Morel and written and produced by Luc Besson")]),
 (49, "Dheepan was partly inspired", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "Dheepan's article identifies Persian Letters as the literary inspiration but lacks its writing year. The copied sentence retrieves the literary-work article, which gives 1721.", None,
  [("before_action", "Dheepan", "Montesquieu's \"Persian Letters\""), ("retrieved_after_search", "Persian Letters", "written in 1721")]),
 (50, "Dheepan was partly inspired", "retain_candidate", ["exposed_development_set"],
  "The film article identifies Persian Letters as the literary work, and its article says it was written in 1721. The answer correctly excludes the unrelated 1971 film inspiration.", None,
  [("before_action", "Dheepan", "Montesquieu's \"Persian Letters\""), ("before_action", "Persian Letters", "written in 1721")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-040.jsonl", cohort="v117_exposed_internal50")
