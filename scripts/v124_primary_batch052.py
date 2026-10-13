"""Primary examination of v122 internal candidate states 61–70."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (61, "The 2006 disaster drama film starring", "retain_candidate", ["answer_fact_already_visible", "search_verifies_actor_identity"],
  "The World Trade Center film article already states the September 11 basis and lists Michael Shannon; its copied cast sentence retrieves Shannon's biography and verifies the two actor-credit clues. The search validates identity rather than finding the answer fact.", None,
  [("before_action", "World Trade Center (film)", "based on the September 11 attacks"), ("before_action", "World Trade Center (film)", "Michael Shannon"), ("retrieved_after_search", "Michael Shannon", "\"Revolutionary Road\" (2008)"), ("retrieved_after_search", "Michael Shannon", "\"99 Homes\" (2014)")]),
 (62, "The 2006 disaster drama film starring", "retain_candidate", ["answer_fact_was_visible_before_search"],
  "The film is World Trade Center; Michael Shannon's biography matches both clue films, and the World Trade Center article explicitly says it is based on the September 11 attacks. The answer was already present before search, so this is not evidence of new answer retrieval.", None,
  [("before_action", "World Trade Center (film)", "based on the September 11 attacks"), ("before_action", "World Trade Center (film)", "Michael Shannon"), ("before_action", "Michael Shannon", "\"Revolutionary Road\" (2008)"), ("before_action", "Michael Shannon", "\"99 Homes\" (2014)")]),
 (63, "Gaston Dufresne had an American composer", "retain_candidate", ["query_copies_visible_student_relation"],
  "Dufresne's article identifies Leroy Anderson as the American composer student. The copied student sentence retrieves Anderson's biography and John Williams's quoted description.", None,
  [("before_action", "Gaston Dufresne", "American composer Leroy Anderson"), ("retrieved_after_search", "Leroy Anderson", "one of the great American masters of light orchestral music")]),
 (64, "Gaston Dufresne had an American composer", "retain_candidate", [],
  "The Dufresne article identifies Leroy Anderson, and Anderson's biography contains the requested description. The answer is supported.", None,
  [("before_action", "Gaston Dufresne", "American composer Leroy Anderson"), ("before_action", "Leroy Anderson", "one of the great American masters of light orchestral music")]),
 (65, "Kate Kendall plays Lauren Turner", "repair_candidate", ["query_copies_irrelevant_episode_superlative", "retrieval_succeeds_despite_query_drift"],
  "The Kate Kendall snippet already identifies Neighbours as the relevant soap. The query instead copies an Episode 7202 sentence about a three-hander; it retrieves the show's creator, but the selected search text does not target the missing creator relation.", None,
  [("before_action", "Kate Kendall", "soap opera \"Neighbours\""), ("before_action", "Episode 7202", "first three-hander in 30 years"), ("retrieved_after_search", "Neighbours", "created by TV executive Reg Watson")]),
 (66, "Kate Kendall plays Lauren Turner", "repair_candidate", ["paired_search_query_drift", "answer_supported_at_action_time"],
  "Reg Watson is supported as Neighbours' creator, and Kendall's role is supported. The paired search follows an episode-format detail unrelated to the missing creator, so the pair should use a targeted search before reuse.", None,
  [("before_action", "Kate Kendall", "soap opera \"Neighbours\""), ("before_action", "Neighbours", "created by TV executive Reg Watson")]),
 (67, "The actor who inspired the movie Partner", "repair_candidate", ["actor_vs_source_film_relation_error", "query_copies_visible_source_film"],
  "Partner's article says it was inspired by the film Hitch starring Will Smith. It does not say the actor personally inspired Partner. The query retrieves Smith's birth year, but the question must name the source film and actor relation accurately.", None,
  [("before_action", "Partner (2007 film)", "inspired by the hollywood movie Will Smith starrer Hitch (2005)"), ("retrieved_after_search", "Will Smith", "born September 25, 1968")]),
 (68, "The actor who inspired the movie Partner", "repair_candidate", ["actor_vs_source_film_relation_error"],
  "Will Smith's 1968 birth year is visible, yet the source attributes Partner's inspiration to the Will Smith film Hitch, not to Smith himself. Repair the pair to ask about the star of the source film.", None,
  [("before_action", "Partner (2007 film)", "inspired by the hollywood movie Will Smith starrer Hitch (2005)"), ("before_action", "Will Smith", "born September 25, 1968")]),
 (69, "Which English actor co-wrote", "retain_candidate", ["search_confirms_visible_answer_qualifier"],
  "The film article already names Sacha Baron Cohen as co-writer and star. The query copies a director sentence but retrieves Cohen's biography confirming the English-actor qualifier; the answer fact itself was already visible.", None,
  [("before_action", "The Dictator (2012 film)", "co-written by and starring Sacha Baron Cohen"), ("retrieved_after_search", "Sacha Baron Cohen", "is an English actor")]),
 (70, "Which English actor co-wrote", "retain_candidate", [],
  "The Dictator article says Sacha Baron Cohen co-wrote and starred in it, and his biography identifies him as an English actor. The answer is supported.", None,
  [("before_action", "The Dictator (2012 film)", "co-written by and starring Sacha Baron Cohen"), ("before_action", "Sacha Baron Cohen", "is an English actor")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-052.jsonl", cohort="v122_internal_candidates")
