"""Primary examination of v123 internal candidate states 111–120."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (111, "Still Reigning was filmed at a venue", "retain_candidate", ["query_uses_visible_venue", "retrieval_adds_city"],
  "The initial Still Reigning article names Augusta Civic Center as the filming venue. The query quotes that venue relation and retrieves the center article locating it in Augusta, Maine.", None,
  [("before_action", "Still Reigning", "Filmed at the Augusta Civic Center"), ("retrieved_after_search", "Augusta Civic Center", "in Augusta, Maine")]),
 (112, "Still Reigning was filmed at a venue", "retain_candidate", [],
  "The DVD was filmed at Augusta Civic Center, whose retrieved article locates it in Augusta, Maine. The city answer with state clarification is supported.", None,
  [("before_action", "Still Reigning", "Filmed at the Augusta Civic Center"), ("before_action", "Augusta Civic Center", "in Augusta, Maine")]),
 (113, "The US state in which Elijah Cutler Behunin Cabin", "repair_candidate", ["query_uses_adjacent_town_not_missing_border", "retrieval_adds_utah_border"],
  "The initial cabin article already places the structure in Utah. The query selects Caineville, another Utah place tied to Elijah Behunin, rather than the missing question of which state borders Utah on the east. The retrieval does find a Utah page with Colorado on the east, but that success does not make the town query an apt supervision target.",
  "Search Utah eastern border directly from the initial cabin location, then replay.",
  [("before_action", "Elijah Cutler Behunin Cabin", "Capitol Reef National Park in Utah"), ("before_action", "Caineville, Utah", "founded by Elijah Cutler Behunin"), ("retrieved_after_search", "Utah", "bordered by Colorado to the east")]),
 (114, "The US state in which Elijah Cutler Behunin Cabin", "repair_candidate", ["paired_query_target_mismatch", "answer_supported"],
  "Colorado is explicitly listed as Utah's eastern neighbor, and the cabin's initial article states it is in Utah. The paired search detoured through Caineville instead of the needed border relation.",
  "Preserve Colorado only after a Utah-border query replay.",
  [("before_action", "Elijah Cutler Behunin Cabin", "Capitol Reef National Park in Utah"), ("before_action", "Utah", "bordered by Colorado to the east")]),
 (115, "In what country did a popular actress play a role", "quarantine_pending_repair", ["answer_not_country", "actress_and_film_not_unique"],
  "The initial input contains multiple coming-of-age films and actresses. The query picks Boys (2003), and retrieval finds Genelia D'Souza's role; however, the frozen answer is South Indian cinema, a regional film-industry category rather than a country. The question also lacks a unique actress or film clue.",
  "Specify Boys and Genelia if the intended answer is India, then create a new answer pair; do not reuse the current target.",
  [("before_action", "Go North (film)", "coming-of-age thriller film"), ("before_action", "Boys (2003 film)", "Indian Tamil coming-of-age film"), ("retrieved_after_search", "Genelia D'Souza", "South Indian cinema")]),
 (116, "In what country did a popular actress play a role", "quarantine_pending_repair", ["paired_non_country_answer", "selection_not_unique"],
  "South Indian cinema is not a country. The Boys article calls the film Indian and lists Genelia, but the frozen wording admits other coming-of-age films and does not uniquely identify an actress. A correct country string alone would not fix the selection issue.",
  "Replace the question with a uniquely identified film/actress and a country target, then fully replay and review.",
  [("before_action", "Boys (2003 film)", "Indian Tamil coming-of-age film"), ("before_action", "Boys (2003 film)", "Genelia"), ("before_action", "Genelia D'Souza", "South Indian cinema")]),
 (117, "In which county is the College Park Centre", "retain_candidate", ["name_variant_centre_center", "query_uses_visible_arlington", "retrieval_adds_county"],
  "The initial College Park Center article places the arena on the UT Arlington campus in Arlington, Texas. A query quoting that location retrieves Arlington's article, which places the city in Tarrant County. The question's Centre spelling is a transparent orthographic variant, with the venue otherwise clear.", None,
  [("before_action", "College Park Center", "campus in Arlington, Texas"), ("retrieved_after_search", "Arlington, Texas", "located in Tarrant County")]),
 (118, "In which county is the College Park Centre", "retain_candidate", ["answer_trailing_period"],
  "College Park Center is in Arlington, Texas, and Arlington's article places it in Tarrant County. The trailing period in the frozen answer is formatting, not a different county.", None,
  [("before_action", "College Park Center", "campus in Arlington, Texas"), ("before_action", "Arlington, Texas", "located in Tarrant County")]),
 (119, "In between Dicky Barrett and Skin", "quarantine_pending_repair", ["question_semantics_uninterpretable", "query_unrelated_nivea_brand", "answer_visible_before_search"],
  "The question's 'In between Dicky Barrett and Skin, who is know as Skin?' has no defined relation between the two performers. The initial Skin musician article already names Deborah Anne Dyer, while the query inserts Nivea, a skin-care brand with no relevant link. The retrieved Dicky Barrett pages do not clarify 'in between'.",
  "If the intended target is Skin's birth name, ask that directly without the unexplained Dicky Barrett comparison; rebuild the pair.",
  [("before_action", "Skin (musician)", "Deborah Anne Dyer"), ("before_action", "Nivea", "personal care brand"), ("retrieved_after_search", "Dicky Barrett", "American singer")]),
 (120, "In between Dicky Barrett and Skin", "quarantine_pending_repair", ["paired_uninterpretable_question", "unsupported_between_relation"],
  "Deborah Anne Dyer is supported as the birth name of singer Skin, but no action-time source explains the question's 'between Dicky Barrett and Skin' relation. The answer is directly visible before search and does not repair the malformed task.",
  "Use a new, single-target question for Skin's name if wanted; quarantine the original pair.",
  [("before_action", "Skin (musician)", "Deborah Anne Dyer"), ("before_action", "Dicky Barrett", "American singer")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-112.jsonl", cohort="v123_internal_candidates")
