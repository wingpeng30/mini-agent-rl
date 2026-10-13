"""Primary examination of v122 internal candidate states 41–50."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (41, "According to the 2010 census", "retain_candidate", ["query_copies_visible_birthplace_fragment"],
  "Captain Robinson's biography identifies Miller Place as his birthplace. The query copies an awkward date-plus-place fragment, but retrieval provides Miller Place's 2010 CDP population.", None,
  [("before_action", "Captain Robert Robinson", "born in Miller Place, New York"), ("retrieved_after_search", "Miller Place, New York", "population was 12,339 at the 2010 census")]),
 (42, "According to the 2010 census", "retain_candidate", [],
  "Robinson's birthplace is Miller Place; the matching CDP article reports 12,339 in the 2010 census. The answer avoids similarly named Captain Cook and Robinson Pond distractors.", None,
  [("before_action", "Captain Robert Robinson", "born in Miller Place, New York"), ("before_action", "Miller Place, New York", "population was 12,339 at the 2010 census")]),
 (43, "What was the population, within the town limits", "retain_candidate", ["query_copies_visible_location", "town_limits_scope_preserved"],
  "The Tide's article places the team in Upper Marlboro, Maryland. Searching that sentence retrieves the town article with the explicitly bounded 2010 town-limits population, distinct from Greater Upper Marlboro.", None,
  [("before_action", "Chesapeake Tide", "based in Upper Marlboro, Maryland"), ("retrieved_after_search", "Upper Marlboro, Maryland", "population within the town limits was 631 at the 2010 U.S. Census")]),
 (44, "What was the population, within the town limits", "retain_candidate", ["town_limits_scope_preserved"],
  "The team was based in Upper Marlboro, and that town's article gives 631 residents inside the town limits at the 2010 census. The answer preserves the requested scope.", None,
  [("before_action", "Chesapeake Tide", "based in Upper Marlboro, Maryland"), ("before_action", "Upper Marlboro, Maryland", "population within the town limits was 631 at the 2010 U.S. Census")]),
 (45, "How many No. 1 country singles", "repair_candidate", ["question_typo_how_for_who", "query_copies_visible_artist"],
  "Old Coyote Town identifies Don Williams as its recording artist, and retrieval gives his 17 number-one country hits. The question says 'artist how recorded' instead of 'who recorded'; correct the malformed relation before reuse.", None,
  [("before_action", "Old Coyote Town", "recorded by American country music artist Don Williams"), ("retrieved_after_search", "Don Williams", "amassing 17 number one country hits")]),
 (46, "How many No. 1 country singles", "repair_candidate", ["question_typo_how_for_who"],
  "The numerical answer 17 is supported for Don Williams, but the paired question has a malformed 'how recorded' clause. A wording-only correction must still receive a new binding and review before reuse.", None,
  [("before_action", "Old Coyote Town", "recorded by American country music artist Don Williams"), ("before_action", "Don Williams", "amassing 17 number one country hits")]),
 (47, "WCCW-FM serves the city", "quarantine_pending_repair", ["ungrounded_city_selected_before_station_evidence", "answer_lookahead_risk"],
  "The initial snippets give Traverse City's population but do not mention WCCW-FM. The query names Traverse City even though the question supplies only the station; the station-city link is returned only after the query. This is a lookahead target selection, not a grounded search hop.", None,
  [("before_action", "Traverse City, Michigan", "population was 14,674 at the 2010 census"), ("retrieved_after_search", "WCCW-FM", "in Traverse City, Michigan")]),
 (48, "WCCW-FM serves the city", "repair_candidate", ["paired_search_ungrounded_city", "answer_supported_at_action_time"],
  "At answer time WCCW-FM's article connects the station to Traverse City and the city article gives 14,674 in 2010. The preceding query selected that city without a visible station relation, so the pair cannot be reused unchanged.", None,
  [("before_action", "WCCW-FM", "in Traverse City, Michigan"), ("before_action", "Traverse City, Michigan", "population was 14,674 at the 2010 census")]),
 (49, "Who wrote the play in which Evelyn Varden", "retain_candidate", ["query_copies_visible_role"],
  "Varden's biography identifies Our Town as the play in which she played Mrs. Gibbs. The copied role sentence retrieves the play article identifying Thornton Wilder as playwright.", None,
  [("before_action", "Evelyn Varden", "Mrs. Gibbs"), ("before_action", "Evelyn Varden", "original production of \"Our Town\""), ("retrieved_after_search", "Our Town", "play by American playwright Thornton Wilder")]),
 (50, "Who wrote the play in which Evelyn Varden", "retain_candidate", ["answer_has_terminal_period"],
  "Varden's Mrs. Gibbs role is in Our Town; its article attributes the play to Thornton Wilder. The terminal period in the action answer is harmless normalization, not a factual defect.", None,
  [("before_action", "Evelyn Varden", "original production of \"Our Town\""), ("before_action", "Our Town", "play by American playwright Thornton Wilder")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-050.jsonl", cohort="v122_internal_candidates")
