"""Primary examination of exposed historical internal states 71–80."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (71, "What rock band was formed", "repair_candidate", ["question_attributes_song_chart_rank_to_band", "query_repeats_visible_band_title", "exposed_development_set"],
  "The visible article establishes that Supergrass formed in Oxford. A repeated-title query retrieves Going Out, which reached No. 5; the song, not the band itself, holds that chart statement. The question should explicitly mention the single.", None,
  [("before_action", "Supergrass", "formed in 1993 in Oxford"), ("retrieved_after_search", "Going Out", "second album by Britpop band Supergrass"), ("retrieved_after_search", "Going Out", "reached 5 in the UK Charts")]),
 (72, "What rock band was formed", "repair_candidate", ["question_attributes_song_chart_rank_to_band", "exposed_development_set"],
  "Supergrass is the band formed in Oxford, and its Going Out single reached No. 5. The answer is plausible, but the question incorrectly says the band reached No. 5 without naming the single; repair the pair's wording.", None,
  [("before_action", "Supergrass", "formed in 1993 in Oxford"), ("before_action", "Going Out", "second album by Britpop band Supergrass"), ("before_action", "Going Out", "reached 5 in the UK Charts")]),
 (73, "What was the population of the city", "retain_candidate", ["query_copies_visible_context", "population_is_estimate", "exposed_development_set"],
  "Rathbun's visible article specifies Atlanta as the city where he is active. The copied sentence retrieves Atlanta's article with an explicitly estimated 2016 city population, distinct from the metropolitan total.", None,
  [("before_action", "Kevin Rathbun", "active in Atlanta"), ("retrieved_after_search", "Atlanta", "estimated 2016 population of 472,522")]),
 (74, "What was the population of the city", "retain_candidate", ["population_is_estimate", "exposed_development_set"],
  "Rathbun is active in Atlanta; Atlanta's article reports an estimated 2016 city population of 472,522. The numeric answer must retain its estimate provenance and not be treated as a census count.", None,
  [("before_action", "Kevin Rathbun", "active in Atlanta"), ("before_action", "Atlanta", "estimated 2016 population of 472,522")]),
 (75, "The Specialists is", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "The Specialists article names Half-Life as the base game. Its copied sentence retrieves the original Half-Life article, which explicitly identifies Valve as developer; the Half-Life 2: Deathmatch hit is a distractor.", None,
  [("before_action", "The Specialists", "computer game \"Half-Life\""), ("retrieved_after_search", "Half-Life (video game)", "developed by Valve")]),
 (76, "The Specialists is", "retain_candidate", ["exposed_development_set"],
  "The Specialists is a modification for Half-Life, and the original game's article states it was developed by Valve. The answer is supported by the correct game article.", None,
  [("before_action", "The Specialists", "computer game \"Half-Life\""), ("before_action", "Half-Life (video game)", "developed by Valve")]),
 (77, "What traditional folk song", "retain_candidate", ["query_uses_question_and_visible_singles", "exposed_development_set"],
  "The album article lists Maybellene and Midnight Special as singles without identifying which is a folk song. The query stays within the visible singles and retrieves Midnight Special's genre description.", None,
  [("before_action", "Here We à Go Go Again!", "\"Maybellene\" and \"Midnight Special\""), ("retrieved_after_search", "Midnight Special (song)", "a traditional folk song")]),
 (78, "What traditional folk song", "retain_candidate", ["exposed_development_set"],
  "The album lists Midnight Special as a hit single and its song article calls it a traditional folk song. The answer is supported.", None,
  [("before_action", "Here We à Go Go Again!", "\"Maybellene\" and \"Midnight Special\""), ("before_action", "Midnight Special (song)", "a traditional folk song")]),
 (79, "Who was the director of a 2015", "retain_candidate", ["query_uses_question_and_visible_film_list", "exposed_development_set"],
  "The company article lists Room among its productions but omits the year and director. The question-plus-list query retrieves Room's 2015 article and its director Lenny Abrahamson.", None,
  [("before_action", "No Trace Camping", "and \"Room\""), ("retrieved_after_search", "Room (2015 film)", "directed by Lenny Abrahamson")]),
 (80, "Who was the director of a 2015", "retain_candidate", ["exposed_development_set"],
  "No Trace Camping's article lists Room, and the Room article identifies it as a 2015 independent drama directed by Lenny Abrahamson. The answer is supported.", None,
  [("before_action", "No Trace Camping", "and \"Room\""), ("before_action", "Room (2015 film)", "2015 independent drama film directed by Lenny Abrahamson")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-043.jsonl", cohort="v117_exposed_internal50")
