"""Primary examination of v123 internal candidate states 31–40."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (31, "In what year was Mark Michael Brzezicki's most well known band", "repair_candidate", ["query_selects_unrelated_pezband", "most_well_known_overstates_source"],
  "The initial input has Big Country's 1981 formation, but no initial article linking Brzezicki to it. The query adds unrelated Pezband; retrieval only then identifies Brzezicki as primarily known for Big Country. The question also upgrades 'primarily known for his work with' into a ranking of his 'most well known band'.",
  "Query Brzezicki's band affiliation from his name before selecting a formation year, then word the question as 'band he is primarily known for working with' and replay.",
  [("before_action", "Pezband", "formed in 1971"), ("before_action", "Big Country", "formed in Dunfermline, Fife, in 1981"), ("retrieved_after_search", "Mark Brzezicki", "primarily known for his work with Big Country")]),
 (32, "In what year was Mark Michael Brzezicki's most well known band", "repair_candidate", ["paired_unrelated_query", "answer_year_previsible", "band_ranking_wording"],
  "1981 is supported as Big Country's formation year, and the retrieved Brzezicki biography links him to it. But the paired query pursued Pezband, and the original 'most well known band' wording is stronger than the biography's 'primarily known for his work with' statement.",
  "Reword the relation and replay with a grounded person-to-band search.",
  [("before_action", "Big Country", "in 1981"), ("before_action", "Mark Brzezicki", "primarily known for his work with Big Country")]),
 (33, '"Grey" is the fourth studio album', "retain_candidate", ["query_uses_visible_album_singer", "retrieval_adds_birth_year"],
  "The initial Grey album article names Sandy Lam as the singer. Searching that visible album relation retrieves her biography, which states 1966 as her birth year. Nearby titles containing Grey are distinct works.", None,
  [("before_action", "Grey (album)", '"Grey" is the fourth studio album by Sandy Lam'), ("retrieved_after_search", "Sandy Lam", "born 26 April 1966")]),
 (34, '"Grey" is the fourth studio album', "retain_candidate", [],
  "The Grey album is explicitly by Sandy Lam, and her retrieved biography gives 1966. The answer is supported by the selected work rather than other Grey-titled albums.", None,
  [("before_action", "Grey (album)", "fourth studio album by Sandy Lam"), ("before_action", "Sandy Lam", "born 26 April 1966")]),
 (35, "What was the debut film of the female star", "retain_candidate", ["query_uses_visible_female_cast_member", "retrieval_adds_debut_film"],
  "The initial Off the Rack article names Eileen Brennan as its female star. The cast-list query retrieves Brennan's biography, which identifies her film debut as Divorce American Style. Other returned Eileen names are distractors.", None,
  [("before_action", "Off the Rack", "stars Ed Asner and Eileen Brennan"), ("retrieved_after_search", "Eileen Brennan", 'film debut in the satire "Divorce American Style"')]),
 (36, "What was the debut film of the female star", "retain_candidate", [],
  "Eileen Brennan is the named female lead in Off the Rack, and her biography directly names Divorce American Style as her film debut. Quotation marks in the answer do not alter the title.", None,
  [("before_action", "Off the Rack", "Eileen Brennan"), ("before_action", "Eileen Brennan", 'film debut in the satire "Divorce American Style"')]),
 (37, 'The Rihanna song "American Oxygen"', "repair_candidate", ["answer_service_preselected_before_song_link", "query_contains_unjustified_tidal"],
  "The initial input contains only a generic Tidal service description; it does not link Rihanna's American Oxygen to Tidal. The query appends Tidal as a selected service, and retrieval later finds the song article confirming it. The later confirmation cannot retroactively ground the search choice.",
  "Query American Oxygen's first streaming service without supplying a candidate service, then replay retrieval and answer.",
  [("before_action", "Tidal (service)", "subscription-based music streaming service"), ("retrieved_after_search", "American Oxygen", "made available for streaming on Tidal on April 5, 2015")]),
 (38, 'The Rihanna song "American Oxygen"', "repair_candidate", ["paired_preselected_service_query", "answer_supported_after_search"],
  "Tidal is explicitly named in the retrieved song article, but its selection in the paired query preceded any action-time song-to-service evidence. The answer's support does not validate that action.",
  "Keep Tidal only after a service-neutral song query replay and independent audit.",
  [("before_action", "American Oxygen", "made available for streaming on Tidal on April 5, 2015")]),
 (39, "What media formally covered issues from the Civil Rights", "repair_candidate", ["question_formally_wording", "shared_publisher_historical_snapshot"],
  "The initial Johnson Publishing article ties The New Ebony Cookbook and Jet to the same company. Searching the company retrieves Jet's Civil Rights Movement coverage. Yet the question's 'formally covered issues from the Civil Rights' is malformed, and the article also says the publications were sold in 2016, so a current shared-operator reading is unsafe.",
  "Ask which Johnson-published magazine chronicled the Civil Rights Movement in the historical publishing period; replay.",
  [("before_action", "Johnson Publishing Company", 'books such as "The New Ebony Cookbook"'), ("before_action", "Johnson Publishing Company", 'also published "Jet" magazine'), ("retrieved_after_search", "Jet (magazine)", "chronicling the Civil Rights Movement")]),
 (40, "What media formally covered issues from the Civil Rights", "repair_candidate", ["paired_question_wording", "publisher_change_needs_time_boundary"],
  "Jet is the supported magazine that chronicled the Civil Rights Movement and shared historical Johnson Publishing ownership with the cookbook. The original question's wording and post-2016 publisher change prevent a clean unqualified operator claim.",
  "Retain Jet under a historical publisher relation with clear wording after replay.",
  [("before_action", "Johnson Publishing Company", 'books such as "The New Ebony Cookbook"'), ("before_action", "Johnson Publishing Company", 'also published "Jet" magazine'), ("before_action", "Jet (magazine)", "chronicling the Civil Rights Movement")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-104.jsonl", cohort="v123_internal_candidates")
