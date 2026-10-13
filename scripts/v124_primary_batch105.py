"""Primary examination of v123 internal candidate states 41–50."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (41, "When was the group founded which recorded", "retain_candidate", ["query_uses_visible_duo", "retrieval_adds_founding_year"],
  "The initial Play Something Country article identifies Brooks & Dunn as the recording duo. The query retains that visible link and retrieves the duo article, which says it was founded in 1990.", None,
  [("before_action", "Play Something Country", "recorded by American country music duo Brooks & Dunn"), ("retrieved_after_search", "Brooks &amp; Dunn", "duo was founded in 1990")]),
 (42, "When was the group founded which recorded", "retain_candidate", [],
  "Play Something Country is credited to Brooks & Dunn, whose retrieved article directly gives 1990 as the founding year. The answer is supported.", None,
  [("before_action", "Play Something Country", "Brooks & Dunn"), ("before_action", "Brooks &amp; Dunn", "founded in 1990")]),
 (43, "Which seminal figure of the Romantic Age", "retain_candidate", ["query_uses_visible_candidate", "retrieval_adds_play_title_source"],
  "The initial William Blake article calls him a seminal figure of the Romantic Age. Searching his visible name retrieves Robert Bolt's The Tiger and the Horse article, which explicitly says its title comes from Blake's work.", None,
  [("before_action", "William Blake", "seminal figure in the history of the poetry and visual arts of the Romantic Age"), ("retrieved_after_search", "The Tiger and the Horse", "takes its title from William Blake's")]),
 (44, "Which seminal figure of the Romantic Age", "retain_candidate", [],
  "William Blake meets the Romantic Age description, and the Bolt play article attributes its title to Blake's The Marriage of Heaven and Hell. The answer is directly supported across the two documents.", None,
  [("before_action", "William Blake", "seminal figure in the history of the poetry and visual arts of the Romantic Age"), ("before_action", "The Tiger and the Horse", "takes its title from William Blake's")]),
 (45, "What is the land area of the state in which A. O. Whitington", "repair_candidate", ["malformed_question_punctuation", "query_uses_visible_state", "retrieval_adds_area"],
  "The initial biography locates A. O. Whitington in South Australia, and retrieval gives the state's land area as 983482 km2. The question's comma between subject and verb and 'in which ... was a businessman' construction are malformed, although the targeted relation is clear.",
  "Ask for South Australia's land area as the state where Whitington worked as businessman and sportsman, then replay.",
  [("before_action", "A. O. Whitington", "businessman and sportsman in South Australia"), ("retrieved_after_search", "South Australia", "total land area of 983482 km2")]),
 (46, "What is the land area of the state in which A. O. Whitington", "repair_candidate", ["paired_question_grammar", "area_supported"],
  "983482 km2 is directly stated for South Australia, and Whitington's biography places him there. The answer is supported, but the paired question needs grammatical repair and a new audit binding.",
  "Retain the exact unit-bearing area only after corrected wording and replay.",
  [("before_action", "A. O. Whitington", "businessman and sportsman in South Australia"), ("before_action", "South Australia", "total land area of 983482 km2")]),
 (47, "Davison's was the major competition", "retain_candidate", ["query_uses_visible_competitor", "retrieval_adds_headquarters"],
  "Davison's initial article names Rich's as its main competition. Searching that visible relation retrieves Rich's article, which directly places its headquarters in Atlanta, Georgia.", None,
  [("before_action", "Davison's", "major competition to Rich's"), ("retrieved_after_search", "Rich's (department store)", "headquartered in Atlanta, Georgia")]),
 (48, "Davison's was the major competition", "retain_candidate", ["answer_includes_state_with_city"],
  "Atlanta is the requested city, and the source gives Atlanta, Georgia as Rich's headquarters. The state suffix in the frozen answer clarifies rather than contradicts the city, subject to downstream exact-match normalization.", None,
  [("before_action", "Davison's", "major competition to Rich's"), ("before_action", "Rich's (department store)", "headquartered in Atlanta, Georgia")]),
 (49, "Robert Bozic fought and lost", "retain_candidate", ["query_uses_visible_opponent", "retrieval_adds_nickname"],
  "The initial Bob Bozic article identifies Larry Holmes as his opponent. A query repeating that visible bout retrieves Holmes's biography, which directly states his Easton Assassin nickname.", None,
  [("before_action", "Bob Bozic", "fought and lost to Larry Holmes"), ("retrieved_after_search", "Larry Holmes", 'boxing nickname of the "Easton Assassin"')]),
 (50, "Robert Bozic fought and lost", "retain_candidate", [],
  "The bout source links Robert Bozic to Larry Holmes, and Holmes's biography names Easton Assassin as his nickname. The answer is supported by the action-time evidence.", None,
  [("before_action", "Bob Bozic", "fought and lost to Larry Holmes"), ("before_action", "Larry Holmes", 'boxing nickname of the "Easton Assassin"')]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-105.jsonl", cohort="v123_internal_candidates")
