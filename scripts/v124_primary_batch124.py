"""Primary examination of v123 internal candidate states 231–240."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (231, "Which of Byron Balasco's series", "retain_candidate", ["query_uses_visible_series_and_actor", "retrieval_adds_film_role"],
  "Balasco's initial biography names Kingdom and Frank Grillo; retrieval says Grillo appeared in Zero Dark Thirty. The query uses an already visible series-actor relation to test the missing film link.", None,
  [("before_action", "Byron Balasco", "\"Kingdom\" featuring Frank Grillo"), ("retrieved_after_search", "Frank Grillo", "\"Zero Dark Thirty\" (2012)")]),
 (232, "Which of Byron Balasco's series", "retain_candidate", [],
  "The initial Balasco and Kingdom passages identify the series and actor Frank Grillo, whose retrieved biography lists Zero Dark Thirty. Kingdom is supported as the requested series.", None,
  [("before_action", "Byron Balasco", "\"Kingdom\" featuring Frank Grillo"), ("before_action", "Frank Grillo", "\"Zero Dark Thirty\" (2012)")]),
 (233, "Which music labels are owned by the group", "retain_candidate", ["query_uses_visible_group", "retrieval_adds_labels"],
  "Acoustic II is identified as an Above & Beyond album in the initial passage; retrieval explicitly names Anjunabeats and Anjunadeep as labels owned by the group.", None,
  [("before_action", "Acoustic II", "group Above & Beyond"), ("retrieved_after_search", "Above &amp; Beyond (band)", "owners of London-based electronic dance music labels Anjunabeats and Anjunadeep")]),
 (234, "Which music labels are owned by the group", "retain_candidate", ["answer_trailing_comma"],
  "The group-to-two-label ownership relation is explicit in the retrieved biography. The frozen answer has a trailing comma, a formatting defect to clean on export rather than a different label.",
  "Remove the trailing comma in a reviewed export.",
  [("before_action", "Acoustic II", "group Above & Beyond"), ("before_action", "Above &amp; Beyond (band)", "labels Anjunabeats and Anjunadeep")]),
 (235, " Sainte-Beuve deemed Athalie", "repair_candidate", ["query_unrelated_kaurab_sentence", "retrieval_adds_oedipus_author"],
  "The initial Athalie article already names Oedipus Rex as Sainte-Beuve's comparison. The query appends an unrelated sentence from Kaurab; retrieval nevertheless finds Oedipus Rex's author Sophocles. The result does not make the unrelated query a valid search target.",
  "Query Oedipus Rex playwright from the visible comparison and replay.",
  [("before_action", "Athalie", "comparable to \"Oedipus Rex\""), ("before_action", "Kaurab", "what keeps the wheel churning at Kaurab today"), ("retrieved_after_search", "Oedipus Rex", "Athenian tragedy by Sophocles")]),
 (236, " Sainte-Beuve deemed Athalie", "repair_candidate", ["paired_unrelated_query", "answer_supported"],
  "Sophocles wrote Oedipus Rex and Athalie is compared to that work in the initial article. The answer is supported, but the paired search contains an unrelated magazine sentence.",
  "Review after an Oedipus-author query replay.",
  [("before_action", "Athalie", "comparable to \"Oedipus Rex\""), ("before_action", "Oedipus Rex", "Athenian tragedy by Sophocles")]),
 (237, "What are two songs sharing title", "quarantine_pending_repair", ["false_same_title_premise", "song_line_confused_with_title"],
  "The Billy Ocean song is titled Get Outta My Dreams, Get into My Car; it is based on a line in the Sherman Brothers' differently titled You're Sixteen. The question falsely says two songs share a title, and retrieval confirms the latter song's title rather than any shared title.",
  "Ask which Sherman Brothers song supplied a line for Billy Ocean's song, then rebuild and review.",
  [("before_action", "Get Outta My Dreams, Get into My Car", "based on a line in the Sherman Brothers' song"), ("retrieved_after_search", "You're Sixteen", "song written by the Sherman Brothers")]),
 (238, "What are two songs sharing title", "quarantine_pending_repair", ["paired_false_same_title_premise"],
  "You're Sixteen is the Sherman Brothers song, not a title shared with Billy Ocean's Get Outta My Dreams, Get into My Car. The frozen answer only makes sense for a different, source-line question.",
  "Use a newly reviewed source-line question; do not retain this answer under the original premise.",
  [("before_action", "Get Outta My Dreams, Get into My Car", "1988 single by Trinidadian-born British singer Billy Ocean"), ("before_action", "Get Outta My Dreams, Get into My Car", "Sherman Brothers' song"), ("before_action", "You're Sixteen", "song written by the Sherman Brothers")]),
 (239, "Which star of The Hurricane Heist", "retain_candidate", ["query_uses_visible_cast", "retrieval_adds_2015_film"],
  "The initial Hurricane Heist passage lists Toby Kebbell in its cast. Retrieval of his biography identifies his appearance in Fantastic Four (2015). The search follows the visible cast list to the missing role.", None,
  [("before_action", "The Hurricane Heist", "stars Toby Kebbell"), ("retrieved_after_search", "Toby Kebbell", "\"Fantastic Four\" (2015)")]),
 (240, "Which star of The Hurricane Heist", "retain_candidate", [],
  "The two film links to Toby Kebbell are present by answer time. The answer identifies a cast member common to The Hurricane Heist and 2015 Fantastic Four.", None,
  [("before_action", "The Hurricane Heist", "stars Toby Kebbell"), ("before_action", "Toby Kebbell", "\"Fantastic Four\" (2015)")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-124.jsonl", cohort="v123_internal_candidates")
