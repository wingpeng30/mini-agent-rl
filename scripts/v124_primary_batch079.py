"""Primary examination of v122 training-supplement candidate states 1–10."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (1, "The book in which the film \"All Quiet on the Western Front\"", "retain_candidate", ["query_uses_visible_author_relation"],
  "Both initially visible film articles identify Erich Maria Remarque as the author of the adapted novel. The query uses the 1979 film's explicit book-author sentence, and retrieval finds his 1898 birth date. The film version is unspecified, but both visible adaptations share the same book author for this question.", None,
  [("before_action", "All Quiet on the Western Front (1979 film)", "book of the same name by Erich Maria Remarque"), ("before_action", "All Quiet on the Western Front (1930 film)", "Erich Maria Remarque novel"), ("retrieved_after_search", "Erich Maria Remarque", "22 June 1898")]),
 (2, "The book in which the film \"All Quiet on the Western Front\"", "retain_candidate", [],
  "The visible adaptation articles identify Remarque and his biography gives 1898. Both available film versions have the same author, so the year is supported despite the unspecific film title.", None,
  [("before_action", "All Quiet on the Western Front (1979 film)", "by Erich Maria Remarque"), ("before_action", "Erich Maria Remarque", "22 June 1898")]),
 (3, "The Man stars an actor who has appeared", "retain_candidate", ["query_uses_visible_cast"],
  "The 2005 film lists Eugene Levy among its stars. The copied cast sentence retrieves Levy's biography, which says he appeared in all eight American Pie films; the query has an action-time anchor.", None,
  [("before_action", "The Man (2005 film)", "starring Eugene Levy"), ("retrieved_after_search", "Eugene Levy", "all eight of the \"American Pie\" films")]),
 (4, "The Man stars an actor who has appeared", "retain_candidate", [],
  "The Man stars Eugene Levy, and his biography explicitly names all eight American Pie films. The target series is supported; quotation marks in the stored answer are formatting only.", None,
  [("before_action", "The Man (2005 film)", "starring Eugene Levy"), ("before_action", "Eugene Levy", "all eight of the \"American Pie\" films")]),
 (5, "who has the last name Ferreira", "repair_candidate", ["question_typo_oals", "query_copies_non_target_coach_sentence"],
  "The question misspells goals and omits a clear league/season construction. The initial Barcelona season article names Rivaldo and says he scored 20 league goals; the query instead copies a coach-and-transfer sentence. Retrieval adds Rivaldo's full Ferreira name, but the query is not aimed at that missing surname relation.",
  "Correct the season question and query Rivaldo's full name or Ferreira identity, then replay both states.",
  [("before_action", "1997–98 FC Barcelona season", "Rivaldo was a perfect centre forward, scoring 20 goals in the league"), ("retrieved_after_search", "Rivaldo", "Rivaldo Vítor Borba Ferreira")]),
 (6, "who has the last name Ferreira", "repair_candidate", ["paired_question_typo", "paired_search_off_target"],
  "The season article ties Rivaldo to 20 league goals, and his biography supplies Ferreira in his full name. The answer is supported after search, but the malformed question and indirect preceding query need a separate replayed record.",
  "Keep Rivaldo only in a corrected, replayed question/search pair.",
  [("before_action", "1997–98 FC Barcelona season", "Rivaldo was a perfect centre forward, scoring 20 goals in the league"), ("before_action", "Rivaldo", "Rivaldo Vítor Borba Ferreira")]),
 (7, "In which city was the band whose members wrote", "retain_candidate", ["query_uses_visible_song_band_link"],
  "The initial Only the Young article credits Journey members with the song. Searching that citation retrieves Journey's biography, which places formation in San Francisco. The query targets the band linked at action time.", None,
  [("before_action", "Only the Young (Journey song)", "Jonathan Cain, Steve Perry and Neal Schon of the band Journey"), ("retrieved_after_search", "Journey (band)", "formed in San Francisco in 1973")]),
 (8, "In which city was the band whose members wrote", "retain_candidate", [],
  "The song source identifies Journey as the band of its writers, and Journey's article says it formed in San Francisco. The answer is supported by the visible chain.", None,
  [("before_action", "Only the Young (Journey song)", "of the band Journey"), ("before_action", "Journey (band)", "formed in San Francisco in 1973")]),
 (9, "In which state was the band who released The Summer EP", "retain_candidate", ["query_uses_visible_ep_artist_link", "near_title_distractor"],
  "The exact-titled The Summer EP source identifies Never Shout Never; a different Summer EP by Sherwood is also visible, so the title article matters. Searching the Never Shout Never link retrieves its formation in Joplin, Missouri.", None,
  [("before_action", "The Summer (EP)", "The Summer EP is an EP by Never Shout Never"), ("retrieved_after_search", "Never Shout Never", "formed in Joplin, Missouri")]),
 (10, "In which state was the band who released The Summer EP", "retain_candidate", ["near_title_distractor"],
  "The Summer EP belongs to Never Shout Never, and that band formed in Joplin, Missouri. The other Summer EP in the input has a distinct title and band, not the same target.", None,
  [("before_action", "The Summer (EP)", "The Summer EP is an EP by Never Shout Never"), ("before_action", "Never Shout Never", "formed in Joplin, Missouri")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-079.jsonl", cohort="v122_train_candidates")
