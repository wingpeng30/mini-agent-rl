"""Primary examination of v122 training-supplement candidate states 71–80."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (71, "In what year was the Nobel Prize in Physiology or Medicine", "retain_candidate", ["query_uses_visible_supervisor_link"],
  "Roger Hendrix's initial biography names James Watson as his PhD supervisor. Searching that biographical anchor retrieves Watson's article and its explicit 1962 Physiology or Medicine Nobel year; Yalow's unrelated 1977 prize is a distractor.", None,
  [("before_action", "Roger Hendrix (biologist)", "under the supervision of James Watson"), ("retrieved_after_search", "James Watson", "awarded the 1962 Nobel Prize in Physiology or Medicine")]),
 (72, "In what year was the Nobel Prize in Physiology or Medicine", "retain_candidate", [],
  "Hendrix's supervisor is James Watson, and Watson's source states the 1962 Nobel Prize in Physiology or Medicine. The answer is supported without confusing another visible laureate.", None,
  [("before_action", "Roger Hendrix (biologist)", "supervision of James Watson"), ("before_action", "James Watson", "1962 Nobel Prize in Physiology or Medicine")]),
 (73, "In what year was the man who commanded the Nachtjagdgeschwader 4", "repair_candidate", ["query_copies_unit_description_not_named_commander", "retrieval_accidental"],
  "The initial NJG 4 article already identifies Heinz-Wolfgang Schnaufer as commander for the exact dates. The query copies a generic night-fighter-wing opening sentence, omitting Schnaufer and the birth-year target. Retrieval happens to return his biography with 1922.",
  "Query Heinz-Wolfgang Schnaufer's birth year using the visible commander line, then replay.",
  [("before_action", "Nachtjagdgeschwader 4", "Heinz-Wolfgang Schnaufer (20 November 1944 – 8 May 1945)"), ("retrieved_after_search", "Heinz-Wolfgang Schnaufer", "16 February 1922")]),
 (74, "In what year was the man who commanded the Nachtjagdgeschwader 4", "repair_candidate", ["paired_generic_unit_query"],
  "The command dates link Schnaufer to the unit and his biography gives 1922. The final answer is supported, but the preceding generic query does not target the missing birth year or commander.",
  "Keep 1922 only with a targeted replayed search pair.",
  [("before_action", "Nachtjagdgeschwader 4", "Heinz-Wolfgang Schnaufer (20 November 1944 – 8 May 1945)"), ("before_action", "Heinz-Wolfgang Schnaufer", "16 February 1922")]),
 (75, "Which actress in the movie Miss Firecracker", "retain_candidate", ["query_uses_visible_actress_film_list"],
  "Amy Wright's initial biography lists both Miss Firecracker and Heartland. The query names her; retrieval finds Heartland's article identifying Letters of a Woman Homesteader as the source memoir. The support runs through Wright's film list, not the abbreviated main cast lists.", None,
  [("before_action", "Amy Wright", "\"Heartland\", \"Wise Blood\""), ("before_action", "Amy Wright", "\"Miss Firecracker\""), ("retrieved_after_search", "Heartland (film)", "based on a memoir by Elinore Pruitt Stewart, titled \"Letters of a Woman Homesteader\"")]),
 (76, "Which actress in the movie Miss Firecracker", "retain_candidate", [],
  "Amy Wright's biography names appearances in both Miss Firecracker and Heartland; the Heartland article gives the memoir adaptation. The answer is supported without assuming that the separate film blurbs list every cast member.", None,
  [("before_action", "Amy Wright", "\"Heartland\", \"Wise Blood\""), ("before_action", "Amy Wright", "\"Miss Firecracker\""), ("before_action", "Heartland (film)", "\"Letters of a Woman Homesteader\"")]),
 (77, "Skin Deep included the actress who played which character on Roswell", "retain_candidate", ["answer_visible_before_search", "search_checks_episode_link"],
  "Emilie de Ravin's initial biography already gives Tess Harding as her Roswell role. Searching her name retrieves the Skin Deep episode article, which establishes that she appears there as Belle. The search verifies the episode qualifier rather than discovers Tess.", None,
  [("before_action", "Emilie de Ravin", "played Tess Harding on \"Roswell\""), ("retrieved_after_search", "Skin Deep (Once Upon a Time)", "Belle (Emilie de Ravin)")]),
 (78, "Skin Deep included the actress who played which character on Roswell", "retain_candidate", ["answer_visible_before_paired_search"],
  "The Skin Deep episode names Emilie de Ravin, and her biography identifies Tess Harding as her Roswell character. The answer is supported at the answer state.", None,
  [("before_action", "Skin Deep (Once Upon a Time)", "Belle (Emilie de Ravin)"), ("before_action", "Emilie de Ravin", "played Tess Harding on \"Roswell\"")]),
 (79, "Whose sixth album closes with the song \"Almost Independence Day\"", "retain_candidate", ["answer_visible_before_search", "search_checks_album_order"],
  "The song article already names Van Morrison and Saint Dominic's Preview. Searching that sentence retrieves the album article confirming it is Morrison's sixth studio album, the qualifying relation absent from initial evidence.", None,
  [("before_action", "Almost Independence Day", "closing song on Northern Irish singer-songwriter Van Morrison's 1972 album"), ("retrieved_after_search", "Saint Dominic's Preview", "sixth studio album by Northern Irish singer-songwriter Van Morrison")]),
 (80, "Whose sixth album closes with the song \"Almost Independence Day\"", "retain_candidate", [],
  "Almost Independence Day closes Saint Dominic's Preview; that album is explicitly Van Morrison's sixth studio album. The answer is supported by the two source articles.", None,
  [("before_action", "Almost Independence Day", "closing song on Northern Irish singer-songwriter Van Morrison's 1972 album"), ("before_action", "Saint Dominic's Preview", "sixth studio album by Northern Irish singer-songwriter Van Morrison")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-086.jsonl", cohort="v122_train_candidates")
