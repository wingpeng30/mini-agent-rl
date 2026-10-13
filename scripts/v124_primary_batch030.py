"""Primary case examination of original training rows 291–300."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (291, "The lead female actress in \"City of Angels\"", "repair_candidate", ["duplicate_training_action"],
  "Third Meg Ryan name query after rows 289–290, with the same 1981 known fact and retrieved City of Angels casting relation. It does not add independent supervision.",
  "Preserve source row and deduplicate after pair replay.",
  [("before_action", "Meg Ryan", "began her acting career in 1981"), ("retrieved_after_search", "City of Angels (film)", "starring Nicolas Cage and Meg Ryan")]),
 (292, "The South Wales Borderers served", "retain_candidate", ["query_copies_conflict_list", "duplicate_training_action_group"],
  "The regiment article lists the Second Boer War among conflicts. The copied-list query retrieves its article with the exact October 11, 1899 to May 31, 1902 span, identifying the conflict. Two duplicate queries follow.", None,
  [("before_action", "South Wales Borderers", "the Zulu War, Second Boer War, and World War I and World War II"), ("retrieved_after_search", "Second Boer War", "started on 11 October 1899 and ended on 31 May 1902")]),
 (293, "The South Wales Borderers served", "repair_candidate", ["duplicate_training_action"],
  "Same question, visible conflict list, query and Second Boer War date result as row 292; no distinct target is added.",
  "Retain original and deduplicate after checking the paired answer.",
  [("before_action", "South Wales Borderers", "Second Boer War"), ("retrieved_after_search", "Second Boer War", "started on 11 October 1899 and ended on 31 May 1902")]),
 (294, "The South Wales Borderers served", "repair_candidate", ["duplicate_training_action"],
  "Third copied-list search with the same date evidence; a changed document order does not make a new training decision.",
  "Select one representative after full pair verification.",
  [("before_action", "South Wales Borderers", "Second Boer War"), ("retrieved_after_search", "Second Boer War", "started on 11 October 1899 and ended on 31 May 1902")]),
 (295, "James are an English rock band", "retain_candidate", ["query_copies_visible_context", "duplicate_training_action_group"],
  "James's band article names the hit Laid but not the album year. Searching its hit-single sentence retrieves the song article explicitly calling it the title song from the 1993 album Laid.", None,
  [("before_action", "James (band)", "\"Laid\", which also became a hit on American college radio"), ("retrieved_after_search", "Laid (song)", "title song from 1993 album \"Laid\"")]),
 (296, "James are an English rock band", "retain_candidate", ["direct_album_query_from_question"],
  "From an empty pre-action context, 'album Laid' is grounded in the question and retrieves the Laid song article with the 1993 album year and James attribution. This is a distinct initial-search state.", None,
  [("retrieved_after_search", "Laid (song)", "title song from 1993 album \"Laid\" by Manchester alternative rock band James")]),
 (297, "James are an English rock band", "repair_candidate", ["duplicate_training_action"],
  "Same band-sentence query and 1993 Laid album result as row 295 with different distractors. No new missing relation is supplied.",
  "Keep source row but deduplicate after answer-state review.",
  [("before_action", "James (band)", "\"Laid\", which also became a hit on American college radio"), ("retrieved_after_search", "Laid (song)", "title song from 1993 album \"Laid\"")]),
 (298, "James are an English rock band", "repair_candidate", ["duplicate_training_action"],
  "Third copy of row 295's band-sentence search, retrieving the same 1993 title-song evidence.",
  "Preserve original and select at most one representative for this path.",
  [("before_action", "James (band)", "\"Laid\", which also became a hit on American college radio"), ("retrieved_after_search", "Laid (song)", "title song from 1993 album \"Laid\"")]),
 (299, "\"A Thousand Years\" is a song", "repair_candidate", ["malformed_question", "query_contains_broken_album_field"],
  "The song article names Steve Kazee and the query retrieves his biography with October 30, 1975. However the question ends 'born on which dater?' and the query appends a corrupted empty album field rather than targeting the missing birth date. This accidental successful hit is a poor search training target.",
  "Correct the question to ask Steve Kazee's birth date and test a query grounded in his visible name; replay exact search, answer and tokenizer input before independent review.",
  [("before_action", "A Thousand Years (Christina Perri song)", "vocals from Steve Kazee"), ("retrieved_after_search", "Steve Kazee", "born October 30, 1975")]),
 (300, "Vale of Tears is a novel", "retain_candidate", ["query_copies_partial_bio", "present_tense_source_unneeded"],
  "The novel article identifies its author Peter T. King and party affiliation. Searching a partial copied clause retrieves King's biography stating an April 5, 1944 birth. The present-tense office wording is unnecessary to the requested birth year and should not be imported into the answer.", None,
  [("before_action", "Vale of Tears (novel)", "third novel by Peter T. King"), ("retrieved_after_search", "Peter T. King", "born April 5, 1944")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-030.jsonl")
