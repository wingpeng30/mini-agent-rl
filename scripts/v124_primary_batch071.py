"""Primary examination of v122 internal candidate states 251–260."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (251, "What is the birthplace for the DJ", "retain_candidate", ["query_copies_visible_album_artist"],
  "The album article names Bally Sagoo as the DJ. The search retrieves his biography, which explicitly says he was born in Delhi, India, distinguishing his birthplace from where he was raised.", None,
  [("before_action", "Rising from the East", "DJ Bally Sagoo"), ("retrieved_after_search", "Bally Sagoo", "Born in Delhi, India")]),
 (252, "What is the birthplace for the DJ", "retain_candidate", [],
  "Bally Sagoo is credited on Rising from the East, and his biography states Delhi, India as his birthplace. The answer is supported.", None,
  [("before_action", "Rising from the East", "DJ Bally Sagoo"), ("before_action", "Bally Sagoo", "Born in Delhi, India")]),
 (253, "In what year was the singer who replaced Fred de Wilde", "retain_candidate", ["query_uses_visible_band_album_context"],
  "The initial Shocking Blue album snippet says Mariska Veres replaced Fred de Wilde on At Home. The query copies that source context and retrieval adds Veres's biography and 1947 birth year.", None,
  [("before_action", "Shocking Blue (album)", "Mariska Veres replaced de Wilde in the band's next album"), ("retrieved_after_search", "Mariska Veres", "1 October 1947")]),
 (254, "In what year was the singer who replaced Fred de Wilde", "retain_candidate", [],
  "Mariska Veres is the named replacement singer, and her biography gives 1947 as her birth year. The answer is supported.", None,
  [("before_action", "Shocking Blue (album)", "Mariska Veres replaced de Wilde"), ("before_action", "Mariska Veres", "1 October 1947")]),
 (255, "Albina Mayorova has had", "retain_candidate", ["search_verifies_race_participation"],
  "The initial Honolulu Marathon snippet supplies December and the exotic-location description but not Mayorova. Searching the visible race title retrieves her biography, which confirms a top-four finish there.", None,
  [("before_action", "Honolulu Marathon", "taking place annually on the second Sunday in December"), ("before_action", "Honolulu Marathon", "popular for its exotic location in Hawaii"), ("retrieved_after_search", "Albina Mayorova", "top four finishes at the Chicago, London and Honolulu Marathons")]),
 (256, "Albina Mayorova has had", "retain_candidate", [],
  "Mayorova's biography names a Honolulu top-four finish, and the marathon article supplies the December/exotic-location clues. The answer is supported.", None,
  [("before_action", "Honolulu Marathon", "second Sunday in December"), ("before_action", "Honolulu Marathon", "exotic location in Hawaii"), ("before_action", "Albina Mayorova", "top four finishes at the Chicago, London and Honolulu Marathons")]),
 (257, "When was the original drummer for the Who", "retain_candidate", ["query_copies_visible_drummer_identity"],
  "The Who Tour article names original drummer Keith Moon. The search retrieves Moon's biography, which states his posthumous Modern Drummer Hall of Fame induction in 1982, distinct from distractor drummers' years.", None,
  [("before_action", "The Who Tour 1981", "original drummer Keith Moon"), ("retrieved_after_search", "Keith Moon", "inducted into the \"Modern Drummer\" Hall of Fame in 1982")]),
 (258, "When was the original drummer for the Who", "retain_candidate", [],
  "Keith Moon is named as the Who's original drummer and his biography gives the 1982 induction. The answer is supported.", None,
  [("before_action", "The Who Tour 1981", "original drummer Keith Moon"), ("before_action", "Keith Moon", "Hall of Fame in 1982")]),
 (259, "Binti is a novella written by an author of what minority", "repair_candidate", ["minority_status_not_established_by_nationality", "vague_identity_category"],
  "Binti's article names Nnedi Okorafor, and retrieval calls her a Nigerian American writer. It does not establish which population, jurisdiction or criterion makes this a 'minority'. The target is a nationality/identity description, not evidence for the question's minority classification.",
  "Ask how Binti's author is described in the cited biography, or specifically ask for her Nigerian-American description; replay.",
  [("before_action", "Binti (novel)", "written by Nnedi Okorafor"), ("retrieved_after_search", "Nnedi Okorafor", "is a Nigerian American writer")]),
 (260, "Binti is a novella written by an author of what minority", "repair_candidate", ["paired_unsupported_minority_category"],
  "Nigerian American is stated for Okorafor, but the original question's unspecified 'minority' status is not supported by the source. The label is only appropriate for a reworded identity-description question.",
  "Retain the source phrase only after a narrower question and paired replay.",
  [("before_action", "Binti (novel)", "written by Nnedi Okorafor"), ("before_action", "Nnedi Okorafor", "is a Nigerian American writer")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-071.jsonl", cohort="v122_internal_candidates")
