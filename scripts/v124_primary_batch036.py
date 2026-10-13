"""Primary examination of exposed historical internal states 1–10."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (1, "Which band has more members", "retain_candidate", ["query_emphasizes_known_duo", "exposed_development_set"],
  "The visible 3OH!3 article explicitly names a two-person duo. The question-grounded query retrieves Civil Twilight as a four-piece band, filling the missing count. This exposed development item is supported but is not a fresh holdout.", None,
  [("before_action", "3OH!3", "made up of Sean Foreman and Nathaniel Motte"), ("retrieved_after_search", "Civil Twilight (band)", "a four-piece rock band")]),
 (2, "Which band has more members", "retain_candidate", ["exposed_development_set"],
  "3OH!3 is a duo while Civil Twilight is explicitly four-piece; the answer Civil Twilight is supported by both visible band documents. This old internal item has been exposed to development.", None,
  [("before_action", "3OH!3", "made up of Sean Foreman and Nathaniel Motte"), ("before_action", "Civil Twilight (band)", "a four-piece rock band")]),
 (3, "When was the album including", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "The Marvins Room article names Take Care as its album, but the album release date is missing. Searching the visible song sentence retrieves Take Care's November 15, 2011 release date.", None,
  [("before_action", "Marvins Room", "second studio album \"Take Care\""), ("retrieved_after_search", "Take Care (album)", "released on November 15, 2011")]),
 (4, "When was the album including", "retain_candidate", ["exposed_development_set"],
  "The song article places Marvins Room on Take Care, and the album article gives November 15, 2011. The answer date is supported; the old internal question remains exposed.", None,
  [("before_action", "Marvins Room", "second studio album \"Take Care\""), ("before_action", "Take Care (album)", "released on November 15, 2011")]),
 (5, "The composer of the ballet", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "The ballet article identifies Arthur Sullivan as composer. Its copied sentence retrieves Sullivan's biography, which explicitly counts 13 major orchestral works.", None,
  [("before_action", "L'Île Enchantée", "an 1864 ballet by Arthur Sullivan"), ("retrieved_after_search", "Arthur Sullivan", "13 major orchestral works")]),
 (6, "The composer of the ballet", "retain_candidate", ["exposed_development_set"],
  "The ballet article attributes L'Île Enchantée to Sullivan, whose biography states 13 major orchestral works. The number is directly supported, distinct from the unrelated other composer's 80-plus total compositions.", None,
  [("before_action", "L'Île Enchantée", "an 1864 ballet by Arthur Sullivan"), ("before_action", "Arthur Sullivan", "13 major orchestral works")]),
 (7, "From the Carpet has a cover", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "The EP article names Working Class Hero as its Lennon cover. Searching the visible EP sentence retrieves the song article naming the 1970 album John Lennon/Plastic Ono Band; the 1995 tribute album is a distractor.", None,
  [("before_action", "From the Carpet", "cover of John Lennon's \"Working Class Hero\""), ("retrieved_after_search", "Working Class Hero", "from his 1970 album \"John Lennon/Plastic Ono Band\"")]),
 (8, "From the Carpet has a cover", "retain_candidate", ["exposed_development_set"],
  "From the Carpet covers Working Class Hero; its song article states that the song is from the 1970 John Lennon/Plastic Ono Band album. The answer does not conflate it with the later tribute album.", None,
  [("before_action", "From the Carpet", "cover of John Lennon's \"Working Class Hero\""), ("before_action", "Working Class Hero", "from his 1970 album \"John Lennon/Plastic Ono Band\"")]),
 (9, "Where was the man who Mick Kelly", "retain_candidate", ["query_known_person_verifies_relation", "exposed_development_set"],
  "Al Jean's Detroit birthplace and upbringing are visible before search, but the Mick Kelly production-assistant relation is not. Querying Jean retrieves Little Orphan Millie, which explicitly identifies Kelly as his production assistant.", None,
  [("before_action", "Al Jean", "born and raised in Detroit, Michigan"), ("retrieved_after_search", "Little Orphan Millie", "written by Mick Kelly, production assistant to Al Jean")]),
 (10, "Where was the man who Mick Kelly", "retain_candidate", ["exposed_development_set"],
  "The episode article identifies Al Jean as the person for whom Mick Kelly was production assistant; Jean's biography says he was born and raised in Detroit, Michigan. The answer is supported.", None,
  [("before_action", "Little Orphan Millie", "written by Mick Kelly, production assistant to Al Jean"), ("before_action", "Al Jean", "born and raised in Detroit, Michigan")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-036.jsonl", cohort="v117_exposed_internal50")
