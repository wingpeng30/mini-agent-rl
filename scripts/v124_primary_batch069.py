"""Primary examination of v122 internal candidate states 231–240."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (231, "Carrie Pilby stars which Irish actor", "retain_candidate", ["query_copies_visible_cast_relation"],
  "The Carrie Pilby film article lists Colin O'Donoghue in its cast. The search retrieves his biography, which explicitly calls him an Irish actor and musician and provides his full name.", None,
  [("before_action", "Carrie Pilby (film)", "Colin O'Donoghue"), ("retrieved_after_search", "Colin O'Donoghue", "is an Irish actor and musician")]),
 (232, "Carrie Pilby stars which Irish actor", "retain_candidate", [],
  "Colin O'Donoghue is in the film cast, and his biography gives the full name Colin Arthur O'Donoghue plus the Irish actor-musician qualification. The answer is supported.", None,
  [("before_action", "Carrie Pilby (film)", "Colin O'Donoghue"), ("before_action", "Colin O'Donoghue", "Colin Arthur O'Donoghue"), ("before_action", "Colin O'Donoghue", "Irish actor and musician")]),
 (233, "What nationality was the female star", "retain_candidate", ["query_copies_visible_cast_relation"],
  "The film article identifies Natasha Richardson as a star. The search retrieves her biography, which calls her an English actress.", None,
  [("before_action", "The Favour, the Watch and the Very Big Fish", "starring Bob Hoskins, Jeff Goldblum and Natasha Richardson"), ("retrieved_after_search", "Natasha Richardson", "was an English stage and screen actress")]),
 (234, "What nationality was the female star", "retain_candidate", [],
  "Natasha Richardson is the female star named in the cast line, and her biography directly identifies her as English. The answer is supported.", None,
  [("before_action", "The Favour, the Watch and the Very Big Fish", "Natasha Richardson"), ("before_action", "Natasha Richardson", "was an English stage and screen actress")]),
 (235, "In what year was the guitarist born", "retain_candidate", ["query_copies_visible_ep_artist"],
  "The Dreaming No. 11 article names guitarist Joe Satriani. The search retrieves his biography and 1956 birth year.", None,
  [("before_action", "Dreaming No. 11", "EP by guitarist Joe Satriani"), ("retrieved_after_search", "Joe Satriani", "born July 15, 1956")]),
 (236, "In what year was the guitarist born", "retain_candidate", [],
  "Joe Satriani is named as the EP artist, and his biography gives a 1956 birth date. The answer is supported.", None,
  [("before_action", "Dreaming No. 11", "EP by guitarist Joe Satriani"), ("before_action", "Joe Satriani", "born July 15, 1956")]),
 (237, "What type of classical music does the song", "repair_candidate", ["raga_category_improperly_transferred_to_song"],
  "The song article says Ovvoru Pookalume was composed in the Sindhu Bhairavi raga. The retrieved raga article says that raga occurs in Hindustani and Carnatic classical music. It does not directly establish that this particular Tamil film song itself belongs to both classical traditions.",
  "Ask in which two classical music traditions the Sindhu Bhairavi raga used in the song occurs, then replay.",
  [("before_action", "Ovvoru Pookalume", "in the raaga Sindhu Bhairavi"), ("retrieved_after_search", "Sindhu Bhairavi (raga)", "raga in Hindustani and Carnatic classical music")]),
 (238, "What type of classical music does the song", "repair_candidate", ["paired_raga_category_transferred_to_song"],
  "Hindustani and Carnatic are the traditions stated for the Sindhu Bhairavi raga; the frozen answer overextends that category to the song itself. The intended relation needs to name the raga explicitly.",
  "Retain the two traditions only with a raga-specific rewritten question and replay.",
  [("before_action", "Ovvoru Pookalume", "in the raaga Sindhu Bhairavi"), ("before_action", "Sindhu Bhairavi (raga)", "raga in Hindustani and Carnatic classical music")]),
 (239, "In what year was the man under whose command", "retain_candidate", ["query_copies_visible_commander_relation"],
  "The flotilla article explicitly says it was founded under Karl Dönitz's command. The search follows that line and retrieves his biography with the 1891 birth year, without confusing him with the person commemorated in the flotilla name.", None,
  [("before_action", "1st U-boat Flotilla", "under the command of \"Fregattenkapitän\" Karl Dönitz"), ("retrieved_after_search", "Karl Dönitz", "16 September 1891")]),
 (240, "In what year was the man under whose command", "retain_candidate", [],
  "The founding commander was Karl Dönitz, not Otto Weddigen, for whom the flotilla was named. Dönitz's biography gives 1891 as birth year. The answer is supported.", None,
  [("before_action", "1st U-boat Flotilla", "under the command of \"Fregattenkapitän\" Karl Dönitz"), ("before_action", "Karl Dönitz", "16 September 1891")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-069.jsonl", cohort="v122_internal_candidates")
