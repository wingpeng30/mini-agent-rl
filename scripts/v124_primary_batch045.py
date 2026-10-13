"""Primary examination of exposed historical internal states 91–100."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (91, "Puhoy is an episode", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "Puhoy's article places it within Adventure Time; copying that series sentence retrieves the season article, which explicitly attributes series creation to Pendleton Ward. Ward's episode story credit alone would not prove series creation.", None,
  [("before_action", "Puhoy", "television series \"Adventure Time\""), ("retrieved_after_search", "Adventure Time (season 5)", "series created by Pendleton Ward")]),
 (92, "Puhoy is an episode", "retain_candidate", ["exposed_development_set"],
  "Puhoy is an Adventure Time episode, and the season article states that the series was created by Pendleton Ward. The answer rests on the series-creation citation, not merely the episode writing credit.", None,
  [("before_action", "Puhoy", "television series \"Adventure Time\""), ("before_action", "Adventure Time (season 5)", "series created by Pendleton Ward")]),
 (93, "How many consecutive championships", "retain_candidate", ["query_targets_visible_trade_counterparty", "exposed_development_set"],
  "The Timberwolves article names Stacey King as the player received for Luc Longley. Searching the visible counterparty team's season article retrieves King's biography with his three consecutive championships.", None,
  [("before_action", "1993–94 Minnesota Timberwolves season", "traded Luc Longley to the Chicago Bulls for Stacey King"), ("retrieved_after_search", "Stacey King", "won three consecutive championships")]),
 (94, "How many consecutive championships", "retain_candidate", ["exposed_development_set"],
  "The trade identifies Stacey King, and his biography explicitly states three consecutive championships with Chicago from 1991 to 1993. The answer is supported.", None,
  [("before_action", "1993–94 Minnesota Timberwolves season", "for Stacey King"), ("before_action", "Stacey King", "won three consecutive championships")]),
 (95, "What nationality was the defending champion", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "The tournament singles article names Martina Hingis as defending champion; its copied sentence retrieves her biography identifying her as Swiss.", None,
  [("before_action", "2007 Sunfeast Open – Singles", "Martina Hingis was the defending champion"), ("retrieved_after_search", "Martina Hingis", "a Swiss professional tennis player")]),
 (96, "What nationality was the defending champion", "retain_candidate", ["exposed_development_set"],
  "The defending champion was Martina Hingis, and her biography calls her Swiss. The answer is supported, despite her nonparticipation in that year's event.", None,
  [("before_action", "2007 Sunfeast Open – Singles", "Martina Hingis was the defending champion"), ("before_action", "Martina Hingis", "a Swiss professional tennis player")]),
 (97, "For what role did the actor", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "The film article identifies Brian Donlevy as Jim Sherwood. Copying that sentence retrieves Donlevy's biography, which names Sergeant Markoff as the role earning his Academy Award nomination.", None,
  [("before_action", "Billy the Kid (1941 film)", "Brian Donlevy as a fictionalized version of Pat Garrett renamed \"Jim Sherwood\""), ("retrieved_after_search", "Brian Donlevy", "role as Sergeant Markoff")]),
 (98, "For what role did the actor", "retain_candidate", ["exposed_development_set"],
  "Brian Donlevy plays Jim Sherwood in Billy the Kid; his biography attributes an Academy Award nomination to Sergeant Markoff in Beau Geste. The answer is supported.", None,
  [("before_action", "Billy the Kid (1941 film)", "Brian Donlevy as a fictionalized version of Pat Garrett renamed \"Jim Sherwood\""), ("before_action", "Brian Donlevy", "role as Sergeant Markoff in \"Beau Geste\"")]),
 (99, "Aaron Zigman scored music", "retain_candidate", ["query_uses_question_and_visible_film_list", "exposed_development_set"],
  "Zigman's article lists The Notebook as one of the films he scored but omits its year and genre. The question-plus-list query retrieves the film article, confirming the 2004 romantic drama description.", None,
  [("before_action", "Aaron Zigman", "including \"The Notebook\""), ("retrieved_after_search", "The Notebook", "2004 American romantic drama film")]),
 (100, "Aaron Zigman scored music", "retain_candidate", ["exposed_development_set"],
  "Zigman's article lists The Notebook among his scores, and The Notebook article identifies it as a 2004 American romantic drama. The answer is supported.", None,
  [("before_action", "Aaron Zigman", "including \"The Notebook\""), ("before_action", "The Notebook", "2004 American romantic drama film")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-045.jsonl", cohort="v117_exposed_internal50")
