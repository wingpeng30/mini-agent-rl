"""Primary examination of v122 training-supplement candidate states 51–60."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (51, "Who is older, John Fitzgerald or Andy Murray", "repair_candidate", ["unqualified_john_fitzgerald", "query_adds_unrelated_award_sentence"],
  "Andy Murray's 1987 birth is initially visible. The question gives no occupation for John Fitzgerald, a non-unique name, and the query appends a BBC award sentence irrelevant to birth-date comparison. Retrieval selects the tennis player John Basil Fitzgerald, born 1960, but this selection is not established by the question.",
  "Specify tennis player John Basil Fitzgerald in the question, remove the award text from the query, and replay.",
  [("before_action", "Andy Murray", "born 15 May 1987"), ("before_action", "2015 BBC Sports Personality of the Year Award", "Tennis player Andy Murray was announced as the winner"), ("retrieved_after_search", "John Fitzgerald (tennis)", "born 28 December 1960")]),
 (52, "Who is older, John Fitzgerald or Andy Murray", "repair_candidate", ["paired_unqualified_john_fitzgerald"],
  "John Basil Fitzgerald the tennis player is older than Andy Murray by the two birth dates. The original bare John Fitzgerald name does not uniquely identify that person; the retrieved article cannot retroactively disambiguate the question.",
  "Keep the comparison only in a tennis-qualified, replayed record.",
  [("before_action", "John Fitzgerald (tennis)", "born 28 December 1960"), ("before_action", "Andy Murray", "born 15 May 1987")]),
 (53, "What is the occupation of the \"Harry Potter\" character", "retain_candidate", ["query_uses_visible_actor_character"],
  "Alan Rickman's initial biography names Severus Snape as his Harry Potter role. Searching that line retrieves Snape's character article, which directly describes him as a Hogwarts professor.", None,
  [("before_action", "Alan Rickman", "Severus Snape in the \"Harry Potter\" film series"), ("retrieved_after_search", "Severus Snape", "A Professor at Hogwarts School of Witchcraft and Wizardry")]),
 (54, "What is the occupation of the \"Harry Potter\" character", "retain_candidate", [],
  "The actor-to-character mapping and Snape's professor role are explicit. The long answer is a supported occupation phrase, though Professor is the concise form.", None,
  [("before_action", "Alan Rickman", "Severus Snape in the \"Harry Potter\" film series"), ("before_action", "Severus Snape", "A Professor at Hogwarts School of Witchcraft and Wizardry")]),
 (55, "Who was the Chairman of the Red Cross Aid to Russia Fund", "retain_candidate", ["query_uses_visible_chair_name"],
  "The Aid to Russia Fund article identifies Clementine Churchill as chair. Searching that sentence retrieves her biography, explicitly identifying Sir Winston Churchill as her husband.", None,
  [("before_action", "Aid to Russia Fund", "Mrs. Clementine Churchill was Chairman"), ("retrieved_after_search", "Clementine Churchill", "wife of Sir Winston Churchill")]),
 (56, "Who was the Chairman of the Red Cross Aid to Russia Fund", "retain_candidate", [],
  "Clementine Churchill is the named chairman in the fund source, and her biography states she was Winston Churchill's wife. The answer is supported across the two documents.", None,
  [("before_action", "Aid to Russia Fund", "Mrs. Clementine Churchill was Chairman"), ("before_action", "Clementine Churchill", "wife of Sir Winston Churchill")]),
 (57, "Andrew Strong starred in which film directed by Alan Parker", "retain_candidate", ["title_visible_before_search", "search_checks_director"],
  "Andrew Strong's initial article already gives The Commitments as his starred film. The query quotes a later sentence mentioning the group name and retrieves the film article, which confirms Alan Parker directed it. This verifies the director qualification rather than finding the title.", None,
  [("before_action", "Andrew Strong", "starred as Deco Cuffe in the 1991 cult film \"The Commitments\""), ("retrieved_after_search", "The Commitments (film)", "directed by Alan Parker")]),
 (58, "Andrew Strong starred in which film directed by Alan Parker", "retain_candidate", ["answer_visible_before_paired_search"],
  "Andrew Strong's source names The Commitments and its film article names Alan Parker as director. The title answer is supported, while the paired search supplies the qualifier.", None,
  [("before_action", "Andrew Strong", "film \"The Commitments\""), ("before_action", "The Commitments (film)", "directed by Alan Parker")]),
 (59, "Who owns the 222 acre park", "retain_candidate", ["query_uses_visible_city_anchor"],
  "The initial Beaverton article supplies the seven-mile Portland relation. Querying Beaverton retrieves the 222-acre Tualatin Hills Nature Park article and its named owner. The city query is grounded and the park size disambiguates the result.", None,
  [("before_action", "Beaverton, Oregon", "7 mi west of downtown Portland"), ("retrieved_after_search", "Tualatin Hills Nature Park", "222-acre nature park and wildlife preserve in Beaverton, Oregon, owned by the Tualatin Hills Park and Recreation District")]),
 (60, "Who owns the 222 acre park", "retain_candidate", [],
  "Beaverton matches the city clue, and the park article explicitly identifies a 222-acre park there owned by the Tualatin Hills Park and Recreation District. The answer is directly supported.", None,
  [("before_action", "Beaverton, Oregon", "7 mi west of downtown Portland"), ("before_action", "Tualatin Hills Nature Park", "owned by the Tualatin Hills Park and Recreation District")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-084.jsonl", cohort="v122_train_candidates")
