"""Primary case examination of original training rows 141–150."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (141, "What is the current home", "repair_candidate", ["unanchored_current_ballpark"],
  "The Baxter article identifies the Cardinals, and the Cardinals article states Busch Stadium has been their home ballpark since 2006. That supports a since-2006 statement but does not anchor the question's 'current' to a source snapshot.",
  "Proposed question: Which ballpark has been home to the St. Louis Cardinals, the team Moose Baxter played for, since 2006? Recheck same-question states and independently review the revised wording.",
  [("before_action", "Moose Baxter", "played for the St. Louis Cardinals"), ("before_action", "St. Louis Cardinals", "Busch Stadium has been their home ballpark since 2006")]),
 (142, "Cestrum and Ericaceae", "retain_candidate", ["different_taxonomic_ranks"],
  "Cestrum is a genus of flowering plants. The query names the missing Ericaceae from the question and retrieves an article calling it a family of flowering plants. The shared category is flowering plants, not genus or family; retain that distinction for the answer state.", None,
  [("before_action", "Cestrum", "species of flowering plants in the family Solanaceae"), ("retrieved_after_search", "Ericaceae", "a family of flowering plants")]),
 (143, "Which person that recorded", "retain_candidate", ["query_verifies_birth_date_for_listed_member"],
  "The initial album article lists Jan Akkerman among the original lineup but has no birth date. Searching the visible album title retrieves Akkerman's biography with the exact requested birth date. The query is broad but fills the missing identifying attribute.", None,
  [("before_action", "Focus Plays Focus", "Thijs van Leer, Jan Akkerman, Martin Dresden and Hans Cleuver"), ("retrieved_after_search", "Jan Akkerman", "born 24 December 1946")]),
 (144, "What profession does Izu", "retain_candidate", [],
  "Ojukwu is already identified as a film director. The question-grounded query retrieves Fred Savage's biography and his television-director role, supporting director as the shared profession.", None,
  [("before_action", "Izu Ojukwu", "is a Nigerian film director"), ("retrieved_after_search", "Fred Savage", "an American actor, television director, and producer")]),
 (145, " Jessie Willcox Smith", "retain_candidate", ["malformed_question_grammar"],
  "Smith's article lists Leslie's Weekly among her clients. The magazine article identifies its former title and explicitly states it was founded in 1855. The malformed connective in the question does not change the intended magazine or date.", None,
  [("before_action", "Jessie Willcox Smith", "\"Leslie's Weekly\""), ("before_action", "Frank Leslie's Illustrated Newspaper", "later renamed Leslie's Weekly"), ("before_action", "Frank Leslie's Illustrated Newspaper", "news magazine founded in 1855")]),
 (146, "When was the  Australian", "retain_candidate", ["question_grammar"],
  "Calabrese's biography identifies Neil Robertson as his compatriot, and Robertson's biography gives the 11 February 1982 birth date. Shawn Budd is an unrelated Australian snooker distractor.", None,
  [("before_action", "Vinnie Calabrese", "together with his compatriot Neil Robertson"), ("before_action", "Neil Robertson (snooker player)", "born 11 February 1982")]),
 (147, "Which English rock band", "retain_candidate", ["one_word_query_broad_but_question_grounded"],
  "No pre-action documents are visible. 'Doom' comes from the question's song title and retrieves both the Hand of Doom article naming Black Sabbath and the Black Science article linking its cover to that song. The short query is broad, but this observed retrieval supplies the required relation.", None,
  [("retrieved_after_search", "Hand of Doom (Black Sabbath song)", "the English heavy metal band Black Sabbath"), ("retrieved_after_search", "Black Science", "intended to represent the Black Sabbath song, \"Hand of Doom\"")]),
 (148, "Adam May", "retain_candidate", ["former_role_vs_current_role"],
  "May's biography explicitly calls WJZ-TV his previous CBS station, and the station article confirms its channel and CBS ownership. The question uses past tense, so the current WBAL job does not conflict.", None,
  [("before_action", "Adam May (television reporter)", "previously an anchor and reporter at Baltimore’s CBS owned and operated station, WJZ-TV"), ("before_action", "WJZ-TV", "virtual and VHF digital channel 13")]),
 (149, "Which comune is located", "quarantine_pending_repair", ["non_unique_comune_in_latium", "generic_query", "question_role_ambiguous"],
  "There is no pre-action evidence. Querying only 'comune' returns at least two different named comuni in Latium, Paganico Sabino and Campagnano di Roma, and a Rome article that itself calls Rome a special comune. The question does not distinguish them or state a unique target. An observed top-three hit cannot determine which answer was intended without an external target rule.",
  "Quarantine the state and its same-question peers. Recover the original official question/answer provenance and a differentiating clue before any revised query or answer; do not choose a top-three document by hindsight.",
  [("retrieved_after_search", "Paganico Sabino", "a \"comune\" (municipality) in the Province of Rieti in the Italian region Latium"), ("retrieved_after_search", "Campagnano di Roma", "a \"comune\" (municipality) in the Metropolitan City of Rome in the Italian region Latium"), ("retrieved_after_search", "Rome", "the capital of Italy and a special comune")]),
 (150, "Who is the the actor", "retain_candidate", ["question_duplicate_word"],
  "The Slingshot article names Balthazar Getty among its stars, and his biography explicitly says he is an actor and member of Ringside. The Jack Griffo result is a same-title song distractor.", None,
  [("before_action", "Slingshot (film)", "It stars David Arquette, Thora Birch, Balthazar Getty and Julianna Margulies"), ("before_action", "Balthazar Getty", "an American actor, member of the band Ringside")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-015.jsonl")
