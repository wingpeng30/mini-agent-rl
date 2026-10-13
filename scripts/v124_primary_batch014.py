"""Primary case examination of original training rows 131–140."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (131, "In which area of Northern Ireland", "retain_candidate", ["paired_search_row_109_needs_query_repair"],
  "The reserve article explicitly locates Murlough in County Down, and the mountain article independently locates Slieve Donard there. The answer is supported, although its same-question search row 109 uses a poor query and must be repaired before retaining a paired training example.", None,
  [("before_action", "Murlough Nature Reserve", "lies on the coast of County Down in Northern Ireland"), ("before_action", "Slieve Donard", "an 850 m mountain in County Down, Northern Ireland")]),
 (132, "Which American musician", "retain_candidate", ["query_verifies_cover_relation"],
  "Randolph's biography already establishes the Nashville-sound clue but not the song cover. Querying his visible name retrieves the song article, which explicitly lists Boots Randolph among cover artists. The search verifies the missing relation.", None,
  [("before_action", "Boots Randolph", "a major part of the \"Nashville sound\""), ("retrieved_after_search", "I'm Gonna Be a Wheel Someday", "Boots Randolph, Defenders, Asleep at the Wheel")]),
 (133, "WHich actor", "retain_candidate", ["question_typo"],
  "The film article lists Ian Holm in The Sweet Hereafter, and his biography says he won the Tony for playing Lenny in The Homecoming. The answer is directly supported.", None,
  [("before_action", "The Sweet Hereafter (film)", "starring Ian Holm, Sarah Polley and Bruce Greenwood"), ("before_action", "Ian Holm", "received the 1967 Tony Award for Best Featured Actor for his performance as Lenny in \"The Homecoming\"")]),
 (134, "Which golden globe", "retain_candidate", ["question_award_unspecified"],
  "The film article lists Jennifer Aniston among its stars. Her biography ties her Friends role to a Golden Globe Award. The question asks for a qualifying actor rather than a specific award category, and this answer satisfies it.", None,
  [("before_action", "'Til There Was You", "Jeanne Tripplehorn, Dylan McDermott, Sarah Jessica Parker and Jennifer Aniston"), ("before_action", "Jennifer Aniston", "a Golden Globe Award")]),
 (135, "What star of", "retain_candidate", [],
  "The Down to Earth article lists Marc Platt as a star, and his biography identifies his role in the Seven Brides for Seven Brothers film. The distinct stage musical article is not needed to support the film link.", None,
  [("before_action", "Down to Earth (1947 film)", "starring Rita Hayworth, Larry Parks, and Marc Platt"), ("before_action", "Marc Platt (dancer)", "Daniel Pontipee, one of the seven brothers in the film \"Seven Brides for Seven Brothers\"")]),
 (136, "In which city did the police chief", "repair_candidate", ["question_missing_verb", "city_role_ambiguous"],
  "The pre-action Lively article names assassinated Police Chief Sid Hatfield. The copied Lively sentence retrieves Hatfield's chief-of-Matewan relation, so the query fills a relevant gap; however, the incomplete question does not specify whether it asks where he served or where he was assassinated. This search cannot be retained under the ambiguous input.",
  "Proposed question: Of which city was Sid Hatfield police chief before Charles Everett Lively assassinated him? Revise both same-question states and independently review the changed input.",
  [("before_action", "Charles Everett Lively", "one of the assassins of Police Chief Sid Hatfield"), ("retrieved_after_search", "Sid Hatfield", "Police Chief of Matewan, West Virginia")]),
 (137, "Nilay Konar", "retain_candidate", ["query_verifies_missing_person_birthplace"],
  "The initial Ankara article identifies Turkey's capital but does not link Nilay Konar to it. Searching that visible candidate location retrieves Konar's biography, which explicitly states her birth in Ankara. The search verifies the missing person-place relation.", None,
  [("before_action", "Ankara", "the capital of the Republic of Turkey"), ("retrieved_after_search", "Nilay Konar", "born August 30, 1980 in Ankara")]),
 (138, "Of what origin", "retain_candidate", [],
  "The documentary article identifies Rick Rescorla as Morgan Stanley's security director, and his biography explicitly describes him as of British origin. The answer is supported by the two documents.", None,
  [("before_action", "The Man Who Predicted 9/11", "director of security of Morgan Stanley, Rick Rescorla"), ("before_action", "Rick Rescorla", "of British origin")]),
 (139, "In which city did the police chief", "repair_candidate", ["question_missing_verb", "city_role_ambiguous"],
  "The stored Matewan answer follows Hatfield's office location. The question omits the verb and can be read as asking for the assassination location, which is not established by these documents. Treat the intended office-city answer as needing a scoped question repair, jointly with search row 136.",
  "Proposed question: Of which city was Sid Hatfield police chief before Charles Everett Lively assassinated him? Retain Matewan, West Virginia only after both paired states, token replay and independent review.",
  [("before_action", "Charles Everett Lively", "one of the assassins of Police Chief Sid Hatfield"), ("before_action", "Sid Hatfield", "Police Chief of Matewan, West Virginia")]),
 (140, "How many homicides", "retain_candidate", [],
  "Bundy's biography says he confessed to 30 homicides before execution, and Keppel's biography names Bundy among the killers he investigated. The answer is a confession count; the source explicitly says the true victim total is unknown.", None,
  [("before_action", "Ted Bundy", "he confessed to 30 homicides"), ("before_action", "Robert D. Keppel", "work investigating serial killers Ted Bundy and Gary Ridgway")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-014.jsonl")
