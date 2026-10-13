"""Primary case examination of original training rows 201–210."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (201, "How long did the dynasty", "repair_candidate", ["duchy_mislabeled_dynasty"],
  "Wulffen's birthplace is the state Saxe-Coburg and Gotha, whose article calls it an Ernestine duchy lasting 1826–1918. A dynasty denotes a ruling family, not the territorial polity whose lifespan is given.",
  "Proposed question: From which year to which year did the duchy of Saxe-Coburg and Gotha, where Gustav Adolf von Wulffen was born, last? Check paired states and review the revised prompt independently.",
  [("before_action", "Gustav Adolf von Wulffen", "born 18 April 1878 in Gotha, Saxe-Coburg and Gotha"), ("before_action", "Saxe-Coburg and Gotha", "was an Ernestine duchy"), ("before_action", "Saxe-Coburg and Gotha", "It lasted from 1826 to 1918")]),
 (202, "Glacier Wind Farm", "retain_candidate", [],
  "The wind-farm article identifies Toole as the other county, and the Toole article explicitly reports 5,324 residents in the 2010 census.", None,
  [("before_action", "Wind power in Montana", "located in Toole and Glacier counties"), ("before_action", "Toole County, Montana", "As of the 2010 census, the population was 5,324")]),
 (203, "How many records did Argentine duo", "repair_candidate", ["duo_membership_inferred_from_judge"],
  "The season-one article lists Joaquín Galán as a judge, while the Pimpinela article identifies Joaquín and Lucía Galán as the duo and states 25 million worldwide. The duo itself is not stated to be a season-one participant; 'duo from season one' blurs the member and group.",
  "Proposed question: How many records worldwide had Pimpinela, the Argentine duo including season-one Tu Cara me Suena judge Joaquín Galán, sold according to the supplied article? Review all same-question states and the revised prompt independently.",
  [("before_action", "Tu Cara me Suena Argentina (season 1)", "Joaquín Galán debuted conforming the panel of judges"), ("before_action", "Pimpinela", "Joaquín and Lucía Galán, better known as Pimpinela"), ("before_action", "Pimpinela", "25 million worldwide")]),
 (204, "How many seats did the Austrian", "retain_candidate", [],
  "The party was originally named Communist Party of German-Austria and the 1945 election article records four seats for the same Communist Party of Austria.", None,
  [("before_action", "Communist Party of Austria", "Established in 1918 as the Communist Party of German-Austria"), ("before_action", "Austrian legislative election, 1945", "The Communist Party of Austria only gained four seats")]),
 (205, "How many tours in the Iraq", "retain_candidate", [],
  "The American Sniper film article attributes the underlying memoir to Chris Kyle and his biography directly reports four Iraq War tours.", None,
  [("before_action", "American Sniper", "the memoir \"American Sniper: The Autobiography of the Most Lethal Sniper in U.S. Military History\" (2012) by Chris Kyle"), ("before_action", "Chris Kyle", "Kyle served four tours in the Iraq War")]),
 (206, "Merrimack Valley High School", "repair_candidate", ["false_school_location", "time_unanchored_enrollment"],
  "The school article places the school in Penacook, Concord, and says it serves Salisbury; the question instead locates it in Salisbury. It also reports approximately 900 students without an as-of date, so present enrollment cannot be asserted.",
  "Proposed source-scoped question: According to the supplied article, approximately how many students attended Merrimack Valley High School in Penacook, which serves Salisbury? Check the same-question pair and independently review any revised label.",
  [("before_action", "Merrimack Valley High School", "located in Penacook, a village in the northern part of Concord"), ("before_action", "Merrimack Valley High School", "approximately 900 students and serves the communities of Andover, Boscawen, Loudon, Penacook, Salisbury, and Webster")]),
 (207, "How many people does the city", "repair_candidate", ["time_unanchored_population", "town_mislabeled_city"],
  "Asamoah's biography names Mampong as birthplace, and the Mampong article gives 42,037. It calls Mampong a small town, whereas the question calls it a city, and the population lacks a date.",
  "Proposed source-scoped question: What population does the supplied article report for Mampong, the town where Gerald Asamoah was born? Replay and independently review the same-question states.",
  [("before_action", "Gerald Asamoah", "born 3 October 1978 in Mampong, Ghana"), ("before_action", "Mampong", "Mampong is a small town"), ("before_action", "Mampong", "Mampong has a population of 42,037 people")]),
 (208, "What volcano, with a height", "repair_candidate", ["causality_shift_from_tharsis_to_volcano"],
  "The polar-wander article attributes the suspected shift to the great mass of volcanic material in the Tharsis dome, which includes Olympus Mons. It does not identify Olympus Mons alone as the cause, although its article confirms the nearly 22-km height.",
  "Proposed question: Which nearly 22-km-high volcano is included in the Tharsis dome whose volcanic mass scientists linked to shifts in Mars's poles? Check the same-question states and obtain independent review.",
  [("before_action", "True polar wander on Mars", "due to the great mass of volcanic material in the Tharsis dome which includes Olympus Mons"), ("before_action", "Olympus Mons", "a height of nearly 22 km")]),
 (209, "What long-running American arbitration", "repair_candidate", ["present_tense_schedule_unanchored", "paired_answer_already_flagged"],
  "Searching the visible WLAX call sign retrieves its syndicated-program list including Judge Judy, but that historical excerpt does not establish what airs now. Its paired answer row 165 was already marked for a source-scoped schedule repair.",
  "Apply the same question-level draft as row 165, replay retrieval and truncation for both states, and independently review. Do not treat this grounded search as proof of a current lineup.",
  [("retrieved_after_search", "WLAX", "Syndicated programming on WLAX includes"), ("retrieved_after_search", "WLAX", "Judge Judy")]),
 (210, "Ohio Brush Creek rises", "retain_candidate", ["query_copies_visible_context", "question_grammar"],
  "The pre-action creek excerpt names Serpent Mound; the copied search retrieves its article, which explicitly places that mound on a plateau of the Serpent Mound crater. The query is redundant but grounded, and it fills the requested crater relation.", None,
  [("before_action", "Ohio Brush Creek", "past the Serpent Mound"), ("retrieved_after_search", "Serpent Mound", "on a plateau of the Serpent Mound crater along Ohio Brush Creek")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-021.jsonl")
