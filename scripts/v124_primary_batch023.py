"""Primary case examination of original training rows 221–230."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (221, "When was the  Australian professional snooker", "retain_candidate", ["query_copies_visible_context", "duplicate_training_action_group"],
  "Calabrese's article names compatriot Neil Robertson; the copied sentence retrieves Robertson's biography, which identifies him as an Australian professional snooker player born 11 February 1982. This is the first of three duplicated searches, so future export needs question/action deduplication.", None,
  [("before_action", "Vinnie Calabrese", "his compatriot Neil Robertson"), ("retrieved_after_search", "Neil Robertson (snooker player)", "born 11 February 1982"), ("retrieved_after_search", "Neil Robertson (snooker player)", "an Australian professional snooker player")]),
 (222, "When was the  Australian professional snooker", "repair_candidate", ["duplicate_training_action"],
  "Same question, query and result as row 221, with a different irrelevant distractor. This should not add another independent training label.",
  "Preserve the row but deduplicate the question/action group after paired answer and provenance checks.",
  [("before_action", "Vinnie Calabrese", "his compatriot Neil Robertson"), ("retrieved_after_search", "Neil Robertson (snooker player)", "born 11 February 1982")]),
 (223, "When was the  Australian professional snooker", "repair_candidate", ["duplicate_training_action"],
  "Same question and query as rows 221–222; the exact target biography is retrieved again. Fewer distractors do not make this a new target.",
  "Choose a justified single representative, retaining original duplicates for audit and checking all answer states.",
  [("before_action", "Vinnie Calabrese", "his compatriot Neil Robertson"), ("retrieved_after_search", "Neil Robertson (snooker player)", "born 11 February 1982")]),
 (224, "Who was perhaps not the only king", "retain_candidate", [],
  "Centwine is named before the action with the distinctive perhaps-not-only-king description. Querying his name retrieves Cissa's article, which attributes both Chisbury Camp and Abingdon Abbey to the reported viceroy of King Centwine.", None,
  [("before_action", "Centwine of Wessex", "perhaps not the only king of the West Saxons at the time"), ("retrieved_after_search", "Cissa (West Saxon)", "viceroy of king Centwine of Wessex"), ("retrieved_after_search", "Cissa (West Saxon)", "constructed Chisbury Camp, and to have founded Abingdon Abbey")]),
 (225, "Who was born first, Gerhard Klopfer", "retain_candidate", [],
  "The two biographies give Klopfer's 1905 birth and Martin Bormann's 1900 birth. The answer Martin Bormann is temporally supported, and the similarly named son is distinguishable.", None,
  [("before_action", "Gerhard Klopfer", "18 February 1905"), ("before_action", "Martin Bormann", "17 June 1900")]),
 (226, "Who was born first, Gerhard Klopfer", "retain_candidate", ["query_restates_question"],
  "Only Klopfer's 1905 birth is available before the search. The query includes both candidate names from the question and retrieves Martin Bormann's 1900 birth, filling the missing comparison fact.", None,
  [("before_action", "Gerhard Klopfer", "18 February 1905"), ("retrieved_after_search", "Martin Bormann", "17 June 1900")]),
 (227, "The first suburban store", "quarantine_pending_repair", ["present_tense_occupant_unanchored", "paired_group_quarantine"],
  "Fred Meyer is correctly identified from the pre-action founding clue, and the search retrieves the Hollywood Theatre excerpt saying the former store is currently occupied by Rite Aid. That excerpt has no as-of date. The same-question answer and search states at rows 67 and 179 are already quarantined.",
  "Keep the entire same-question group isolated until a dated source supports a defined historical occupancy question; do not derive an as-of year from neighboring unrelated dates.",
  [("before_action", "Fred Meyer", "founded in 1922 in Portland, Oregon"), ("retrieved_after_search", "Hollywood Theatre (Portland, Oregon)", "first suburban Fred Meyer store, which is currently occupied by Rite Aid")]),
 (228, "An Equal Music, published", "retain_candidate", ["query_targets_author_not_work"],
  "The pre-action author biography has no publication year for An Equal Music. A query for the already-known author nevertheless retrieves the book article and its explicit 1999 year. This is an observed success but not strong evidence of good query targeting; pair with answer row 184.", None,
  [("before_action", "Vikram Seth", "an Indian novelist and poet"), ("retrieved_after_search", "An Equal Music", "An Equal Music (1999) is a novel by Vikram Seth")]),
 (229, "Who created the first alternative education", "repair_candidate", ["program_location_conflated", "duplicate_training_action_group"],
  "Oneonta matches the city clue and the retrieved school article names creators Mike Newell and Mark Rathbun. However it calls the school 'in Hartwick' and says it was first located in a Oneonta church; calling the program simply 'in a city' erases this location distinction.",
  "Proposed question: Who created the first alternative education program in Otsego County, which was initially located in Oneonta, the City of the Hills? Verify the paired answer and the same-query duplicate before independent review.",
  [("before_action", "Oneonta, New York", "a city in southern Otsego County"), ("before_action", "Oneonta, New York", "Its nickname is \"City of the Hills.\""), ("retrieved_after_search", "Upward Bound High School", "Created by English teacher Mike Newell and principal Mark Rathbun"), ("retrieved_after_search", "Upward Bound High School", "school was first located in the basement of a Unitarian church in Oneonta")]),
 (230, "Who created the first alternative education", "repair_candidate", ["program_location_conflated", "duplicate_training_action"],
  "The duplicate Oneonta query retrieves the same first-alternative-program and creator evidence as row 229. The original wording obscures the school's Hartwick designation versus its first Oneonta location, and duplicate supervision would overweight the same path.",
  "Apply one consistent source-scoped question repair and retain at most one justified search representative after answer-state replay.",
  [("before_action", "Oneonta, New York", "Its nickname is \"City of the Hills.\""), ("retrieved_after_search", "Upward Bound High School", "High School in Hartwick, New York"), ("retrieved_after_search", "Upward Bound High School", "first located in the basement of a Unitarian church in Oneonta")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-023.jsonl")
