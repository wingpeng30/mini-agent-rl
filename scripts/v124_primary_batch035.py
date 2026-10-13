"""Primary case examination of original training rows 341–345."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (341, "The administrative center of what region", "retain_candidate", ["administrative_center_outside_region_boundary"],
  "Masaliyev's biographical line gives Bishkek as death place, and the retrieved Bishkek article names it administrative center of Chuy Region. The article also says the city itself is a separate province-level unit, so this is an administrative-center relation, not physical containment in the region.", None,
  [("before_action", "Absamat Masaliyev", "July 31, 2004, Bishkek"), ("retrieved_after_search", "Bishkek", "administrative center of the Chuy Region"), ("retrieved_after_search", "Bishkek", "the city itself is not part of the province")]),
 (342, "Do Jane Campion and Werner Schroeter", "retain_candidate", ["query_biases_known_person"],
  "Schroeter is identified as German before the action; the query emphasizing him nevertheless retrieves Jane Campion's New Zealand nationality. The two cited biographies support a negative same-nationality answer.", None,
  [("before_action", "Werner Schroeter", "was a German film director"), ("retrieved_after_search", "Jane Campion", "is a New Zealand screenwriter, producer, and director")]),
 (343, "Southbridge Municipal Airport", "repair_candidate", ["town_mislabeled_city", "paired_answer_requires_amendment"],
  "The event article identifies the June 1, 2011 tornado outbreak; querying it retrieves the airport article explicitly naming damage from that event. The question, however, calls the Town of Southbridge 'a city', a contradictory municipality description. The same-question answer row 4 was previously retained and must be amended consistently.",
  "Remove the unsupported 'a city' apposition while preserving the event clue; replay search and answer states and independently review the revised question. Amend row 4's effective primary disposition without overwriting its old vote.",
  [("before_action", "2011 New England tornado outbreak", "On June 1, 2011, seven tornadoes appeared"), ("retrieved_after_search", "Southbridge Municipal Airport", "Town of Southbridge"), ("retrieved_after_search", "Southbridge Municipal Airport", "airport sustained damage on June 1st from the 2011 New England tornado outbreak")]),
 (344, "Which actor, singer, and director", "retain_candidate", ["query_copies_song_sentence"],
  "The song excerpt names Jack Cassidy as its performer and Harold Rome as lyricist; the copied-song-sentence search retrieves Jack Cassidy's biography identifying him as actor, singer and director. This resolves the person without confusing him with David Cassidy.", None,
  [("before_action", "Wish You Were Here (1952 song)", "introduced in the show and on the RCA Victor cast album by Jack Cassidy"), ("retrieved_after_search", "Jack Cassidy", "an American actor, singer and director")]),
 (345, "What is the name of this industrial city", "retain_candidate", ["query_restates_question"],
  "Šetkus's biography gives Tauragė as birthplace; the retrieved Tauragė article identifies it as an industrial city in Lithuania. The other industrial city distractor is distinguishable by the exact birthplace link.", None,
  [("before_action", "Ernestas Šetkus", "born 25 May 1985 in Tauragė"), ("retrieved_after_search", "Tauragė", "an industrial city in Lithuania")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-035.jsonl")
