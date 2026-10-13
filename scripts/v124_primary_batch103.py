"""Primary examination of v123 internal candidate states 21–30."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (21, '"Mugby Junction" was first published', "retain_candidate", ["query_uses_visible_magazine", "retrieval_adds_publication_country"],
  "Mugby Junction's initial excerpt names All the Year Round as its first-publication magazine. Searching that visible title retrieves a magazine article stating it was published throughout the United Kingdom.", None,
  [("before_action", "Mugby Junction", 'magazine "All the Year Round"'), ("retrieved_after_search", "All the Year Round", "published between 1859 and 1895 throughout the United Kingdom")]),
 (22, '"Mugby Junction" was first published', "retain_candidate", [],
  "The first-publication relation to All the Year Round and the magazine's United Kingdom publication location are both directly supported. The answer is a country rather than only the adjective British.", None,
  [("before_action", "Mugby Junction", 'magazine "All the Year Round"'), ("before_action", "All the Year Round", "throughout the United Kingdom")]),
 (23, "What is the nationality of the person who played", "retain_candidate", ["query_uses_visible_film_actor", "retrieval_adds_nationality"],
  "The initial Mind Games film excerpt identifies Fiona Shaw as its protagonist. The query copies that grounded film and actor relation; retrieval finds her biography calling her Irish. The similarly titled album and magazine are distractors, not sources for the protagonist role.", None,
  [("before_action", "Mind Games (2001 film)", "starring Fiona Shaw as the protagonist"), ("retrieved_after_search", "Fiona Shaw", "Irish actress")]),
 (24, "What is the nationality of the person who played", "retain_candidate", [],
  "The Mind Games film names Fiona Shaw as protagonist, and her retrieved biography identifies her as an Irish actress. The answer is supported by two documents at the correct action time.", None,
  [("before_action", "Mind Games (2001 film)", "Fiona Shaw as the protagonist"), ("before_action", "Fiona Shaw", "Irish actress")]),
 (25, "New Jersey's 11th congressional district is represented", "repair_candidate", ["current_representative_unanchored", "query_uses_snapshot_representative"],
  "The initial district article calls Rodney Frelinghuysen its representative, and retrieval says he had served since 1995. The question states the representation as current without a source date, making a historical snapshot into a timeless officeholder claim.",
  "Date the question's representation status or ask when Frelinghuysen began representing the district as a historical relation, then replay.",
  [("before_action", "New Jersey's 11th congressional district", "represented by Republican Rodney Frelinghuysen"), ("retrieved_after_search", "Rodney Frelinghuysen", "serving since 1995")]),
 (26, "New Jersey's 11th congressional district is represented", "repair_candidate", ["paired_current_representative_unanchored"],
  "1995 is supported as the start of Rodney Frelinghuysen's service in the retrieved biography. The undated question nevertheless calls him the district's present representative, which the frozen historical excerpt cannot support indefinitely.",
  "State a historical service-start question or a valid as-of date before retaining the answer.",
  [("before_action", "New Jersey's 11th congressional district", "represented by Republican Rodney Frelinghuysen"), ("before_action", "Rodney Frelinghuysen", "serving since 1995")]),
 (27, "Which band's debut album is named after a hockey played", "repair_candidate", ["hockey_player_typo", "band_name_visible_before_search", "retrieval_checks_nickname"],
  "The initial Gretzky album article already names Electro Quarterstaff and its Wayne Gretzky namesake. Searching the album title retrieves Gretzky's 'The Great One' nickname, verifying the qualifier; the question mistypes 'player' as 'played'.",
  "Correct the question's hockey-player wording and replay; the query can remain a qualifier check if the input stays equivalent.",
  [("before_action", "Gretzky (album)", "debut full-length album by Canadian instrumental progressive metal band Electro Quarterstaff"), ("retrieved_after_search", "Wayne Gretzky", 'Nicknamed "The Great One"')]),
 (28, "Which band's debut album is named after a hockey played", "repair_candidate", ["paired_hockey_player_typo", "answer_supported"],
  "Electro Quarterstaff is explicitly named on the Gretzky album page, and Wayne Gretzky's nickname is confirmed. The typo in the frozen question should be corrected before treating this as a clean pair.",
  "Use a corrected hockey-player question and rerun the pair-level review.",
  [("before_action", "Gretzky (album)", "band Electro Quarterstaff"), ("before_action", "Wayne Gretzky", 'Nicknamed "The Great One"')]),
 (29, "Britain's Secret Treasures is a British documentary", "retain_candidate", ["query_uses_visible_cohost", "retrieval_adds_specialty"],
  "The initial documentary article identifies Bettany Hughes as Michael Buerk's cohost. Searching the visible pair retrieves Hughes's biography, which states her specialty is classical history.", None,
  [("before_action", "Britain's Secret Treasures", "hosted by Michael Buerk and Bettany Hughes"), ("retrieved_after_search", "Bettany Hughes", "speciality is classical history")]),
 (30, "Britain's Secret Treasures is a British documentary", "retain_candidate", [],
  "Bettany Hughes is the documentary's other host, and her retrieved biography directly states classical history as her specialty. The answer is supported.", None,
  [("before_action", "Britain's Secret Treasures", "Michael Buerk and Bettany Hughes"), ("before_action", "Bettany Hughes", "speciality is classical history")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-103.jsonl", cohort="v123_internal_candidates")
