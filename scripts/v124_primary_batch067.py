"""Primary examination of v122 internal candidate states 211–220."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (211, "What is the nationality of the UFC Heavyweight Champion", "repair_candidate", ["query_targets_inaugural_event_not_art_of_war_12_guest"],
  "The initial Art of War article names Tim Sylvia among guests of Art of War 12. The query instead copies a sentence about the inaugural Art of War I event, unrelated to Sylvia's nationality. Retrieval happens to bring in Sylvia's biography, which supports American, but the search action is misdirected.",
  "Search Tim Sylvia from the visible Art of War 12 guest sentence, then replay and review.",
  [("before_action", "Art of War Fighting Championship", "former UFC Heavyweight Champion Tim Sylvia"), ("before_action", "Art of War Fighting Championship", "inaugural event made it the first professional mixed martial arts contest"), ("retrieved_after_search", "Tim Sylvia", "retired American mixed martial arts")]),
 (212, "What is the nationality of the UFC Heavyweight Champion", "repair_candidate", ["paired_query_targets_inaugural_event"],
  "American is supported for Tim Sylvia, the named Art of War 12 guest, but the preceding search targets an unrelated inaugural-event sentence. The paired action cannot be retained unchanged.",
  "Keep the answer only after a Sylvia-targeted query is replayed.",
  [("before_action", "Art of War Fighting Championship", "former UFC Heavyweight Champion Tim Sylvia"), ("before_action", "Tim Sylvia", "retired American mixed martial arts")]),
 (213, "Century eggs are made", "retain_candidate", ["query_uses_visible_quicklime_ingredient"],
  "The century-egg article lists quicklime as an ingredient. The search preserves that ingredient, and retrieval supplies a separate chemistry article equating quicklime with calcium oxide, the missing requested chemical name.", None,
  [("before_action", "Century egg", "clay, ash, salt, quicklime, and rice hulls"), ("retrieved_after_search", "Calcium oxide", "commonly known as quicklime")]),
 (214, "Century eggs are made", "retain_candidate", [],
  "The food article gives quicklime; the chemical article says quicklime is calcium oxide. The answer is supported by the synonym bridge.", None,
  [("before_action", "Century egg", "clay, ash, salt, quicklime, and rice hulls"), ("before_action", "Calcium oxide", "commonly known as quicklime")]),
 (215, "What number prime minister of Canada", "retain_candidate", ["search_verifies_ministry_number_mapping"],
  "The initial Stephen Harper biography gives his ordinal as 22nd prime minister, but does not identify the Twenty-Eighth Ministry. Searching Harper retrieves the ministry article confirming he chaired it. The ordinal and ministry number are distinct and correctly linked.", None,
  [("before_action", "Stephen Harper", "served as the 22nd Prime Minister of Canada"), ("retrieved_after_search", "28th Canadian Ministry", "chaired by Prime Minister Stephen Harper")]),
 (216, "What number prime minister of Canada", "retain_candidate", [],
  "The Twenty-Eighth Ministry article names Stephen Harper as chair; Harper's biography explicitly calls him the 22nd prime minister. The answer does not confuse cabinet numbering with prime-minister numbering.", None,
  [("before_action", "Stephen Harper", "22nd Prime Minister of Canada"), ("before_action", "28th Canadian Ministry", "chaired by Prime Minister Stephen Harper")]),
 (217, "Go North starred the eldest son", "repair_candidate", ["elder_vs_eldest_overstatement", "bodybuilder_qualification_not_in_visible_sources"],
  "Go North lists Patrick Schwarzenegger. His biography calls him the elder son of Arnold Schwarzenegger and Maria Shriver; it does not establish an absolute 'eldest son' claim or Arnold's bodybuilder qualification in the visible excerpts. The answer identity is plausible and the parent link is direct, but the full question is not fully supported here.",
  "Ask which parent of actor Patrick Schwarzenegger starred in Go North, or add independent visible evidence for the bodybuilder and eldest-son qualifiers; replay.",
  [("before_action", "Go North (film)", "Patrick Schwarzenegger"), ("retrieved_after_search", "Patrick Schwarzenegger", "elder son of Arnold Schwarzenegger and Maria Shriver")]),
 (218, "Go North starred the eldest son", "repair_candidate", ["paired_unverified_bodybuilder_and_eldest_qualifiers"],
  "Arnold Schwarzenegger is explicitly Patrick's father, but the excerpt does not support the additional bodybuilder or absolute eldest-son qualifiers. The answer can be retained only with narrower wording or stronger evidence.",
  "Retain Arnold only in a fully evidenced, replayed pair.",
  [("before_action", "Go North (film)", "Patrick Schwarzenegger"), ("before_action", "Patrick Schwarzenegger", "elder son of Arnold Schwarzenegger and Maria Shriver")]),
 (219, "which property in Oklahoma", "quarantine_pending_repair", ["art_collection_location_splice", "chain_location_not_work_location"],
  "Özkaya's article names three 21c Museum Hotels locations holding his work: Louisville, Bentonville and Nashville. The retrieved chain article separately lists an Oklahoma City property. It does not say Özkaya's work is held at that Oklahoma property; the question splices chain presence into artwork presence.",
  "Exclude the frozen pair. A new Oklahoma claim needs explicit artwork-at-that-property evidence; otherwise ask for one of the three documented locations.",
  [("before_action", "Serkan Özkaya", "three different 21c Museum Hotels locations—Louisville, Kentucky; Bentonville, Arkansas; and Nashville, Tennessee"), ("retrieved_after_search", "21c Museum Hotels", "Oklahoma City, OK")]),
 (220, "which property in Oklahoma", "quarantine_pending_repair", ["unsupported_artwork_at_oklahoma_property", "answer_is_chain_not_specific_property"],
  "21c Museum Hotels operates in Oklahoma City in the retrieved snapshot, but the artist source lists his works only at three other chain locations. The frozen answer gives the chain name rather than proving an Oklahoma property holding his work.",
  "Keep quarantined pending direct site-specific evidence and a specific property answer.",
  [("before_action", "Serkan Özkaya", "three different 21c Museum Hotels locations—Louisville, Kentucky; Bentonville, Arkansas; and Nashville, Tennessee"), ("before_action", "21c Museum Hotels", "Oklahoma City, OK")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-067.jsonl", cohort="v122_internal_candidates")
