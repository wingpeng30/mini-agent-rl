"""Primary examination of v123 internal candidate states 201–210."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (201, "Which German businessman co-founded Too Much Media", "retain_candidate", ["query_uses_visible_founder_list", "retrieval_adds_nationality"],
  "The initial company article lists Fabian Thylmann among the founders. Retrieval independently identifies him as a German businessman, completing the nationality filter on the founder list.", None,
  [("before_action", "Too Much Media", "founded in 2003 by John Albright, Fabian Thylmann and Charles Berrebbi"), ("retrieved_after_search", "Fabian Thylmann", "German businessman")]),
 (202, "Which German businessman co-founded Too Much Media", "retain_candidate", [],
  "The founder list and Fabian Thylmann's German-businessman biography are both visible by answer time. The answer is supported without relying on the question's premise alone.", None,
  [("before_action", "Too Much Media", "Fabian Thylmann and Charles Berrebbi"), ("before_action", "Fabian Thylmann", "German businessman")]),
 (203, "What was the population (according to the 2010 census)", "retain_candidate", ["query_uses_visible_city", "retrieval_adds_dated_population"],
  "The courthouse is initially placed in Cimarron, Kansas, and the retrieved city article gives 2,184 residents as of the 2010 census. The question states that date explicitly.", None,
  [("before_action", "Old Gray County Courthouse", "Cimarron, Kansas"), ("retrieved_after_search", "Cimarron, Kansas", "As of the 2010 census, the city population was 2,184")]),
 (204, "What was the population (according to the 2010 census)", "retain_candidate", [],
  "The Old Gray County Courthouse is in Cimarron, Kansas; the city passage supplies the 2010 census figure 2,184. No undated current-population inference is needed.", None,
  [("before_action", "Old Gray County Courthouse", "Cimarron, Kansas"), ("before_action", "Cimarron, Kansas", "2010 census, the city population was 2,184")]),
 (205, "What airline did one of the", "quarantine_pending_repair", ["both_collision_airlines_big_four", "query_unrelated_carmel_company"],
  "The collision passage names Eastern and TWA; the retrieved TWA article identifies both Eastern and TWA as members of the Big Four. Hence 'one of the Big Four collided with which airline' can point to either airline depending on which one is the unnamed subject. The query adds an unrelated Carmel car-service company.",
  "Name TWA or Eastern as the subject and query the collision directly; rebuild both states.",
  [("before_action", "1965 Carmel mid-air collision", "Eastern Air Lines Flight 853"), ("before_action", "1965 Carmel mid-air collision", "Trans World Airlines Flight 42"), ("retrieved_after_search", "Trans World Airlines", "With American, United, and Eastern, it was one of the \"Big Four\"")]),
 (206, "What airline did one of the", "quarantine_pending_repair", ["paired_bidirectional_ambiguity"],
  "Eastern Air Lines is a collision party, but so is TWA, and the Big Four source names both. The original wording does not identify which Big Four airline should be considered the first party, so Eastern is not uniquely selected.",
  "Specify the first airline in the question before approving a partner answer.",
  [("before_action", "1965 Carmel mid-air collision", "Eastern Air Lines Flight 853"), ("before_action", "1965 Carmel mid-air collision", "Trans World Airlines Flight 42"), ("before_action", "Trans World Airlines", "With American, United, and Eastern, it was one of the \"Big Four\"")]),
 (207, "What state is the university, in which Albert Levitt", "retain_candidate", ["query_uses_visible_university", "retrieval_adds_state"],
  "Albert Levitt's article identifies Washington and Lee University as a place where he taught. The retrieved university article puts it in Lexington, Virginia, giving the requested state.", None,
  [("before_action", "Albert Levitt", "teacher at Washington and Lee University"), ("retrieved_after_search", "Washington and Lee University", "Lexington, Virginia")]),
 (208, "What state is the university, in which Albert Levitt", "retain_candidate", [],
  "The teacher-university link and Virginia university location are explicit at answer time. Virginia is the requested state.", None,
  [("before_action", "Albert Levitt", "teacher at Washington and Lee University"), ("before_action", "Washington and Lee University", "Lexington, Virginia")]),
 (209, "What year was the recorder of Somebody Said a Prayer", "repair_candidate", ["malformed_song_quote", "recorder_wording", "retrieval_adds_birth_year"],
  "The initial song passage names Billy Ray Cyrus as the recording artist, and retrieval gives his 1961 birth year. The question has an unmatched quote and calls him 'the recorder', which can mean equipment rather than performer; these editorial defects need repair even though the search is grounded.",
  "Ask 'In what year was Billy Ray Cyrus, who recorded Somebody Said a Prayer, born?' and replay.",
  [("before_action", "Somebody Said a Prayer", "recorded by American country music artist Billy Ray Cyrus"), ("retrieved_after_search", "Billy Ray Cyrus", "born August 25, 1961")]),
 (210, "What year was the recorder of Somebody Said a Prayer", "repair_candidate", ["paired_malformed_question", "answer_supported"],
  "1961 is Billy Ray Cyrus's documented birth year; the initial song document connects him to the recording. The original question wording and quote are malformed, so the answer needs a corrected pair.",
  "Use a clean performer/birth-year question and independent review.",
  [("before_action", "Somebody Said a Prayer", "recorded by American country music artist Billy Ray Cyrus"), ("before_action", "Billy Ray Cyrus", "born August 25, 1961")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-121.jsonl", cohort="v123_internal_candidates")
