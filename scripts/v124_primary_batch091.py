"""Primary examination of v122 training-supplement candidate states 121–130."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (121, "What booking agency represents a woman who was assigned male at birth", "quarantine_pending_repair", ["group_category_substituted_for_specific_person", "no_named_representation_link", "query_on_unrelated_identity_definition"],
  "The initial Trans woman article defines a category and Muksamse'lapli article discusses a different named person. The retrieved Discwoman article says the agency represents female-identified talent including trans women, but identifies no specific represented woman and no individual's assigned sex. Category inclusion does not prove the question's person-to-agency relation.",
  "Isolate both states. A representation question would need a named artist with a direct agency link and consent-appropriate public identity evidence; do not infer an individual's birth assignment from a category.",
  [("before_action", "Trans woman", "a woman who was assigned male at birth"), ("retrieved_after_search", "Discwoman", "representing and showcasing female-identified (cis women, transwomen, and gender queer) talent")]),
 (122, "What booking agency represents a woman who was assigned male at birth", "quarantine_pending_repair", ["paired_category_to_person_inference", "no_named_representation_link"],
  "Discwoman is described as representing groups that include trans women. Neither that statement nor the separate definition identifies the unnamed person in the question or establishes that person is represented by Discwoman. The frozen answer is unsupported at the individual level.",
  "Do not qualify this pair or construct a target from demographic inference.",
  [("before_action", "Trans woman", "a woman who was assigned male at birth"), ("before_action", "Discwoman", "representing and showcasing female-identified (cis women, transwomen, and gender queer) talent")]),
 (123, "the chemical element Lawrencium", "repair_candidate", ["query_preselects_eponym_before_element_link", "birth_year_visible_before_search", "question_case"],
  "The initial Ernest Lawrence biography supplies 1901 but does not mention lawrencium. The query selects Ernest Lawrence by name before the element's eponym is visible; retrieval then confirms that link. The question also begins lowercase and has stray spacing.",
  "Search Lawrencium's namesake first, then use the scientist's birth date; replay with corrected wording.",
  [("before_action", "Ernest Lawrence", "August 8, 1901"), ("retrieved_after_search", "Lawrencium", "named in honor of Ernest Lawrence")]),
 (124, "the chemical element Lawrencium", "repair_candidate", ["paired_eponym_preselected", "paired_question_format"],
  "Lawrencium's retrieved article names Ernest Lawrence and his biography gives 1901. The answer is supported only after the eponym document appears; the prior query guessed the person and the frozen question wording needs correction.",
  "Retain 1901 only in a question-grounded, replayed pair.",
  [("before_action", "Lawrencium", "named in honor of Ernest Lawrence"), ("before_action", "Ernest Lawrence", "August 8, 1901")]),
 (125, "Ted Simmons was often overshadowed", "repair_candidate", ["query_omits_visible_johnny_bench", "retrieval_accidental"],
  "The initial Ted Simmons article already names Johnny Bench as the overshadowing contemporary. The query copies an unrelated Simmons team-career sentence and omits Bench; retrieval happens to return Bench's Reds biography. The target missing Bench-team relation was available for a direct grounded query.",
  "Query Johnny Bench's team, then replay the paired answer.",
  [("before_action", "Ted Simmons", "overshadowed by his contemporary, Johnny Bench"), ("retrieved_after_search", "Johnny Bench", "played in the Major Leagues for the Cincinnati Reds")]),
 (126, "Ted Simmons was often overshadowed", "repair_candidate", ["paired_query_omits_target_bench"],
  "Simmons's article identifies Johnny Bench and Bench's biography gives the Cincinnati Reds, so the answer is supported. The paired query nonetheless follows Simmons's own teams rather than Bench's missing team.",
  "Use Reds only with a direct Bench-team replayed search.",
  [("before_action", "Ted Simmons", "overshadowed by his contemporary, Johnny Bench"), ("before_action", "Johnny Bench", "Cincinnati Reds from 1967 to 1983")]),
 (127, "What nationality was the person whose music inspired", "retain_candidate", ["query_uses_visible_composer_link"],
  "The Folias article explicitly says Astor Piazzolla's music inspired the duo. Searching that sentence retrieves Piazzolla's biography, which calls him an Argentine tango composer; the nationality answer is grounded.", None,
  [("before_action", "Folias Flute and Guitar Duo", "music of Astor Piazzolla"), ("retrieved_after_search", "Astor Piazzolla", "was an Argentine tango composer")]),
 (128, "What nationality was the person whose music inspired", "retain_candidate", [],
  "The named inspiration is Astor Piazzolla and his biography explicitly says Argentine. The answer is supported; Bulgarian and Czech duo articles are distractors.", None,
  [("before_action", "Folias Flute and Guitar Duo", "music of Astor Piazzolla"), ("before_action", "Astor Piazzolla", "Argentine tango composer")]),
 (129, "Benjamin Furly was a friend", "retain_candidate", ["query_uses_visible_friend_link"],
  "The initial Furly article identifies John Locke as his friend. Searching that line retrieves Locke's biography, which gives 1632 and describes him as a philosopher.", None,
  [("before_action", "Benjamin Furly", "friend of John Locke"), ("retrieved_after_search", "John Locke", "29 August 1632"), ("retrieved_after_search", "John Locke", "English philosopher and physician")]),
 (130, "Benjamin Furly was a friend", "retain_candidate", [],
  "Benjamin Furly's friend is John Locke, and Locke's biography supplies the 1632 birth year and philosopher qualification. The answer is supported.", None,
  [("before_action", "Benjamin Furly", "friend of John Locke"), ("before_action", "John Locke", "29 August 1632")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-091.jsonl", cohort="v122_train_candidates")
