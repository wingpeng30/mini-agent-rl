"""Primary examination of v123 internal candidate states 211–220."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (211, "Toby Kebbell worked on the final episode", "retain_candidate", ["program_name_visible_before_search", "retrieval_confirms_final_episode"],
  "Kebbell's initial biography already names Black Mirror and its episode The Entire History of You. The search is useful only for confirming that this episode was the first series' final installment, which retrieval does. It should not be counted as newly discovering the program name.", None,
  [("before_action", "Toby Kebbell", "\"Black Mirror\" episode, \"The Entire History of You\""), ("retrieved_after_search", "The Entire History of You", "third and final episode of the first series")]),
 (212, "Toby Kebbell worked on the final episode", "retain_candidate", [],
  "The Kebbell passage connects him to The Entire History of You, and the episode passage identifies it as the final episode of Black Mirror's first series. The answer is supported, although the series name was present before search.", None,
  [("before_action", "Toby Kebbell", "\"Black Mirror\" episode, \"The Entire History of You\""), ("before_action", "The Entire History of You", "final episode of the first series of British science fiction anthology series \"Black Mirror\"")]),
 (213, "Which treaty ended the war", "repair_candidate", ["query_preselects_war_without_battle_link", "retrieval_adds_battle_relation"],
  "The initial context gives the Franco-Dutch War's ending treaty but does not connect Battle of Stromboli to that war. The query jumps straight to Franco-Dutch War; retrieval later supplies the missing battle-war link. The search choice relies on an unestablished association rather than the battle named in the question.",
  "Search Battle of Stromboli war first, then use the documented war-to-treaty passage in a replay.",
  [("before_action", "Franco-Dutch War", "war ended with the Treaty of Nijmegen"), ("retrieved_after_search", "Battle of Stromboli", "during the Franco-Dutch War")]),
 (214, "Which treaty ended the war", "repair_candidate", ["paired_war_preselection", "answer_supported"],
  "The Treaty of Nijmegen is documented as ending the Franco-Dutch War, and retrieval establishes that Stromboli occurred during that war. The answer is supported but the paired query selected the war before its battle relation was available.",
  "Review after a battle-first retrieval replay.",
  [("before_action", "Franco-Dutch War", "war ended with the Treaty of Nijmegen"), ("before_action", "Battle of Stromboli", "during the Franco-Dutch War")]),
 (215, "Which branch of dialect is the Alemannic Wikipedia", "quarantine_pending_repair", ["question_dialect_branch_confusion", "all_alemannic_not_low_only"],
  "The Wikipedia passage says it expanded from Alsatian to accept all Alemannic dialects. The query chooses Low Alemannic despite no basis to restrict the wiki to that branch; retrieval says Alemannic is part of Upper German, while the frozen answer is merely German, not a dialect branch. Question and target ontology disagree.",
  "Decide whether asking about the accepted dialect group or its Upper German family; rewrite and fully review.",
  [("before_action", "Alemannic Wikipedia", "all Alemannic dialects are accepted"), ("before_action", "Low Alemannic German", "branch of Alemannic German"), ("retrieved_after_search", "Alemannic German", "Upper German branch")]),
 (216, "Which branch of dialect is the Alemannic Wikipedia", "quarantine_pending_repair", ["paired_answer_wrong_granularity", "paired_all_dialects_scope"],
  "German is not the branch asked for, and the source says the Alemannic Wikipedia accepts all Alemannic dialects rather than one Low Alemannic branch. The answer cannot be rescued by the retrieved dialect taxonomy alone.",
  "Create a well-defined dialect-group question and a new answer target.",
  [("before_action", "Alemannic Wikipedia", "all Alemannic dialects are accepted"), ("before_action", "Alemannic German", "Upper German branch")]),
 (217, "What is the home stadium of the University where Roy Finch", "retain_candidate", ["query_uses_visible_team", "retrieval_adds_stadium"],
  "The initial Roy Finch article says he played for the Oklahoma Sooners. Retrieval identifies the Sooners as the University of Oklahoma's team and gives its home stadium. The question loosely attributes a football team's home stadium to the university, but the intended team link is explicit.", None,
  [("before_action", "Roy Finch (gridiron football)", "college football for the Oklahoma Sooners"), ("retrieved_after_search", "Oklahoma Sooners football", "home games at Gaylord Family Oklahoma Memorial Stadium")]),
 (218, "What is the home stadium of the University where Roy Finch", "retain_candidate", ["minor_team_university_wording"],
  "The Sooners are explicitly the University of Oklahoma football team, and that team plays home games at Gaylord Family Oklahoma Memorial Stadium. The answer is supported under the natural sports-program reading.", None,
  [("before_action", "Roy Finch (gridiron football)", "college football for the Oklahoma Sooners"), ("before_action", "Oklahoma Sooners football", "represents the University of Oklahoma"), ("before_action", "Oklahoma Sooners football", "home games at Gaylord Family Oklahoma Memorial Stadium")]),
 (219, "What nationality is the singer behind", "retain_candidate", ["query_uses_visible_artist", "retrieval_adds_nationality"],
  "The song article identifies BoA as the artist. The retrieved BoA biography explicitly calls her a South Korean singer; a Japan release does not override that nationality.", None,
  [("before_action", "Amazing Kiss", "BoA's second single in Japan"), ("retrieved_after_search", "BoA", "South Korean singer")]),
 (220, "What nationality is the singer behind", "retain_candidate", [],
  "BoA is tied to Amazing Kiss in the initial song article and identified as a South Korean singer in the retrieved biography. South Korean answers the nationality question.", None,
  [("before_action", "Amazing Kiss", "BoA's second single in Japan"), ("before_action", "BoA", "South Korean singer")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-122.jsonl", cohort="v123_internal_candidates")
