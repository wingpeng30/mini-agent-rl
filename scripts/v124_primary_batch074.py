"""Primary examination of v122 internal candidate states 281–290."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (281, "The ethnic group of which Rev. Charles Bluejacket", "repair_candidate", ["language_family_not_specific_language", "question_wording"],
  "The visible Bluejacket article establishes Shawnee, and retrieval adds a Shawnee article describing them as Algonquian-speaking. That describes a language family, not a specific language as the question asks. The query has a legitimate Shawnee anchor, but the target wording requires correction.",
  "Ask which language family the Shawnee speak, answer Algonquian, and replay the paired state.",
  [("before_action", "Bluejacket, Oklahoma", "one-time chief of the Shawnee"), ("retrieved_after_search", "Shawnee", "an Algonquian-speaking ethnic group")]),
 (282, "The ethnic group of which Rev. Charles Bluejacket", "repair_candidate", ["paired_language_family_not_specific_language"],
  "The two-document chain identifies Shawnee and an Algonquian-speaking ethnic group. 'Algonquian-speaking' names a family-level descriptor rather than the particular language implied by the original question.",
  "Use the family-specific revised question and concise Algonquian answer only after replay and review.",
  [("before_action", "Bluejacket, Oklahoma", "one-time chief of the Shawnee"), ("before_action", "Shawnee", "an Algonquian-speaking ethnic group")]),
 (283, "In which war did \"Cognac\"", "repair_candidate", ["answer_visible_before_search", "query_copies_unrelated_campaign_clause"],
  "The single initial Karol Pniak article already calls him a Polish World War II flying ace and connects Cognac and five victories to the Battle of Britain. The query copies his September Campaign sentence; retrieval is unnecessary for the requested war.",
  "Use a direct-answer state if retained; do not train this as a necessary search step.",
  [("before_action", "Karol Pniak", "Polish World War II flying ace"), ("before_action", "Karol Pniak", "known by the nickname \"Cognac.\"")]),
 (284, "In which war did \"Cognac\"", "retain_candidate", ["answer_already_visible_before_paired_search"],
  "World War II is stated in Karol Pniak's initial biography; the retrieved Battle of Britain article independently labels that battle a Second World War campaign. The answer is supported, although the paired search is redundant.", None,
  [("before_action", "Karol Pniak", "Polish World War II flying ace"), ("before_action", "Battle of Britain", "campaign of the Second World War")]),
 (285, "What Hungarian composer wrote a piece", "repair_candidate", ["answer_visible_before_search", "query_copies_non_target_sentence"],
  "The initial Suite, Op. 14 article explicitly names Béla Bartók and all three rhythmic influences. Searching a separate folk-melody sentence does not fill a missing relation; the returned Bartók biography merely reconfirms an already visible answer.",
  "Retain only as a direct-answer example if needed; remove the redundant search target after paired replay.",
  [("before_action", "Suite, Op. 14 (Bartók)", "piece for solo piano written by Béla Bartók"), ("before_action", "Suite, Op. 14 (Bartók)", "Romanian, Arabic, and North African rhythmic influences")]),
 (286, "What Hungarian composer wrote a piece", "retain_candidate", ["answer_already_visible_before_paired_search"],
  "The suite article names Béla Bartók as composer and states the exact rhythmic-influence relation; a separate biography labels him Hungarian. The answer is supported without the paired search.", None,
  [("before_action", "Suite, Op. 14 (Bartók)", "written by Béla Bartók"), ("before_action", "Béla Bartók", "was a Hungarian composer")]),
 (287, "In what city is the college at which Christine Gardner", "repair_candidate", ["query_selects_college_before_person_link", "retrieval_fills_missing_person_link"],
  "The initial Wheaton College article gives the city, but none of the initial snippets associates Christine Gardner with that college. The search chooses Wheaton College prematurely; retrieval subsequently finds the missing Gardner link, which cannot justify the earlier choice.",
  "First query Christine Gardner and her college, then use a supported college-to-city lookup only if necessary; replay the pair.",
  [("before_action", "Wheaton College (Illinois)", "located in Wheaton, Illinois"), ("retrieved_after_search", "Christine Gardner", "professor at Wheaton College")]),
 (288, "In what city is the college at which Christine Gardner", "repair_candidate", ["paired_query_preselects_unlinked_college"],
  "Wheaton, Illinois is supported after retrieval combines Gardner's college with its location. The preceding search nevertheless selected the college before the initial state established that Gardner taught there.",
  "Keep the answer only as part of a replayed pair with a question-grounded first search.",
  [("before_action", "Christine Gardner", "professor at Wheaton College"), ("before_action", "Wheaton College (Illinois)", "located in Wheaton, Illinois")]),
 (289, "Where are the headquarters of the company which developed GoTa", "retain_candidate", ["query_uses_visible_developer_link"],
  "The initial GoTa text names ZTE as developer; searching that developer retrieves the headquarters line. The full two-hop chain is grounded at action time.", None,
  [("before_action", "GoTa", "developed by ZTE"), ("retrieved_after_search", "ZTE", "headquartered in Shenzhen, Guangdong")]),
 (290, "Where are the headquarters of the company which developed GoTa", "retain_candidate", [],
  "GoTa was developed by ZTE, and ZTE's article explicitly places its headquarters in Shenzhen, Guangdong. The answer is supported by separate documents.", None,
  [("before_action", "GoTa", "developed by ZTE"), ("before_action", "ZTE", "headquartered in Shenzhen, Guangdong")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-074.jsonl", cohort="v122_internal_candidates")
