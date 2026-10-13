"""Primary examination of v122 internal candidate states 11–20."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (11, '"Mugby Junction" was', "retain_candidate", ["query_copies_visible_context"],
  "The stories article names All the Year Round as the Christmas-issue magazine. Searching that sentence retrieves the magazine article, which says it was published throughout the United Kingdom.", None,
  [("before_action", "Mugby Junction", "Christmas edition of the magazine \"All the Year Round\""), ("retrieved_after_search", "All the Year Round", "published between 1859 and 1895 throughout the United Kingdom")]),
 (12, '"Mugby Junction" was', "retain_candidate", [],
  "Mugby Junction's publication is All the Year Round; the magazine article identifies its publication country as the United Kingdom. The answer is supported.", None,
  [("before_action", "Mugby Junction", "magazine \"All the Year Round\""), ("before_action", "All the Year Round", "throughout the United Kingdom")]),
 (13, "What place is not only a capital city", "retain_candidate", ["query_copies_visible_location", "capital_means_provincial_capital"],
  "Tillya Tepe's article places it near Sheberghan. Copying that passage retrieves Sheberghan's article, which calls it the capital of Jowzjan Province; the capital is provincial, not national.", None,
  [("before_action", "Tillya Tepe", "near Sheberghan"), ("retrieved_after_search", "Sheberghan", "capital city of the Jowzjan Province")]),
 (14, "What place is not only a capital city", "retain_candidate", ["capital_means_provincial_capital"],
  "Tillya Tepe is near Sheberghan and the Sheberghan article identifies it as Jowzjan Province's capital city. The answer is supported with provincial scope.", None,
  [("before_action", "Tillya Tepe", "near Sheberghan"), ("before_action", "Sheberghan", "capital city of the Jowzjan Province")]),
 (15, "The Rihanna song \"American Oxygen\"", "quarantine_pending_repair", ["answer_entity_inserted_before_support", "query_assumes_tidal_without_song_relation"],
  "The initial snippets describe Tidal as a streaming service but do not connect it to American Oxygen. The query adds Tidal to the song question before that relation is visible. The returned song article later supplies the link, but that cannot retrospectively ground the search target.", None,
  [("before_action", "Tidal (service)", "a subscription-based music streaming service"), ("retrieved_after_search", "American Oxygen", "made available for streaming on Tidal on April 5, 2015")]),
 (16, "The Rihanna song \"American Oxygen\"", "repair_candidate", ["paired_search_assumes_answer", "answer_supported_at_action_time"],
  "The song article visible at answer time confirms streaming on Tidal on April 5, before the stated digital-download date. The answer is supported, but the paired query preselected Tidal without prior song evidence, so the trajectory needs repair.", None,
  [("before_action", "American Oxygen", "made available for streaming on Tidal on April 5, 2015"), ("before_action", "Tidal (service)", "music streaming service")]),
 (17, "What media formally covered issues", "repair_candidate", ["question_semantics_vague_media_formally_issues", "query_repeats_visible_publisher"],
  "The company snippet links its New Ebony Cookbook and Jet magazine. A title-repeat query retrieves Jet's civil-rights coverage, but 'media formally covered issues from the Civil Rights' is vague and should ask which magazine chronicled the Civil Rights Movement.", None,
  [("before_action", "Johnson Publishing Company", "published \"Jet\" magazine"), ("before_action", "Johnson Publishing Company", "\"The New Ebony Cookbook\""), ("retrieved_after_search", "Jet (magazine)", "chronicling the Civil Rights Movement")]),
 (18, "What media formally covered issues", "repair_candidate", ["question_semantics_vague_media_formally_issues"],
  "Jet is the magazine published by Johnson Publishing, which also published the cookbook; Jet's article explicitly mentions chronicling the Civil Rights Movement. The factual join is supported but the question wording needs correction.", None,
  [("before_action", "Johnson Publishing Company", "published \"Jet\" magazine"), ("before_action", "Johnson Publishing Company", "\"The New Ebony Cookbook\""), ("before_action", "Jet (magazine)", "chronicling the Civil Rights Movement")]),
 (19, "The ringer and 90210", "repair_candidate", ["question_ambiguous_appeared_vs_premiered", "query_copies_partial_context"],
  "The visible 90210 season article already states its September 2011 premiere as a lead-in to Ringer. Search returns Ringer's first air date, but 'both appeared' could mean first series appearance or concurrent broadcast; specify the 2011 90210 season-four premiere and Ringer series premiere.", None,
  [("before_action", "90210 (season 4)", "premiered on September 13, 2011"), ("retrieved_after_search", "Ringer (TV series)", "initially aired on The CW from September 13, 2011")]),
 (20, "The ringer and 90210", "repair_candidate", ["question_ambiguous_appeared_vs_premiered"],
  "2011 is supported for the 90210 fourth-season premiere and the Ringer series premiere. The question omits the season and event distinction, so the answer pair should be rewritten before reuse.", None,
  [("before_action", "90210 (season 4)", "premiered on September 13, 2011"), ("before_action", "Ringer (TV series)", "initially aired on The CW from September 13, 2011")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-047.jsonl", cohort="v122_internal_candidates")
