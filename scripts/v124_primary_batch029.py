"""Primary case examination of original training rows 281–290."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (281, "Taylor McLam composed music", "retain_candidate", ["query_known_brand_verifies_relation"],
  "The McDonald's founding year is visible before search, but McLam's connection is not. Querying the candidate brand retrieves McLam's advertisement-client list, including McDonald's. Both needed facts are then visible.", None,
  [("before_action", "McDonald's", "founded in 1940"), ("retrieved_after_search", "Taylor McLam", "brands including Verizon, Chrysler, Mitsubishi, Cotton, Visa, Miller Lite, Sears, Lego, McDonald's and Coca-Cola")]),
 (282, "Taylor McLam composed music", "retain_candidate", ["long_query_copies_brand_list", "duplicate_training_action_group"],
  "McLam's client list is visible before search but brand founding dates are not. The long copied-list query retrieves McDonald's 1940 founding paragraph. This is a valid observed join, though the query carries many irrelevant brands and appears in three copies.", None,
  [("before_action", "Taylor McLam", "McDonald's and Coca-Cola"), ("retrieved_after_search", "McDonald's", "founded in 1940")]),
 (283, "Taylor McLam composed music", "repair_candidate", ["duplicate_training_action", "long_query_copies_brand_list"],
  "Same question, copied-list query and McDonald's 1940 retrieval as row 282, with the same target facts. A duplicate state should not count as a second example.",
  "Keep frozen row, deduplicate after checking paired answer and the reverse-evidence path at row 281.",
  [("before_action", "Taylor McLam", "McDonald's and Coca-Cola"), ("retrieved_after_search", "McDonald's", "founded in 1940")]),
 (284, "Taylor McLam composed music", "repair_candidate", ["duplicate_training_action", "long_query_copies_brand_list"],
  "Third copy of the same McLam list search and McDonald's founding result. Removing one unrelated distractor does not make a new target.",
  "Preserve original input but count at most one justified list-to-date search after pair replay.",
  [("before_action", "Taylor McLam", "McDonald's and Coca-Cola"), ("retrieved_after_search", "McDonald's", "founded in 1940")]),
 (285, "The World War II hero whom the USS Elrod", "repair_candidate", ["graduated_claim_not_supported_by_attended", "duplicate_training_action_group"],
  "The frigate article identifies Henry T. Elrod. The search retrieves his biography saying he attended Georgia and Yale, not that he graduated from either. This repeats the false question premise already found in answer row 68 and search row 183.",
  "Apply one question-level change from 'graduated from' to 'attended', replay every related state and deduplicate the identical ship-name searches before independent review.",
  [("before_action", "USS Elrod (FFG-55)", "named after Captain Henry T. Elrod"), ("retrieved_after_search", "Henry T. Elrod", "He attended the University of Georgia and Yale University")]),
 (286, "The World War II hero whom the USS Elrod", "repair_candidate", ["graduated_claim_not_supported_by_attended", "duplicate_training_action"],
  "Same ship-name query and Elrod attendance paragraph as row 285. The retrieved source still does not support graduation, and this copy does not add independent supervision.",
  "Use the attended-question repair consistently and retain original for audit.",
  [("before_action", "USS Elrod (FFG-55)", "named after Captain Henry T. Elrod"), ("retrieved_after_search", "Henry T. Elrod", "He attended the University of Georgia and Yale University")]),
 (287, "The World War II hero whom the USS Elrod", "repair_candidate", ["graduated_claim_not_supported_by_attended", "duplicate_training_action"],
  "Third copy of the same Elrod ship query and attendance evidence with a different distractor arrangement. No graduation claim is established.",
  "Keep frozen row, include it in the same-question replay and deduplication decision.",
  [("before_action", "USS Elrod (FFG-55)", "named after Captain Henry T. Elrod"), ("retrieved_after_search", "Henry T. Elrod", "He attended the University of Georgia and Yale University")]),
 (288, "The lead female actress in \"City of Angels\"", "retain_candidate", [],
  "The City of Angels film article stars Meg Ryan alongside Nicolas Cage; Ryan's biography explicitly dates the start of her acting career to 1981.", None,
  [("before_action", "City of Angels (film)", "starring Nicolas Cage and Meg Ryan"), ("before_action", "Meg Ryan", "began her acting career in 1981")]),
 (289, "The lead female actress in \"City of Angels\"", "retain_candidate", ["query_known_actress_verifies_film", "duplicate_training_action_group"],
  "Ryan's 1981 career start is visible before the action, but her City of Angels role is not. A name query retrieves the film article naming her as a star, completing the identity check rather than discovering the already-known year.", None,
  [("before_action", "Meg Ryan", "began her acting career in 1981"), ("retrieved_after_search", "City of Angels (film)", "starring Nicolas Cage and Meg Ryan")]),
 (290, "The lead female actress in \"City of Angels\"", "repair_candidate", ["duplicate_training_action"],
  "Same Ryan-name query, known 1981 year and retrieved City of Angels role as row 289, with irrelevant distractors changed.",
  "Preserve original state but deduplicate this path after answer-state and token replay.",
  [("before_action", "Meg Ryan", "began her acting career in 1981"), ("retrieved_after_search", "City of Angels (film)", "starring Nicolas Cage and Meg Ryan")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-029.jsonl")
