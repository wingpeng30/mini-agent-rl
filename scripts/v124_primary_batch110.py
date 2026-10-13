"""Primary examination of v123 internal candidate states 91–100."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (91, "What sport involves both CD Guadalajara", "repair_candidate", ["answer_visible_before_search", "query_repeats_season_title"],
  "The initial CD Guadalajara season excerpt already calls the club's competition Spanish football and lists the Copa del Rey. Its sport answer is thus visible before the search; the query repeats the same season title and the returned cup article does not add a missing sport relation.",
  "Treat as a direct-answer example or revise the question to require a relation absent initially, then replay.",
  [("before_action", "2011–12 CD Guadalajara season", "second tier of Spanish football"), ("before_action", "2011–12 CD Guadalajara season", "Copa del Rey")]),
 (92, "What sport involves both CD Guadalajara", "repair_candidate", ["paired_redundant_search", "answer_supported_before_search"],
  "Football is directly stated in the initial season article that also names CD Guadalajara and Copa del Rey. The paired search adds no needed evidence, so the pair is unsuitable as a search-required state.",
  "Use the answer only as a direct-answer state after export review.",
  [("before_action", "2011–12 CD Guadalajara season", "second tier of Spanish football"), ("before_action", "2011–12 CD Guadalajara season", "Copa del Rey")]),
 (93, "How would you describe both Zayre and Ames", "retain_candidate", ["query_uses_visible_ames_store", "retrieval_adds_zayre_category"],
  "The initial Ames store article calls Ames a chain of discount stores. Searching that visible chain retrieves Zayre's article, which independently calls Zayre a chain of discount stores and documents its later sale to Ames. The common retail category is grounded.", None,
  [("before_action", "Ames (store)", "chain of discount stores"), ("retrieved_after_search", "Zayre", "chain of discount stores")]),
 (94, "How would you describe both Zayre and Ames", "retain_candidate", [],
  "Both Ames and Zayre are independently described as chains of discount stores. The answer 'discount stores' is an accurate common category, though a polished target may say 'discount-store chains'.", None,
  [("before_action", "Ames (store)", "chain of discount stores"), ("before_action", "Zayre", "chain of discount stores")]),
 (95, "The Wheeler Family Farmstead is a historic farm complex", "retain_candidate", ["query_uses_visible_town", "retrieval_adds_county"],
  "The initial Wheeler Farmstead article places the property in Great Barrington, Massachusetts. Searching that visible town link retrieves its article, which states Great Barrington lies in Berkshire County.", None,
  [("before_action", "Wheeler Family Farmstead", "in Great Barrington, Massachusetts"), ("retrieved_after_search", "Great Barrington, Massachusetts", "town in Berkshire County")]),
 (96, "The Wheeler Family Farmstead is a historic farm complex", "retain_candidate", [],
  "The farmstead is in Great Barrington and the town is in Berkshire County. The county answer is supported by action-time documents.", None,
  [("before_action", "Wheeler Family Farmstead", "in Great Barrington, Massachusetts"), ("before_action", "Great Barrington, Massachusetts", "town in Berkshire County")]),
 (97, "Jabalpur Airport and Indira Gandhi International Airport", "retain_candidate", ["query_uses_visible_airline", "retrieval_adds_second_airport"],
  "The initial Zoom Air article ties that airline to Indira Gandhi International Airport in Delhi, India. Searching the visible airline retrieves Jabalpur Airport, which is also explicitly located in India. The query is indirect but grounded in action-time evidence.", None,
  [("before_action", "Zoom Air", "based at Indira Gandhi International Airport in Delhi, India"), ("retrieved_after_search", "Jabalpur Airport", "Madhya Pradesh, India")]),
 (98, "Jabalpur Airport and Indira Gandhi International Airport", "retain_candidate", [],
  "The initial Zoom Air source places Indira Gandhi International Airport in India, and the retrieved Jabalpur Airport article also places that airport in India. Both required locations are supported.", None,
  [("before_action", "Zoom Air", "Indira Gandhi International Airport in Delhi, India"), ("before_action", "Jabalpur Airport", "Madhya Pradesh, India")]),
 (99, "Did Isaac Brock and Dee Snider both play banjo", "quarantine_pending_repair", ["negative_from_missing_mention", "snider_instrument_unverified"],
  "The retrieved Isaac Brock article explicitly calls him a banjoist. Dee Snider's excerpt lists singer-songwriter and other roles but neither affirms nor denies banjo playing. A no answer cannot be derived from the absence of a banjo mention; Mike Snider is a different musician and cannot substitute for Dee.",
  "Obtain a direct reliable source about Dee Snider's banjo ability or replace the task with a positively supported comparison; do not infer negation from silence.",
  [("before_action", "Dee Snider", "American singer-songwriter"), ("retrieved_after_search", "Isaac Brock (musician)", "banjoist"), ("retrieved_after_search", "Mike Snider (musician)", "banjo player")]),
 (100, "Did Isaac Brock and Dee Snider both play banjo", "quarantine_pending_repair", ["paired_unsupported_negative_answer"],
  "The evidence proves Brock is a banjoist but supplies no positive or negative banjo claim for Dee Snider. The answer 'no' is unsupported as a conjunction negation on the available input.",
  "Keep the original quarantined until Dee-specific evidence establishes the missing predicate.",
  [("before_action", "Dee Snider", "American singer-songwriter"), ("before_action", "Isaac Brock (musician)", "banjoist")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-110.jsonl", cohort="v123_internal_candidates")
