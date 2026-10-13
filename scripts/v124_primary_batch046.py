"""Primary examination of v122 internal candidate states 1–10."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (1, "The breed of dog in the movie Wasao", "retain_candidate", ["query_copies_visible_context"],
  "The film snippet identifies its dog as Akita but does not give breed origin. Copying the film sentence retrieves the Akita article, which explicitly locates the breed's origin in Japan.", None,
  [("before_action", "Wasao", "abandoned Akita Dog"), ("retrieved_after_search", "Akita (dog)", "originating from the mountainous northern regions of Japan")]),
 (2, "The breed of dog in the movie Wasao", "retain_candidate", [],
  "Wasao identifies an Akita dog, and the breed article states Japanese origin. The answer Japan is supported at answer time.", None,
  [("before_action", "Wasao", "abandoned Akita Dog"), ("before_action", "Akita (dog)", "originating from the mountainous northern regions of Japan")]),
 (3, "St Michael's Church is", "retain_candidate", ["query_copies_visible_location", "same_name_church_distractors"],
  "The initial church article places this St Michael's in Aughton. Searching its full location retrieves the Aughton article, which specifies the two towns; other St Michael's churches are separate distractors.", None,
  [("before_action", "St Michael's Church, Aughton", "Church Lane, Aughton, Lancashire"), ("retrieved_after_search", "Aughton, Lancashire", "between Ormskirk and Maghull")]),
 (4, "St Michael's Church is", "retain_candidate", ["same_name_church_distractors"],
  "The Aughton church and Aughton parish articles form the same-location join; the parish lies between Ormskirk and Maghull. The answer avoids the Baddiley church distractor.", None,
  [("before_action", "St Michael's Church, Aughton", "Church Lane, Aughton, Lancashire"), ("before_action", "Aughton, Lancashire", "between Ormskirk and Maghull")]),
 (5, "The only athlete to compete for Montserrat", "quarantine_pending_repair", ["same_name_entity_collision", "retrieved_biography_is_actor_not_athlete"],
  "The Games article names high jumper Gavin Lee. The returned 'Gavin Lee' biography instead identifies an English musical-theatre actor, with no evidence linking him to Montserrat athletics. A name match alone cannot supply the athlete's birth year.", None,
  [("before_action", "Montserrat at the 2002 Commonwealth Games", "Gavin Lee who competed in the Men's High Jump"), ("retrieved_after_search", "Gavin Lee", "is an English actor who has appeared on the stage in musical theatre")]),
 (6, "The only athlete to compete for Montserrat", "quarantine_pending_repair", ["same_name_entity_collision", "answer_birth_year_unsupported_for_athlete"],
  "The 1971 date belongs to a musical-theatre actor named Gavin Lee. No visible document establishes that the actor is the Montserrat high jumper, so the answer has an unsupported entity join and must not enter a qualified set.", None,
  [("before_action", "Montserrat at the 2002 Commonwealth Games", "Gavin Lee who competed in the Men's High Jump"), ("before_action", "Gavin Lee", "is an English actor who has appeared on the stage in musical theatre")]),
 (7, "The 1968 FA Charity Shield", "repair_candidate", ["query_misses_required_venue_relation", "incidental_retrieval_success"],
  "The match article says the venue was Maine Road. The query copies only 'Manchester City won 6–1', not the missing venue or country relation; retrieval nevertheless returns Maine Road with its English location. The action is a poor search exemplar despite a supported answer.", None,
  [("before_action", "1968 FA Charity Shield", "played at City's home ground, Maine Road"), ("retrieved_after_search", "Maine Road", "Moss Side, Manchester, England")]),
 (8, "The 1968 FA Charity Shield", "repair_candidate", ["paired_search_not_targeted", "answer_supported_at_action_time"],
  "The answer England is supported by the match venue and Maine Road article, but the paired search omitted that venue and reached it incidentally. Keep the answer evidence distinct from search quality; repair the pair before reuse.", None,
  [("before_action", "1968 FA Charity Shield", "played at City's home ground, Maine Road"), ("before_action", "Maine Road", "Moss Side, Manchester, England")]),
 (9, "Empress Bianca is", "retain_candidate", ["query_copies_visible_context"],
  "The book article identifies Lady Colin Campbell as author. Searching that sentence retrieves her biography with the full birth date.", None,
  [("before_action", "Empress Bianca", "first novel by Lady Colin Campbell"), ("retrieved_after_search", "Lady Colin Campbell", "born 17 August 1949")]),
 (10, "Empress Bianca is", "retain_candidate", ["day_wording_full_date_answer"],
  "The book names Lady Colin Campbell; her biography gives 17 August 1949. The full-date answer covers the question's 'which day' wording.", None,
  [("before_action", "Empress Bianca", "first novel by Lady Colin Campbell"), ("before_action", "Lady Colin Campbell", "born 17 August 1949")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-046.jsonl", cohort="v122_internal_candidates")
