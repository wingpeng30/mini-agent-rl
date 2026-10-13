"""Primary examination of v122 internal candidate states 191–200."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (191, "Chronologie is a studio album", "retain_candidate", ["query_copies_visible_inspiration_book"],
  "Chronologie explicitly names A Brief History of Time as the inspiring book. The search retrieves that book's first publication year, 1988.", None,
  [("before_action", "Chronologie", "inspired by Stephen Hawking's book \"A Brief History of Time\""), ("retrieved_after_search", "A Brief History of Time", "first published in 1988")]),
 (192, "Chronologie is a studio album", "retain_candidate", [],
  "The album is linked to Hawking's book, whose article states a first publication in 1988. The answer is supported.", None,
  [("before_action", "Chronologie", "inspired by Stephen Hawking's book \"A Brief History of Time\""), ("before_action", "A Brief History of Time", "first published in 1988")]),
 (193, "What is the relationship of Anastasia Nikolaevna", "quarantine_pending_repair", ["generation_error_granddaughter_vs_great_granddaughter"],
  "The initial Alexandra Feodorovna article calls Alexandra a granddaughter of Queen Victoria. The retrieved Anastasia article says Anastasia was Alexandra's daughter. Therefore Anastasia is Victoria's great-granddaughter, not her granddaughter as in the paired target. The search provides the missing generation but the intended answer fails to apply it.",
  "Exclude the frozen pair. A corrected great-granddaughter answer requires a separately bound pair and independent review.",
  [("before_action", "Alexandra Feodorovna (Alix of Hesse)", "granddaughter of Queen Victoria of the United Kingdom"), ("retrieved_after_search", "Grand Duchess Anastasia Nikolaevna of Russia", "youngest daughter of Tsar Nicholas II"), ("retrieved_after_search", "Grand Duchess Anastasia Nikolaevna of Russia", "wife, Tsarina Alexandra Feodorovna")]),
 (194, "What is the relationship of Anastasia Nikolaevna", "quarantine_pending_repair", ["wrong_generation_answer"],
  "The two visible relations are Victoria to granddaughter Alexandra, then Alexandra to daughter Anastasia. The target 'granddaughter' omits one generation; it belongs to Alexandra, not Anastasia.",
  "Keep quarantined; any great-granddaughter correction must be versioned and independently reviewed.",
  [("before_action", "Alexandra Feodorovna (Alix of Hesse)", "granddaughter of Queen Victoria of the United Kingdom"), ("before_action", "Grand Duchess Anastasia Nikolaevna of Russia", "wife, Tsarina Alexandra Feodorovna")]),
 (195, "When was the person who was taught", "retain_candidate", ["query_copies_visible_teacher_student_relation"],
  "Santosh Lal's article names MS Dhoni as the person he taught the helicopter shot. The search follows this relation and retrieves Dhoni's birth date.", None,
  [("before_action", "Santosh Lal", "taught MS Dhoni how to play the helicopter shot"), ("retrieved_after_search", "MS Dhoni", "born 7 July 1981")]),
 (196, "When was the person who was taught", "retain_candidate", [],
  "The taught player is MS Dhoni, and the retrieved biography gives 7 July 1981 as his birth date. The answer is supported.", None,
  [("before_action", "Santosh Lal", "taught MS Dhoni how to play the helicopter shot"), ("before_action", "MS Dhoni", "born 7 July 1981")]),
 (197, "Which of the Seven Wonders", "retain_candidate", ["query_copies_visible_coin_foundation_relation"],
  "The initial coin article says a hemihekte was found in the Temple of Artemis at Ephesus foundation deposit. Searching that grounded relation retrieves the temple article, which confirms it as one of the Seven Wonders.", None,
  [("before_action", "Phanes (coin issuer)", "a hemihekte of the issue was found in the foundation deposit of the Temple of Artemis at Ephesus"), ("retrieved_after_search", "Temple of Artemis", "One of the Seven Wonders of the Ancient World")]),
 (198, "Which of the Seven Wonders", "retain_candidate", [],
  "The coin is placed in the Temple of Artemis foundation deposit, and the temple article identifies it as one of the Seven Wonders. The answer is supported.", None,
  [("before_action", "Phanes (coin issuer)", "foundation deposit of the Temple of Artemis at Ephesus"), ("before_action", "Temple of Artemis", "One of the Seven Wonders of the Ancient World")]),
 (199, "What Maine county is Orr's Island", "repair_candidate", ["query_targets_road_not_missing_county", "county_inferred_via_bridge_in_same_town"],
  "The initial Orr's Island article places it in Harpswell. The query instead targets Maine State Route 24, a road snippet that does not name a county; retrieval happens to add Bailey Island Bridge in Harpswell within Cumberland County. This permits an indirect same-town inference, but the query did not target the missing Harpswell-to-county relation and the answer is not stated for Orr's Island itself.",
  "Search for Harpswell's county from the visible island location, then replay with a direct Harpswell-to-Cumberland source before approval.",
  [("before_action", "Orr's Island", "within the town of Harpswell, Maine"), ("retrieved_after_search", "Bailey Island Bridge", "town of Harpswell within Cumberland County")]),
 (200, "What Maine county is Orr's Island", "repair_candidate", ["paired_indirect_county_support", "paired_query_targets_road"],
  "Cumberland County is supported for a Harpswell bridge, and Orr's Island is in Harpswell, but the relation is inferred through another landmark while the query targeted a road. A direct county source and corrected query are preferable for a clean training pair.",
  "Retain Cumberland County only after Harpswell-to-county retrieval and paired replay.",
  [("before_action", "Orr's Island", "within the town of Harpswell, Maine"), ("before_action", "Bailey Island Bridge", "town of Harpswell within Cumberland County")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-065.jsonl", cohort="v122_internal_candidates")
