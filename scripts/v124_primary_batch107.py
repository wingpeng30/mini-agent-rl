"""Primary examination of v123 internal candidate states 61–70."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (61, "When was the person who was taught how to play", "retain_candidate", ["query_uses_visible_student", "retrieval_adds_birth_date"],
  "The initial Santosh Lal biography says he taught MS Dhoni the helicopter shot. The query quotes that relation and retrieves Dhoni's biography with a full birth date of 7 July 1981.", None,
  [("before_action", "Santosh Lal", "He taught MS Dhoni how to play the helicopter shot"), ("retrieved_after_search", "MS Dhoni", "born 7 July 1981")]),
 (62, "When was the person who was taught how to play", "retain_candidate", [],
  "MS Dhoni is explicitly the person taught by Santosh Lal in the initial article, and his retrieved biography gives 7 July 1981. The complete date is supported.", None,
  [("before_action", "Santosh Lal", "He taught MS Dhoni how to play the helicopter shot"), ("before_action", "MS Dhoni", "born 7 July 1981")]),
 (63, "Which of the Seven Wonders of the Ancient World", "retain_candidate", ["answer_name_visible_before_search", "retrieval_checks_wonder_membership"],
  "The initial Phanes coin article already names the Temple of Artemis at Ephesus as the hemihekte foundation-deposit site. The search reproduces that passage and retrieves the Temple article, which confirms it was one of the Seven Wonders. This is qualifier verification, not answer discovery.", None,
  [("before_action", "Phanes (coin issuer)", "a hemihekte of the issue was found in the foundation deposit of the Temple of Artemis at Ephesus"), ("retrieved_after_search", "Temple of Artemis", "One of the Seven Wonders of the Ancient World")]),
 (64, "Which of the Seven Wonders of the Ancient World", "retain_candidate", [],
  "A hemihekte is reported in the Temple of Artemis foundation deposit, and the Temple article lists it among the Seven Wonders. The answer is supported; its name was already visible before search.", None,
  [("before_action", "Phanes (coin issuer)", "foundation deposit of the Temple of Artemis at Ephesus"), ("before_action", "Temple of Artemis", "One of the Seven Wonders of the Ancient World")]),
 (65, "Burnt by the Sun contains a starring role", "retain_candidate", ["query_uses_visible_female_cast", "retrieval_adds_russian_actress"],
  "The initial Burnt by the Sun article names Nadezhda Mikhalkova among its cast. The query repeats the visible cast list and retrieves her biography, which confirms she is a Russian actress and gives her full name.", None,
  [("before_action", "Burnt by the Sun", "Mikhalkov's daughter Nadezhda Mikhalkova"), ("retrieved_after_search", "Nadezhda Mikhalkova", "Russian actress")]),
 (66, "Burnt by the Sun contains a starring role", "retain_candidate", [],
  "The film article lists Nadezhda Mikhalkova among its stars, and the retrieved biography gives the full name Nadezhda Nikitichna Mikhalkova and Russian actress status. The answer is supported.", None,
  [("before_action", "Burnt by the Sun", "Nadezhda Mikhalkova"), ("before_action", "Nadezhda Mikhalkova", "Nadezhda Nikitichna Mikhalkova"), ("before_action", "Nadezhda Mikhalkova", "Russian actress")]),
 (67, "What is the nationality of the writer of Crook and Ladder", "retain_candidate", ["query_uses_visible_writer", "retrieval_adds_nationality"],
  "The initial Crook and Ladder episode excerpt identifies Bill Odenkirk as writer. Searching the visible writing credit retrieves his biography calling him an American comedy writer. Same-string Crook distractors are not used.", None,
  [("before_action", "Crook and Ladder", "written by Bill Odenkirk"), ("retrieved_after_search", "Bill Odenkirk", "American comedy writer")]),
 (68, "What is the nationality of the writer of Crook and Ladder", "retain_candidate", [],
  "Bill Odenkirk is the named episode writer, and his retrieved biography identifies him as American. The answer follows the action-time evidence.", None,
  [("before_action", "Crook and Ladder", "written by Bill Odenkirk"), ("before_action", "Bill Odenkirk", "American comedy writer")]),
 (69, "In which year the female co-founder of Monkeystone Games", "repair_candidate", ["cofounder_identity_not_visible_before_search", "query_preselects_stevie_case", "question_grammar"],
  "Stevie Case's birth year is already visible in the initial input, but no initial document links her to Monkeystone Games. The query selects her name before that co-founder relation appears; retrieval later establishes it, which cannot retrospectively ground the choice. The question also omits 'was'.",
  "Search Monkeystone Games founders first, then use a supported female co-founder identity to answer the birth year; repair the question and replay.",
  [("before_action", "Stevie Case", "born September 7, 1976"), ("retrieved_after_search", "Monkeystone Games", "founded by John Romero, Tom Hall, Stevie Case, and Brian Moon")]),
 (70, "In which year the female co-founder of Monkeystone Games", "repair_candidate", ["paired_preselected_founder", "answer_supported_after_search"],
  "The retrieved Monkeystone article names Stevie Case as a founder, and her initial biography supports 1976. The answer is supported after search, but the search action preselected Case without an action-time founder link, and the question is ungrammatical.",
  "Keep 1976 only after a founders-first retrieval replay and independently reviewed wording.",
  [("before_action", "Stevie Case", "born September 7, 1976"), ("before_action", "Monkeystone Games", "founded by John Romero, Tom Hall, Stevie Case, and Brian Moon")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-107.jsonl", cohort="v123_internal_candidates")
