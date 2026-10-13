"""Primary examination of v122 internal candidate states 91–100."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (91, "Which band's debut album", "repair_candidate", ["unnecessary_search_answer_already_visible", "question_typo_hockey_played"],
  "The initial Gretzky album snippet already names Electro Quarterstaff and Wayne Gretzky. The search repeats its title, so it adds nickname confirmation but no missing fact needed to identify the band; the question also says 'hockey played' instead of 'hockey player'.",
  "Correct the typo and either answer from the initial state or write a genuinely missing-relation question before reconsidering a search example.",
  [("before_action", "Gretzky (album)", "debut full-length album by Canadian instrumental progressive metal band Electro Quarterstaff"), ("retrieved_after_search", "Wayne Gretzky", "Nicknamed \"The Great One\"")]),
 (92, "Which band's debut album", "repair_candidate", ["paired_unnecessary_search", "question_typo_hockey_played"],
  "Electro Quarterstaff is supported by the album article even before search. The paired search is redundant for this answer and the wording has a typo, so this is not a clean two-step training pair.",
  "Use a corrected direct-answer item only after separate replay and review, or exclude it from paired search training.",
  [("before_action", "Gretzky (album)", "band Electro Quarterstaff"), ("before_action", "Wayne Gretzky", "Nicknamed \"The Great One\"")]),
 (93, "The writer and director of the French film", "retain_candidate", ["query_copies_visible_relation"],
  "The film snippet names Luc Moullet as a Cahiers du cinéma critic. The query copies that visible link and retrieves the magazine article, which supplies the missing foundation year.", None,
  [("before_action", "Brigitte et Brigitte", "film critic Luc Moullet"), ("retrieved_after_search", "Cahiers du cinéma", "French film magazine founded in 1951")]),
 (94, "The writer and director of the French film", "retain_candidate", [],
  "The film article identifies Cahiers du cinéma as the critic's magazine; its article explicitly states 1951. The answer follows the same magazine identity.", None,
  [("before_action", "Brigitte et Brigitte", "Cahiers du cinéma"), ("before_action", "Cahiers du cinéma", "founded in 1951")]),
 (95, "What Indian actress produced Sarvann", "repair_candidate", ["unnecessary_search_answer_already_visible", "query_contains_full_answer"],
  "The initial Sarvann snippet directly states Priyanka Chopra produced the film, and the proposed query repeats her name. The search is unnecessary for the asked answer.",
  "Answer from the initial state or exclude the redundant search pair; do not train search necessity from it.",
  [("before_action", "Sarvann", "produced by Priyanka Chopra")]),
 (96, "What Indian actress produced Sarvann", "repair_candidate", ["paired_unnecessary_search", "answer_supported_before_search"],
  "Priyanka Chopra is correct and visible before search. The added biography only confirms she is an Indian actress; the initial question and source already support the identity, making this a poor paired search example.",
  "Use a direct answer if retained after replay and review.",
  [("before_action", "Sarvann", "produced by Priyanka Chopra"), ("before_action", "Priyanka Chopra", "is an Indian actress")]),
 (97, "What country are Belgravia", "quarantine_pending_repair", ["answer_type_city_for_country", "retrieval_does_not_supply_country", "query_targets_peripheral_landmark"],
  "The question asks for a country, but the paired answer is London, a city. Initial snippets already locate Belgravia and Knightsbridge in London; the query searches a peripheral street/department-store sentence and retrieves no explicit country for both districts. The pair cannot be repaired by merely accepting London as the country.",
  "Exclude the frozen pair. A future country question needs explicit country evidence for both districts and a newly audited answer; a city question would require a separately revised target.",
  [("before_action", "Lowndes Square", "Belgravia, London"), ("before_action", "Sloane Square", "districts of Knightsbridge, Belgravia and Chelsea"), ("retrieved_after_search", "Harvey Nichols", "flagship store in Knightsbridge, London")]),
 (98, "What country are Belgravia", "quarantine_pending_repair", ["answer_type_city_for_country", "unsupported_country_answer"],
  "London answers the city rather than the requested country. The visible evidence establishes both districts' London context, not an explicit country answer in this state; the frozen answer must not be counted as supported for the stated question.",
  "Keep quarantined unless a new question/answer pair is constructed and independently reviewed with explicit country support.",
  [("before_action", "Lowndes Square", "Belgravia, London"), ("before_action", "Sloane Square", "districts of Knightsbridge, Belgravia and Chelsea"), ("before_action", "Harvey Nichols", "Knightsbridge, London")]),
 (99, "Britain's Secret Treasures is", "retain_candidate", ["query_copies_visible_host_relation"],
  "The programme snippet names co-host Bettany Hughes. Searching that visible relation retrieves her biography and the missing specialization, classical history.", None,
  [("before_action", "Britain's Secret Treasures", "hosted by Michael Buerk and Bettany Hughes"), ("retrieved_after_search", "Bettany Hughes", "Her speciality is classical history")]),
 (100, "Britain's Secret Treasures is", "retain_candidate", [],
  "Bettany Hughes is the programme's other host, and her biography directly states classical history as her speciality. The answer is supported.", None,
  [("before_action", "Britain's Secret Treasures", "hosted by Michael Buerk and Bettany Hughes"), ("before_action", "Bettany Hughes", "Her speciality is classical history")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-055.jsonl", cohort="v122_internal_candidates")
