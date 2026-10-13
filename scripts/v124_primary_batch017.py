"""Primary case examination of original training rows 161–170."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (161, "How many inhabitants", "quarantine_pending_repair", ["population_without_as_of_date", "corrupted_source_markup"],
  "The Ferretti article locates her studio near Rimini and the Rimini article states 146,606 inhabitants, but the count lacks a date. The Ferretti text also contains a corrupted reference suffix. This is the answer counterpart of quarantined search row 50; a number in the visible source alone does not cure temporal or source-quality defects.",
  "Recover the population count's dated provenance and separately document any source-text cleanup. Rebuild both paired states and independently review; do not silently mutate the original corpus.",
  [("before_action", "Alberta Ferretti", "her studio is in the village of Cattolica, near Rimini, Italy"), ("before_action", "Alberta Ferretti", "<ref name=philosophy/cstop telling lies"), ("before_action", "Rimini", "a city of 146,606 inhabitants")]),
 (162, "The Baker–Fancher party", "retain_candidate", [],
  "The massacre article places the Baker–Fancher victims at Mountain Meadows, a rest stop on the Old Spanish Trail. The trail article connects its New Mexico end to Santa Fe. The answer is a supported two-document join.", None,
  [("before_action", "Baker–Fancher party", "Mountain Meadows, a rest stop on the Old Spanish Trail"), ("before_action", "Old Spanish Trail (trade route)", "(or near) Santa Fe, New Mexico")]),
 (163, "What was Patrick Scheuermann", "retain_candidate", ["question_grammar"],
  "Scheuermann's biography identifies his earlier directorship of John C. Stennis Space Center. The center article places it at the Mississippi–Louisiana border, distinguishing it from his later Marshall role in Alabama.", None,
  [("before_action", "Patrick Scheuermann", "served as the director of the John C. Stennis Space Center"), ("before_action", "John C. Stennis Space Center", "at the Mississippi–Louisiana border")]),
 (164, "Do Jane Campion", "retain_candidate", [],
  "The two biographies explicitly describe Campion as New Zealand and Schroeter as German. The stored no answer follows direct nationality comparison; the father biography is not used in place of Jane's own document.", None,
  [("before_action", "Jane Campion", "is a New Zealand screenwriter"), ("before_action", "Werner Schroeter", "was a German film director")]),
 (165, "What long-running American arbitration", "repair_candidate", ["unanchored_current_broadcast_claim"],
  "The WLAX article's syndicated-programming list includes Judge Judy, and the show article supplies the long-running arbitration-show description. The question's present-tense 'airs' lacks a date; the lineup is a source snapshot and can change.",
  "Proposed question: According to the supplied WLAX syndicated-programming list, which long-running American arbitration-based reality court show was included? Recheck same-question states and independently review the changed prompt.",
  [("before_action", "WLAX", "Syndicated programming on WLAX includes"), ("before_action", "WLAX", "\"Judge Judy\" among others"), ("before_action", "Judge Judy", "a long-running American arbitration-based reality court show")]),
 (166, "Which important religious", "retain_candidate", [],
  "Adiyappa's biography places him as a court musician in Thanjavur in the 18th century. The Thanjavur article identifies the city's religious, art and architectural role. The answer is supported.", None,
  [("before_action", "Pacchimiriam Adiyappa", "court musician at the Maratha kingdom of Thanjavur in the 18th century"), ("before_action", "Thanjavur", "an important center of South Indian religion, art, and architecture")]),
 (167, "Who is the current woman", "quarantine_pending_repair", ["unanchored_current_first_lady"],
  "The documents establish that Nana Konadu formerly held Ghana's First Lady title and name Rebecca Akufo-Addo as 'current' First Lady in an undated snapshot. The question has no as-of date, so the stored person cannot be certified as a time-independent answer. This is the answer counterpart of quarantined search row 1.",
  "Recover source snapshot provenance before a date-anchored question; rebuild both states together and independently review. Do not infer today's officeholder from these records.",
  [("before_action", "Nana Konadu Agyeman Rawlings", "was the First Lady of Ghana from 4 June 1979"), ("before_action", "First Lady of Ghana", "The current First Lady is Rebecca Akufo-Addo.")]),
 (168, "MTX Mototrax", "retain_candidate", [],
  "The game article names motocross as its sport. The motocross article states that the sport evolved from motorcycle trials competitions in the United Kingdom. The answer follows the documented relation.", None,
  [("before_action", "MTX Mototrax", "is a Motocross video game"), ("before_action", "Motocross", "evolved from motorcycle trials competitions held in the United Kingdom")]),
 (169, "Here is My Heart", "retain_candidate", ["query_verifies_missing_film_role"],
  "The initial Carlisle biography supplies American singer identity and the 1991 National Medal of Arts, but not her film role. Querying her visible name retrieves Here is My Heart with Carlisle among its stars, completing the question's film-person link. This pairs with supported answer row 98.", None,
  [("before_action", "Kitty Carlisle", "In 1991, she received the National Medal of Arts"), ("retrieved_after_search", "Here is My Heart", "starring Bing Crosby, Kitty Carlisle, and Roland Young")]),
 (170, "Khet and Quarto", "retain_candidate", [],
  "The Quarto article calls it a board game, and the Khet article calls it an abstract strategy board game. The short answer 'board' denotes their shared game type.", None,
  [("before_action", "Quarto (board game)", "Quarto is a board game for two players"), ("before_action", "Khet (game)", "abstract strategy board game")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-017.jsonl")
