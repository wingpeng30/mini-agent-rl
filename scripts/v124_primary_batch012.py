"""Primary case examination of original training rows 111–120."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (111, "The administrative center", "retain_candidate", ["administrative_center_outside_region_boundary"],
  "Masaliyev's biography gives Bishkek as place of death, and the Bishkek article names it as administrative center of Chuy Region. The article also says the city is administratively outside the region; that does not negate the administrative-center relationship asked here.", None,
  [("before_action", "Absamat Masaliyev", "July 31, 2004, Bishkek"), ("before_action", "Bishkek", "Bishkek is also the administrative center of the Chuy Region")]),
 (112, "What is the organization", "repair_candidate", ["created_by_figer_not_established_by_ownership", "question_typo_midfielderm"],
  "The Figer article names MJF Publicidade e Promoções S/C Ltda as his organisation and says he co-owns it with Wagner Ribeiro; the player article verifies Zé Roberto's Palmeiras role. Neither document establishes that Figer created the firm. This confirms the same premise problem found in search-state row 83.",
  "Use the row-83 proposed question about Figer's organisation, removing the unsupported creation claim. Repair both same-question states together, replay their inputs, and seek independent review.",
  [("before_action", "Juan Figer", "His organisation, MJF Publicidade e Promoções S/C Ltda"), ("before_action", "Juan Figer", "He is co-owner of the firm with Wagner Ribeiro."), ("before_action", "Zé Roberto", "plays for Palmeiras as a left wing back or as a midfielder")]),
 (113, "The InterContinental Bucharest", "retain_candidate", ["broad_other_brands_question"],
  "The Bucharest hotel article identifies InterContinental Hotels Group as operator. A query copying that visible operator relation retrieves the group's article listing its other hotel brands. The question's 'related to' and plurality need answer-state checking for whether it asks the group or a list.", None,
  [("before_action", "InterContinental Bucharest", "operated by InterContinental Hotels Group"), ("retrieved_after_search", "InterContinental Hotels Group", "Its brands include Candlewood Suites, Crowne Plaza, EVEN Hotels, Holiday Inn")]),
 (114, "Dyson is a well-known", "retain_candidate", ["concept_vs_product_invention_wording"],
  "The bladeless-fan article says the first concept was created by Toshiba in 1981. The answer names that company. The source supports origin of the concept rather than any claim that a marketed Dyson design was invented by Toshiba.", None,
  [("before_action", "Bladeless fan", "The first concept was created by Toshiba in 1981.")]),
 (115, "James are an English", "retain_candidate", ["question_grammar"],
  "The Laid song article explicitly calls it the title song of James's 1993 album Laid. The James article verifies the band and hit-single relation. The answer year is supported.", None,
  [("before_action", "Laid (song)", "title song from 1993 album \"Laid\" by Manchester alternative rock band James"), ("before_action", "James (band)", "Their best-known singles include \"Come Home\", \"Sit Down\", \"She's a Star\" and \"Laid\"")]),
 (116, "In which year was this Chilean", "retain_candidate", ["question_repeats_boot_and_player_clue", "club_affiliation_source_snapshot"],
  "The initial boot article lists Arturo Vidal among endorsers but not his birth year or Chilean Bayern role. The question-grounded search retrieves Vidal's biography with 1987 and the missing role relation. This pairs with answer row 102.", None,
  [("before_action", "Nike Mercurial Vapor", "Luka Modrić, Arturo Vidal, Douglas Costa"), ("retrieved_after_search", "Arturo Vidal", "born 22 May 1987) is a Chilean professional footballer")]),
 (117, "The day Percy Jones", "repair_candidate", ["query_targets_holiday_before_person_birth_known", "accidental_retrieval_risk"],
  "Only the Boxing Day definition is visible before action; no Percy Jones birth date or identity is present. Searching 'Boxing Day' repeats the holiday candidate rather than targeting the missing person's birth date. It happens to retrieve a Percy Jones boxer article that says he was born on Boxing Day. This success is insufficient to endorse the query as a training target.",
  "Proposed query: Percy Jones birth date. It uses only the question's name and missing relation; replay fixed retrieval and resolve name ambiguity before using any revised state.",
  [("before_action", "Boxing Day", "Boxing Day is on 26 December"), ("retrieved_after_search", "Percy Jones (boxer)", "born on Boxing Day")]),
 (118, "Andover Maine", "retain_candidate", [],
  "The Andover town article names the Andover Earth Station. The station article explicitly gives its Maine location and AT&T construction in 1961. The target answer is fully supported.", None,
  [("before_action", "Andover, Maine", "home to the Andover Earth Station"), ("before_action", "Andover Earth Station", "built by AT&T in 1961")]),
 (119, "How many active duty", "repair_candidate", ["unanchored_current_personnel_count", "question_typo_personal"],
  "The Holland article identifies the US Navy as the commissioning service. Its Navy article reports 322,421 active-duty personnel with an explicit 2017 snapshot, while the question says 'currently' without an as-of date. The numerical answer cannot be treated as current without that anchor.",
  "Proposed question: How many active-duty personnel did the United States Navy, which commissioned USS Holland (SS-1), have according to the cited 2017 snapshot? Retain 322,421 only after paired-state, source-scope, token and independent review.",
  [("before_action", "USS Holland (SS-1)", "the United States Navy's first modern commissioned submarine"), ("before_action", "United States Navy", "322,421 personnel on active duty"), ("before_action", "United States Navy", "as of 2017")]),
 (120, "Paulo Coelho", "retain_candidate", [],
  "Coelho's article identifies The Alchemist as his widely translated novel. Its novel article states it was first published in 1988, exactly answering the question.", None,
  [("before_action", "Paulo Coelho", "widely translated novel \"The Alchemist\""), ("before_action", "The Alchemist (novel)", "first published in 1988")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-012.jsonl")
