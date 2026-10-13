"""Primary examination of exposed historical internal states 21–30."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (21, " Lady Augusta Gordon was", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "The Lady Augusta article names William IV as her father without his birth date. The copied sentence retrieves William IV's biography with the date.", None,
  [("before_action", "Lady Augusta Gordon", "fourth illegitimate daughter of William IV"), ("retrieved_after_search", "William IV of the United Kingdom", "21 August 1765")]),
 (22, " Lady Augusta Gordon was", "retain_candidate", ["exposed_development_set"],
  "Lady Augusta's father is William IV, and his biography states 21 August 1765 as his birth date. The answer is supported.", None,
  [("before_action", "Lady Augusta Gordon", "fourth illegitimate daughter of William IV"), ("before_action", "William IV of the United Kingdom", "21 August 1765")]),
 (23, "Mike Swick trains at", "retain_candidate", ["search_verifies_missing_person_gym_relation", "exposed_development_set"],
  "The initial gym article already states San Jose, but it does not connect Mike Swick to this gym. Searching its title retrieves Swick's article, which supplies the missing relation.", None,
  [("before_action", "American Kickboxing Academy", "Based in San Jose, California"), ("retrieved_after_search", "Mike Swick", "trains at American Kickboxing Academy")]),
 (24, "Mike Swick trains at", "retain_candidate", ["exposed_development_set"],
  "Swick's article identifies American Kickboxing Academy as his gym; its article places it in San Jose. The answer does not confuse the San Diego distractor.", None,
  [("before_action", "Mike Swick", "trains at American Kickboxing Academy"), ("before_action", "American Kickboxing Academy", "Based in San Jose, California")]),
 (25, "What company did the owner", "repair_candidate", ["owner_temporal_ambiguity", "query_targets_former_owner", "exposed_development_set"],
  "The question says 'owner' without a time. The source calls Paul Getty a former resident and Mark Getty the current resident; the query copies the Paul sentence, while retrieval happens to find Mark, the co-founder. The current-owner interpretation is plausible but the query teaches a misleading entity choice.", None,
  [("before_action", "Wormsley Park", "former home of the philanthropist Sir Paul Getty"), ("before_action", "Wormsley Park", "now the home of Mark Getty"), ("retrieved_after_search", "Mark Getty", "co-founder of Getty Images")]),
 (26, "What company did the owner", "repair_candidate", ["owner_temporal_ambiguity", "exposed_development_set"],
  "Getty Images is supported if 'owner' refers to Mark Getty, whose current-home link and co-founder role are visible. The question omits that temporal restriction, while the source also foregrounds Paul Getty as the former resident, so the pair needs an explicit current-owner wording before reuse.", None,
  [("before_action", "Wormsley Park", "now the home of Mark Getty"), ("before_action", "Mark Getty", "co-founder of Getty Images")]),
 (27, "In what year was Bradley Orr's", "retain_candidate", ["query_truncates_visible_relation", "exposed_development_set"],
  "Bradley Orr's article already names nephew Jon Flanagan but lacks his birth year. The query unnecessarily truncates the visible relation to 'Liverpool F.C.', yet retrieval returns Flanagan's biography and year.", None,
  [("before_action", "Bradley Orr", "uncle of the Liverpool F.C. full back Jon Flanagan"), ("retrieved_after_search", "Jon Flanagan", "born 1 January 1993")]),
 (28, "In what year was Bradley Orr's", "retain_candidate", ["exposed_development_set"],
  "Bradley Orr's nephew is identified as Jon Flanagan; Flanagan's biography gives a 1993 birth year. The answer is supported despite the search query's truncation.", None,
  [("before_action", "Bradley Orr", "full back Jon Flanagan"), ("before_action", "Jon Flanagan", "born 1 January 1993")]),
 (29, "California Tortilla was voted", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "The California Tortilla article identifies Washingtonian as the voting magazine but lacks its founding year. The copied sentence retrieves the magazine article, which states 1965.", None,
  [("before_action", "California Tortilla", "readers of \"Washingtonian\" magazine"), ("retrieved_after_search", "Washingtonian (magazine)", "founded in 1965")]),
 (30, "California Tortilla was voted", "retain_candidate", ["exposed_development_set"],
  "California Tortilla's article names Washingtonian as the relevant magazine; that magazine's article gives 1965. The answer does not use the unrelated Memphis Magazine distractor.", None,
  [("before_action", "California Tortilla", "readers of \"Washingtonian\" magazine"), ("before_action", "Washingtonian (magazine)", "founded in 1965")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-038.jsonl", cohort="v117_exposed_internal50")
