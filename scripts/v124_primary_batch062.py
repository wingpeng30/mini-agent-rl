"""Primary examination of v122 internal candidate states 161–170."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (161, "What British crime novelist wrote", "retain_candidate", ["answer_author_visible_before_search", "search_verifies_novelist_qualification"],
  "The book snippet names Martin Edwards as author. The query includes this visible name, and retrieval verifies that he is a British crime novelist, the missing qualification in the question.", None,
  [("before_action", "The Golden Age of Murder", "book written by Martin Edwards"), ("retrieved_after_search", "Martin Edwards (author)", "British crime novelist")]),
 (162, "What British crime novelist wrote", "retain_candidate", [],
  "Martin Edwards is named as the book's author, and the retrieved biography identifies him as a British crime novelist. The answer is supported.", None,
  [("before_action", "The Golden Age of Murder", "book written by Martin Edwards"), ("before_action", "Martin Edwards (author)", "British crime novelist")]),
 (163, "When the author of a poem", "repair_candidate", ["question_missing_did_auxiliary"],
  "The poem article identifies the Vivians and author John Ashbery; the search retrieves Ashbery's death date. The intended relation is sound, but 'When the author ... pass away?' lacks the auxiliary verb and is not a clean question.",
  "Ask 'When did the author of Girls on the Run, whose characters include the Vivians, die?' and replay.",
  [("before_action", "Girls on the Run (poem)", "American writer John Ashbery"), ("before_action", "Girls on the Run (poem)", "group of girls known as the Vivians"), ("retrieved_after_search", "John Ashbery", "September 3, 2017")]),
 (164, "When the author of a poem", "repair_candidate", ["paired_question_missing_did_auxiliary"],
  "September 3, 2017 is the documented death date for John Ashbery. The question's missing verb must be corrected, with the pair replayed, before eligibility.",
  "Retain the date only in a grammatical, revalidated pair.",
  [("before_action", "Girls on the Run (poem)", "American writer John Ashbery"), ("before_action", "John Ashbery", "September 3, 2017")]),
 (165, "Who is the producer of the  computer", "repair_candidate", ["malformed_question", "query_targets_unrelated_sequel", "target_game_visible_before_search"],
  "The initial Amazon Trail II article explicitly says it is a spin-off of The Oregon Trail. The question is malformed ('who's spin-off of is the') and the search instead copies the Oregon Trail 3rd Edition release sentence, an unrelated sequel. Retrieval reaches the original Oregon Trail producer, but this is not a grounded search exemplar.",
  "Ask who produced The Oregon Trail, of which Amazon Trail II is a spin-off; search for the original Oregon Trail, replay and review.",
  [("before_action", "Amazon Trail II", "spin-off of \"The Oregon Trail\""), ("before_action", "The Oregon Trail 3rd Edition", "developed by MECC and released on Dec 31, 1997"), ("retrieved_after_search", "The Oregon Trail (video game)", "produced by the Minnesota Educational Computing Consortium (MECC) in 1974")]),
 (166, "Who is the producer of the  computer", "repair_candidate", ["paired_malformed_question", "paired_query_targets_unrelated_sequel"],
  "The retrieved original-game article supports Minnesota Educational Computing Consortium as the producer. The preceding query concerns The Oregon Trail 3rd Edition, and the question itself is syntactically broken; the supported answer does not validate that search action.",
  "Retain the answer only with an original-game query and grammatical paired question after replay.",
  [("before_action", "Amazon Trail II", "spin-off of \"The Oregon Trail\""), ("before_action", "The Oregon Trail (video game)", "produced by the Minnesota Educational Computing Consortium (MECC) in 1974")]),
 (167, "Painkiller and Sea Breeze", "retain_candidate", ["query_uses_visible_cocktail_class"],
  "The initial Sea Breeze article explicitly calls it a cocktail. The search copies its IBA classification and retrieves a separate Painkiller article, which also explicitly calls it a cocktail. This verifies both named drinks.", None,
  [("before_action", "Sea Breeze (cocktail)", "A Sea Breeze is a cocktail"), ("retrieved_after_search", "Painkiller (cocktail)", "A Painkiller is a rum cocktail")]),
 (168, "Painkiller and Sea Breeze", "retain_candidate", [],
  "The visible Sea Breeze and Painkiller articles both identify their subject as a cocktail. The common category is supported.", None,
  [("before_action", "Sea Breeze (cocktail)", "A Sea Breeze is a cocktail"), ("before_action", "Painkiller (cocktail)", "A Painkiller is a rum cocktail")]),
 (169, "Marvin Camel was born", "retain_candidate", ["query_uses_birthplace_not_fight_location"],
  "The initial Camel biography gives Ronan as birthplace and Missoula only as where he fought. The query includes both but retrieval reaches Ronan's city article, which gives Lake County. The final chain correctly uses birthplace rather than fight location.", None,
  [("before_action", "Marvin Camel", "born in Ronan and fought out of Missoula"), ("retrieved_after_search", "Ronan, Montana", "city in Lake County, Montana")]),
 (170, "Marvin Camel was born", "retain_candidate", [],
  "Camel was born in Ronan, and the Ronan article places the city in Lake County, Montana. The answer follows the birthplace relation.", None,
  [("before_action", "Marvin Camel", "born in Ronan and fought out of Missoula"), ("before_action", "Ronan, Montana", "city in Lake County, Montana")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-062.jsonl", cohort="v122_internal_candidates")
