"""Primary case examination of original training rows 181–190."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (181, "Leckie Mine and Norrie Mine", "retain_candidate", [],
  "Both mine articles explicitly place the sites in Northeastern Ontario, Canada. The shared geographic answer is directly supported.", None,
  [("before_action", "Leckie Mine", "in Northeastern Ontario, Canada"), ("before_action", "Norrie Mine", "in Northeastern Ontario, Canada")]),
 (182, "Adam May", "retain_candidate", ["query_verifies_former_role"],
  "WJZ-TV's channel and CBS ownership are known before the action, but Adam May's connection is not. Querying the visible station name retrieves his biography confirming he formerly anchored and reported there. This verifies the missing relation.", None,
  [("before_action", "WJZ-TV", "virtual and VHF digital channel 13, is a CBS owned-and-operated television station"), ("retrieved_after_search", "Adam May (television reporter)", "previously an anchor and reporter at Baltimore’s CBS owned and operated station, WJZ-TV")]),
 (183, "The World War II hero", "repair_candidate", ["graduated_claim_not_supported_by_attended"],
  "The ship article identifies Henry T. Elrod, and the search of its visible ship name retrieves Elrod's biography naming the two universities he attended. It does not establish that he graduated from them. The search is useful but its prompt shares the false graduation premise already identified at answer row 68.",
  "Revise this and all same-question states to ask which two universities Elrod attended; replay the query and input and independently review the pair.",
  [("before_action", "USS Elrod (FFG-55)", "named after Captain Henry T. Elrod"), ("retrieved_after_search", "Henry T. Elrod", "He attended the University of Georgia and Yale University")]),
 (184, "An Equal Music", "retain_candidate", [],
  "The novel article directly states An Equal Music (1999) is by Vikram Seth. The answer year is explicit.", None,
  [("before_action", "An Equal Music", "An Equal Music (1999) is a novel by Vikram Seth")]),
 (185, "\"The Simpsons\"' fourteenth", "retain_candidate", [],
  "The season article identifies How I Spent My Strummer Vacation as the Scully-run episode but gives no episode air date. Its copied sentence retrieves the episode article with a November 10, 2002 US first-air date. This is a grounded gap-filling search.", None,
  [("before_action", "The Simpsons (season 14)", "The other episode, \"How I Spent My Strummer Vacation\", was run by Mike Scully."), ("retrieved_after_search", "How I Spent My Strummer Vacation", "first aired on the Fox network in the United States on November 10, 2002")]),
 (186, "Who did the box cover", "repair_candidate", ["pen_name_authorship_ambiguous", "question_grammar"],
  "Night Below credits Carl Sargent as writer and Jeff Easley for box cover art. The Carl Sargent article says he used Keith Martin as a pen name for Fighting Fantasy gamebooks; it does not say Night Below was published under that pen name. The question's 'for Keith Martin to make Night Below' formulation confuses person, pen name and credited work.",
  "Proposed question: Who created the box cover art for Night Below, a boxed set written by Carl Sargent? Keep Jeff Easley only after checking all same-question states and independently reviewing the prompt.",
  [("before_action", "Night Below", "was written by Carl Sargent, with box cover art by Jeff Easley"), ("before_action", "Carl Sargent", "using the pen name Keith Martin to write Fighting Fantasy gamebooks")]),
 (187, "The 283rd Rifle Division", "retain_candidate", [],
  "The division article says it fought in the Battle of Berlin. The battle article calls that battle the final major offensive of the European theatre of World War II. The stored category is supported.", None,
  [("before_action", "283rd Rifle Division", "the Battle of Smolensk and the Battle of Berlin"), ("before_action", "Battle of Berlin", "the final major offensive of the European theatre of World War II")]),
 (188, "What actress famously played", "retain_candidate", [],
  "The Miracle Run article lists Mary-Louise Parker as a star, and her biography explicitly credits her as Rita in the Broadway Prelude to a Kiss. The answer is supported.", None,
  [("before_action", "Miracle Run", "starring Mary-Louise Parker"), ("before_action", "Mary-Louise Parker", "stage debut as Rita in a Broadway production")]),
 (189, "Which is Irish Michael Gambon", "repair_candidate", ["irish_born_vs_nationality", "question_grammar"],
  "The Gambon biography calls him an Irish-born English actor, while Viceroy's House is a British-Indian film. The original 'which is Irish' wording may be read as nationality and is not supported for Gambon by this source. The intended distinguishing property is his Irish birthplace, not an established Irish citizenship.",
  "Proposed question: Which of Michael Gambon and the film Viceroy's House is described as Irish-born? Keep the name answer only after checking paired states and independent review.",
  [("before_action", "Michael Gambon", "an Irish-born English actor"), ("before_action", "Viceroy's House (film)", "British-Indian historical drama film")]),
 (190, "Are Royal Blood", "retain_candidate", [],
  "Royal Blood's article identifies an English duo; the Smashing Pumpkins article identifies an American band from Chicago. The statement that both are English is false.", None,
  [("before_action", "Royal Blood (band)", "an English rock duo"), ("before_action", "The Smashing Pumpkins", "an American alternative rock band from Chicago")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-019.jsonl")
