"""Primary case examination of original training rows 91–100."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (91, "Pomponius Grekeinos", "retain_candidate", [],
  "The person article explicitly places the inscription in the Catacombs of Rome; the place article defines those catacombs as underground burial places. A New York cemetery distractor does not satisfy the person link.", None,
  [("before_action", "Pomponius Grekeinos", "commemorated by an inscription in the Catacombs of Rome"), ("before_action", "Catacombs of Rome", "underground burial places under Rome")]),
 (92, "what does Wardle", "retain_candidate", ["broad_shared_attribute_question"],
  "The initial South Pennines article establishes Northern England but lacks Wardle. The question-grounded search retrieves Wardle in Greater Manchester, England and in the South Pennines foothills. This supports the stored shared-country answer, while the broad 'have in common' wording remains a question-quality risk.", None,
  [("before_action", "South Pennines", "in Northern England"), ("retrieved_after_search", "Wardle, Greater Manchester", "in Greater Manchester, England"), ("retrieved_after_search", "Wardle, Greater Manchester", "foothills of the South Pennines")]),
 (93, "Which of the film", "retain_candidate", [],
  "Handel's biography names Noah among his Aronofsky co-written films. The Noah film article supplies the 2014 epic biblical drama and director. The answer is directly supported.", None,
  [("before_action", "Ari Handel", "co-writing the films \"Noah\" and \"The Fountain\""), ("before_action", "Noah (2014 film)", "a 2014 American epic biblical drama film directed by Darren Aronofsky")]),
 (94, "Strangers in Paradise", "retain_candidate", ["awkward_common_category_question"],
  "The Academy Is... is described as an American rock band; Strangers in Paradise as a British alternative rock band. 'Rock band' is a shared category, though the question's 'genre and musician group' phrasing is clumsy.", None,
  [("before_action", "The Academy Is...", "was an American rock band"), ("before_action", "Strangers in Paradise (band)", "is a British alternative rock band")]),
 (95, "Are Yeager Airport", "retain_candidate", [],
  "Yeager Airport is explicitly in West Virginia, United States. Albert J. Ellis Airport is in North Carolina, United States. The similarly named Lindberg Airport is not the target; the yes answer is supported by both target documents.", None,
  [("before_action", "Yeager Airport", "West Virginia, United States"), ("before_action", "Albert J. Ellis Airport", "North Carolina, United States")]),
 (96, "After Hours talk show", "retain_candidate", [],
  "The A. J. After Hours article names gossip columnist A. J. Benza as host, and Benza's biography supplies the June 2, 1962 date. The other gossip columnist in the results lacks this link.", None,
  [("before_action", "A. J. After Hours", "hosted by gossip columnist A. J. Benza"), ("before_action", "A. J. Benza", "born June 2, 1962")]),
 (97, "Who created the first alternative", "retain_candidate", ["question_city_county_scope_wording"],
  "The school article calls Upward Bound the first alternative education program in Otsego County, names creators Mike Newell and Mark Rathbun, and says it first operated in Oneonta. The Oneonta article verifies the city clue. County-first entails no earlier such program in that county's city, but retain the original source scope when reviewing this answer.", None,
  [("before_action", "Upward Bound High School", "the first alternative education program in Otsego County, New York"), ("before_action", "Upward Bound High School", "Created by English teacher Mike Newell and principal Mark Rathbun"), ("before_action", "Upward Bound High School", "first located in the basement of a Unitarian church in Oneonta"), ("before_action", "Oneonta, New York", "Its nickname is \"City of the Hills.\"")]),
 (98, "Here is My Heart", "retain_candidate", ["question_staring_typo"],
  "The film article includes Kitty Carlisle among its stars. Her biography identifies the 1991 National Medal of Arts, directly supporting the stored answer despite the typo in the question.", None,
  [("before_action", "Here is My Heart", "starring Bing Crosby, Kitty Carlisle, and Roland Young"), ("before_action", "Kitty Carlisle", "In 1991, she received the National Medal of Arts")]),
 (99, "Where is the investment", "retain_candidate", [],
  "The Milwaukee article names Techtronic Industries as its brand owner. Searching that visible relation retrieves Techtronic's article, which calls it an investment holding company based in Hong Kong. This fills the requested location evidence.", None,
  [("before_action", "Milwaukee Electric Tool Corporation", "a brand of Techtronic Industries"), ("retrieved_after_search", "Techtronic Industries", "is an investment holding company based in Hong Kong")]),
 (100, "What was the nickname", "retain_candidate", ["noisy_retrieved_biography_markup"],
  "The Shops Bill article identifies Thatcher's government, and the Thatcher article explicitly gives Iron Lady as her nickname. The damaged pronunciation/template prefix does not affect that intact evidence span.", None,
  [("before_action", "Shops Bill 1986", "Introduced by the government of Margaret Thatcher"), ("before_action", "Margaret Thatcher", "dubbed her the \"Iron Lady\", a nickname")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-010.jsonl")
