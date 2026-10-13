"""Primary case examination of original training rows 101–110."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (101, "Dawn Penn (born", "retain_candidate", ["question_duplicate_word"],
  "The Penn and album articles both give 1994 for the debut No, No, No; the song article also links the single to that album. The repeated 'her' is an editing defect, not an answer mismatch.", None,
  [("before_action", "No, No, No (Dawn Penn album)", "the debut album by a Jamaican reggae singer Dawn Penn, released in 1994"), ("before_action", "You Don't Love Me (No, No, No)", "from her debut studio album, \"No, No, No\" (1994)")]),
 (102, "In which year was this Chilean", "retain_candidate", ["club_affiliation_source_snapshot"],
  "The Vidal biography supplies Chilean midfielder, Bayern Munich role and 1987 birth year; the boot article includes Arturo Vidal among endorsers. The past club role is source-snapshot context and should not be presented as current in future reuse.", None,
  [("before_action", "Arturo Vidal", "born 22 May 1987) is a Chilean professional footballer who plays as a midfielder for Bundesliga club Bayern Munich"), ("before_action", "Nike Mercurial Vapor", "Luka Modrić, Arturo Vidal, Douglas Costa")]),
 (103, "Which actor, singer", "retain_candidate", [],
  "The song article credits Jack Cassidy as its introducer and Harold Rome for music and lyrics. Cassidy's biography confirms actor, singer and director. The stored answer names the supported person.", None,
  [("before_action", "Wish You Were Here (1952 song)", "music and lyrics by Harold Rome"), ("before_action", "Wish You Were Here (1952 song)", "introduced in the show and on the RCA Victor cast album by Jack Cassidy"), ("before_action", "Jack Cassidy", "was an American actor, singer and director")]),
 (104, "Where does the college", "retain_candidate", [],
  "Abercrombie's biography identifies Berklee College of Music. The college article explicitly locates its additional campus in Valencia, Spain, supporting the location answer.", None,
  [("before_action", "John Abercrombie (guitarist)", "studied at Berklee College of Music"), ("before_action", "Berklee College of Music", "operated a campus in Valencia, Spain")]),
 (105, "Dave Franco had a role", "retain_candidate", ["query_copies_multiple_candidate_titles"],
  "The pre-action Franco list includes Warm Bodies among many titles but lacks the film description. Copying that visible list retrieves Warm Bodies, whose article supplies the 2013 paranormal romantic film and novel adaptation. Selection quality may be weak, but the observed search fills the target relation without a hidden answer query.", None,
  [("before_action", "Dave Franco", "\"Warm Bodies\", \"Neighbors\", \"Nerve\""), ("retrieved_after_search", "Warm Bodies (film)", "a 2013 American paranormal romantic zombie comedy film based on Isaac Marion's novel")]),
 (106, "In what city is Cadillac", "retain_candidate", ["indirect_brand_counterpart_lookup"],
  "Navigator's article establishes the Ford Expedition counterpart. Searching its visible name retrieves the Escalade article, which explicitly identifies Cadillac's response and Arlington, Texas assembly. The city answer awaits its paired answer-state check.", None,
  [("before_action", "Lincoln Navigator", "the Lincoln counterpart of the Ford Expedition"), ("retrieved_after_search", "Cadillac Escalade", "introduced for the 1999 model year in response to German and Japanese competitors and to Ford's 1998 release of the Lincoln Navigator"), ("retrieved_after_search", "Cadillac Escalade", "built in Arlington, Texas")]),
 (107, "What not-for-profit teaching", "repair_candidate", ["time_sensitive_rank_without_date"],
  "The hospital is explicitly in Silver Spring and its nonprofit teaching role is supported. Silver Spring's fourth-most-populous rank is tied to 2013 estimates in the source, but the question presents that rank without an as-of date.",
  "Proposed question: What not-for-profit teaching hospital is located in the Maryland place ranked fourth most populous by the cited 2013 Census Bureau estimates? Preserve the hospital answer, then check paired states, replay tokens and independently review.",
  [("before_action", "Holy Cross Hospital (Silver Spring)", "not-for-profit teaching hospital"), ("before_action", "Silver Spring, Maryland", "according to 2013 estimates by the United States Census Bureau, making it the fourth most populous place in Maryland")]),
 (108, "Are Euptelea and Richea", "repair_candidate", ["region_granularity_undefined"],
  "The articles place Richea in Tasmania and southeastern mainland Australia, and Euptelea from Assam through East Asia. A no answer is justified at the continent level, but 'same region of the world' leaves the scale undefined and can invite inconsistent labels.",
  "Proposed question: Are Euptelea and Richea native to the same continent? Keep 'no' only after checking paired states and independently validating the revised wording.",
  [("before_action", "Richea", "endemic to Tasmania and the other two are endemic to the south-east of the Australian mainland"), ("before_action", "Euptelea", "found from Assam east through China to Japan")]),
 (109, "In which area of Northern Ireland", "repair_candidate", ["query_targets_irrelevant_rank_not_missing_reserve_location"],
  "Before searching, Slieve Donard and the Mourne Mountains already establish County Down. The query copies the mountain's island-wide height rank, which does not target the missing reserve-location relation. It happens to retrieve Murlough Nature Reserve and County Down, but that success does not validate this query as a training target.",
  "Proposed query: Murlough Nature Reserve County Down. Both names are available from the question and pre-action evidence; replay the fixed corpus and token budget, then review the changed action independently.",
  [("before_action", "Slieve Donard", "an 850 m mountain in County Down, Northern Ireland"), ("before_action", "Slieve Donard", "It is also the 19th highest peak on the island of Ireland"), ("retrieved_after_search", "Murlough Nature Reserve", "lies on the coast of County Down in Northern Ireland")]),
 (110, "Who was born first", "retain_candidate", ["same_surname_son_distractor"],
  "Martin Bormann's 1900 birth year is visible, while Klopfer is absent. The question-grounded search retrieves Klopfer's 1905 date, making the comparison possible. The Martin Adolf Bormann result is a distinct son and must not be substituted.", None,
  [("before_action", "Martin Bormann", "17 June 1900"), ("retrieved_after_search", "Gerhard Klopfer", "18 February 1905")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-011.jsonl")
