"""Primary case examination of original training rows 51–60."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (51, "What number president", "retain_candidate", ["retrieved_snapshot_calls_president_current"],
  "The book names Trump as Kelly's feud counterpart. The search copies that visible relation and retrieves a biography saying he is the 45th president. The question asks for the ordinal, so the retrieved page's time-bound 'current' wording is not required for this answer; do not reuse it as a present-day assertion.", None,
  [("before_action", "Settle for More", "her feud with then Presidential nominee Donald Trump"), ("retrieved_after_search", "Donald Trump", "the 45th and current President of the United States")]),
 (52, "Which unincorporated community", "retain_candidate", [],
  "The Spofford document supplies the northeastern Chesterfield location and lake outlet; the lake document supplies 732 acres and explicitly identifies the village at its outlet. West Chesterfield is a distractor, not a second valid match.", None,
  [("before_action", "Spofford, New Hampshire", "an unincorporated community in the northeastern part of the town of Chesterfield"), ("before_action", "Spofford Lake", "Spofford Lake is a 732 acre water body"), ("before_action", "Spofford Lake", "The village of Spofford is located at the lake's outlet.")]),
 (53, "What was the nickname", "retain_candidate", ["noisy_retrieved_biography_markup"],
  "The bill document already identifies Thatcher but lacks her nickname. The full question and bill title are grounded; the returned Thatcher biography supplies Iron Lady. Its damaged pronunciation/template prefix should be flagged as source noise, though the quoted nickname span is intact.", None,
  [("before_action", "Shops Bill 1986", "Introduced by the government of Margaret Thatcher"), ("retrieved_after_search", "Margaret Thatcher", "dubbed her the \"Iron Lady\", a nickname")]),
 (54, "Who formed first, The Get Up Kids", "retain_candidate", [],
  "The two formation years are explicit in the visible documents: Get Up Kids in 1995 and Moonbabies in 1997. The answer correctly compares them.", None,
  [("before_action", "The Get Up Kids", "Formed in 1995"), ("before_action", "Moonbabies (band)", "Moonbabies is a Swedish duo formed in 1997")]),
 (55, "what does Wardle", "retain_candidate", ["broad_shared_attribute_question"],
  "Both named places are explicitly in England. The shared-country answer is supported, although the open-ended 'have in common' wording can admit other shared properties; keep that wording risk visible for later question-quality review.", None,
  [("before_action", "Wardle, Greater Manchester", "in Greater Manchester, England"), ("before_action", "South Pennines", "in Northern England")]),
 (56, "Who owned the PA-24", "retain_candidate", ["ownership_wording_requires_answer_state_check"],
  "The crash document names pilot Randy Hughes but does not state ownership. The grounded crash description retrieves Patsy Cline's biography, which calls it the private plane of her manager Randy Hughes. This adds relevant possession evidence; legal-title ownership is not separately documented here, so the later answer state must be checked against the question's wording.", None,
  [("before_action", "1963 Camden PA-24 crash", "along with the pilot Randy Hughes"), ("retrieved_after_search", "Patsy Cline", "the private plane of her manager, Randy Hughes")]),
 (57, "When was one of the co founder", "retain_candidate", ["query_repeats_known_sentence"],
  "The company article names both founders but gives neither footballer identity nor birth date. Its copied sentence is grounded and retrieves Mathieu Flamini's footballer biography and 7 March 1984 birth date. Query efficiency is unproven; the actual returned evidence fills both missing relations.", None,
  [("before_action", "GF Biochemicals", "co-founded by Mathieu Flamini and Pasquale Granata"), ("retrieved_after_search", "Mathieu Flamini", "born 7 March 1984) is a French professional footballer")]),
 (58, "Jasminum multipartitum", "retain_candidate", [],
  "The plant article lists Zimbabwe among native countries, and the Zimbabwe article gives the exact landlocked southern-African location between the named rivers. The other listed countries do not undermine this positive two-document identification.", None,
  [("before_action", "Jasminum multipartitum", "native to Mozambique, Zimbabwe, Swaziland and South Africa"), ("before_action", "Zimbabwe", "a landlocked country located in southern Africa, between the Zambezi and Limpopo Rivers")]),
 (59, "What did Willis H. O'Brien", "retain_candidate", [],
  "O'Brien's biography lists King Kong among his work. The King Kong film article supplies Bruce Cabot, Robert Armstrong, and its New York City opening, distinguishing it from the nearby Son of Kong distractor.", None,
  [("before_action", "Willis H. O'Brien", "his work on \"The Lost World\" (1925), \"King Kong\" (1933)"), ("before_action", "King Kong (1933 film)", "It stars Fay Wray, Bruce Cabot and Robert Armstrong, and opened in New York City")]),
 (60, "What is the town on the opposite", "retain_candidate", ["deictic_question_wording"],
  "The population clue identifies Penobscot (1,263 in 2010). The river article places Brooksville on the left bank and Penobscot on the right. The answer follows that specific clue despite the awkward 'this town' phrasing and other right-bank towns.", None,
  [("before_action", "Penobscot, Maine", "The population was 1,263 at the 2010 census."), ("before_action", "Bagaduce River", "forming the border between Brooksville on its left bank and Sedgwick, Penobscot, and Castine on its right")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-006.jsonl")
