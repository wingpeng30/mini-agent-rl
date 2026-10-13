"""Primary case examination of original training rows 61–70."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (61, "One type of shop", "repair_candidate", ["answer_drops_over_qualifier"],
  "Butter pie names fish and chip shops as a stockist. The fish-and-chips article states 'over 35,000' in the 1930s, whereas the stored answer is exactly '35,000'. That changes a lower bound into an exact count and should not pass as a precisely supported answer.",
  "Proposed answer: over 35,000. Recheck the paired state, exact-match scoring convention, and independent review after changing the label; retain the original row for provenance.",
  [("before_action", "Butter pie", "stocked by fish and chip shops"), ("before_action", "Fish and chips", "by the 1930s there were over 35,000")]),
 (62, "The Defector is a 2009", "retain_candidate", ["verbose_question"],
  "The initial documents only disambiguate Big Ben and do not name the novel's author. A grounded query containing the novel title and visible Big Ben clue retrieves The Defector (novel), whose first sentence identifies Daniel Silva. The redundant appended Big Ben term does not make the observed search unsupported.", None,
  [("before_action", "Big Ben", "nickname for the Great Bell of the clock"), ("retrieved_after_search", "The Defector (novel)", "The Defector is a 2009 spy novel by Daniel Silva.")]),
 (63, "Are both London LeGrand", "retain_candidate", [],
  "Both nationality and occupation are explicit: LeGrand is an American musician and Locher a Swiss musician. The conjunction in the question is false.", None,
  [("before_action", "London LeGrand", "Is an American musician"), ("before_action", "Michael Locher", "is a Swiss musician")]),
 (64, "The winner of the 1976", "retain_candidate", [],
  "The 1976 Missouri election article names John Danforth as winner. The query repeats this visible winner relation and returns Danforth's biography, which explicitly identifies him as an ordained Episcopal priest; the 2016 and Montana election results are distractors.", None,
  [("before_action", "United States Senate election in Missouri, 1976", "Republican John Danforth won the open seat"), ("retrieved_after_search", "John Danforth", "Danforth is an ordained Episcopal priest.")]),
 (65, "Ohio Brush Creek rises", "repair_candidate", ["answer_is_mound_not_asked_crater"],
  "The question's final target is 'which crater'. The proposed answer Serpent Mound names the mound, whereas the source explicitly calls the crater Serpent Mound crater. The short answer risks substituting a distinct feature even though the right entity is in context.",
  "Proposed answer: Serpent Mound crater. Confirm the intended answer convention and paired state, then independently review the changed label.",
  [("before_action", "Serpent Mound crater", "Serpent Mound crater, also known as the Serpent Mound Disturbance"), ("before_action", "Serpent Mound", "on a plateau of the Serpent Mound crater")]),
 (66, "Which retail outlet", "retain_candidate", ["query_verifies_missing_retailer_link"],
  "MEC and its largest-supplier claim are already visible, but the Coghlan's stockist relation is absent. Querying the visible retailer name retrieves the Coghlan's article with an explicit MEC retailer link. This is useful verification rather than a new answer entity.", None,
  [("before_action", "Mountain Equipment Co-op", "bills itself as Canada's largest supplier of outdoor equipment"), ("retrieved_after_search", "Coghlan's", "Mountain Equipment Co-op and Cabela's in Canada")]),
 (67, "The first suburban store", "quarantine_pending_repair", ["unanchored_current_occupant", "awkward_question_wording"],
  "Fred Meyer is grounded by its 1922 Portland founding. The search finds a Hollywood Theatre article identifying Rite Aid as occupant of the first suburban Fred Meyer site, but the claim is explicitly 'currently' with no snapshot date. Preserve the useful retrieval while withholding this temporally unstable training target.",
  "Recover a verified source snapshot date or rewrite as an explicitly source-dated historical occupant; recheck same-question states and independently review the revision.",
  [("before_action", "Fred Meyer", "founded in 1922 in Portland, Oregon"), ("retrieved_after_search", "Hollywood Theatre (Portland, Oregon)", "first suburban Fred Meyer store, which is currently occupied by Rite Aid")]),
 (68, "The World War II hero", "repair_candidate", ["graduated_claim_not_supported_by_attended"],
  "The ship article identifies Henry T. Elrod. His biography says he attended the University of Georgia and Yale University, not that he graduated from either. Both institution names are supported, but the original graduation premise is not.",
  "Proposed question: Which two universities did the World War II hero for whom USS Elrod was named attend? Keep the two-school answer, update all paired inputs, and independently review the changed prompt.",
  [("before_action", "USS Elrod (FFG-55)", "named after Captain Henry T. Elrod"), ("before_action", "Henry T. Elrod", "He attended the University of Georgia and Yale University")]),
 (69, "\"A Thousand Years\" is", "retain_candidate", ["question_typo_dater", "source_has_missing_album_fields"],
  "The song article links its rerecording to Steve Kazee, and his biography gives October 30, 1975. The 'dater' typo and blank album fields are quality defects but do not obscure the asked person's birth date; any cleanup must be versioned separately.", None,
  [("before_action", "A Thousand Years (Christina Perri song)", "Perri re-recorded the song with vocals from Steve Kazee"), ("before_action", "Steve Kazee", "Steve Kazee (born October 30, 1975)")]),
 (70, "What major metropolitan area", "retain_candidate", ["search_verifies_missing_publisher_link"],
  "The geographic description already identifies Metro Detroit. Searching that visible name retrieves Visible Ink Press, whose article links popular reference publishing and headquarters in Metro Detroit. The added source verifies the other condition in the question.", None,
  [("before_action", "Metro Detroit", "a major metropolitan area in Southeast Michigan"), ("retrieved_after_search", "Visible Ink Press", "a publisher of popular reference works.  Its headquarters are in Canton Charter Township, Michigan in Metro Detroit")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-007.jsonl")
