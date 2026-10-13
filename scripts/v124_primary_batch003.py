"""Primary case examination of original training rows 21–30, not independent votes."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (21, "Are Euptelea and Richea", "retain_candidate", ["region_granularity_needs_answer_check"],
  "Richea's Tasmanian/Australian range is visible but Euptelea's range is absent. The query names both genera and asks the requested comparison; it retrieves Euptelea's Assam-to-Japan range. It is a useful grounded search. Any later yes/no answer should make the geographical granularity explicit instead of inferring a universally defined 'region'.", None,
  [("before_action","Richea","Nine of the species are endemic to Tasmania"),("retrieved_after_search","Euptelea","The genus is found from Assam east through China to Japan")]),
 (22, "In which American private research university", "retain_candidate", ["search_verifies_descriptor_not_new_answer_entity"],
  "The prior biography names Clark University but does not establish the requested American/private/research descriptors. The query includes the already-visible school entity; the returned Clark article explicitly verifies all three descriptors. This is constraint verification, not a new answer-name discovery, and should not be mislabeled as a missing-entity bridge.", None,
  [("before_action","Andrew J. Olmsted","educated at St. John's High School in Shrewsbury, MA, and Clark University."),("retrieved_after_search","Clark University","Clark University is an American private research university located in Worcester, Massachusetts")]),
 (23, "What sub-Saharan African mammal", "retain_candidate", [],
  "The prior hippopotamus article establishes sub-Saharan habitat and third-largest land-mammal rank, but not the Kazinga Channel concentration. The query joins the explicit question's location with that visible candidate and retrieves the missing channel evidence. It is justified constraint verification, not merely a repeat of the known species name.", None,
  [("before_action","Hippopotamus","the common hippopotamus is the third-largest type of land mammal"),("retrieved_after_search","Kazinga Channel","one of the world's largest concentration of hippos")]),
 (24, "Which band was formed earlier", "retain_candidate", [],
  "Walk the Moon's 2006 formation is known; the Marilyn Manson band's founding date is not. The full comparison query retrieves the band article's 1989 date. The separate same-name singer biography does not substitute for the band evidence.", None,
  [("before_action","Walk the Moon","Lead singer Nicholas Petricca started the band in 2006"),("retrieved_after_search","Marilyn Manson (band)","formed by singer Marilyn Manson and guitarist Daisy Berkowitz in Fort Lauderdale, Florida, in 1989.")]),
 (25, " An actor who is best known", "retain_candidate", [],
  "Robinson's biography links David Ruffin in The Temptations to his role in Cool Runnings; the film article explicitly provides 1993, sports comedy and director Jon Turteltaub. The three requested relations are supported without using unrelated Ruffin music hits.", None,
  [("before_action","Leon Robinson","David Ruffin in the TV Movie \"The Temptations\" (1998)"),("before_action","Leon Robinson","Derice Bannock in the 1993 film, \"Cool Runnings\""),("before_action","Cool Runnings","Cool Runnings is a 1993 American comedy sports film directed by Jon Turteltaub")]),
 (26, "Who is the the actor who stars", "retain_candidate", [],
  "Getty's Ringside membership is visible but the Slingshot role is not. Searching the named actor is a legitimate entity lookup for the missing film credit; the returned 2005 film article explicitly lists Getty among its stars. A relation keyword is not required for this query to be grounded.", None,
  [("before_action","Balthazar Getty","is an American actor, member of the band Ringside"),("retrieved_after_search","Slingshot (film)","Slingshot is a 2005 American independent crime film"),("retrieved_after_search","Slingshot (film)","It stars David Arquette, Thora Birch, Balthazar Getty and Julianna Margulies.")]),
 (27, "NXT TakeOver: Rival took place", "retain_candidate", [],
  "The specified Rival event identifies Full Sail University, whose article states the institution type and 1979 founding date. The similarly located later NXT event is not needed for the inference.", None,
  [("before_action","NXT TakeOver: Rival","NXT TakeOver took place on February 11, 2015 at NXT's \"home\" arena, Full Sail University"),("before_action","Full Sail University","The school was founded in 1979 in Dayton, Ohio, as Full Sail Recording Workshop.")]),
 (28, "What Pulitzer Prize considered", "retain_candidate", [],
  "The album identifies Wadada Leo Smith as one of its two artists, and his biography explicitly identifies Pulitzer Prize for Music finalist status. The short answer Music supplies the requested award category rather than confusing the album with the nominated work Ten Freedom Summers.", None,
  [("before_action","Wisdom in Time","an album by American jazz trumpeter Wadada Leo Smith and German drummer Günter Baby Sommer"),("before_action","Wadada Leo Smith","He was one of three finalists for the 2013 Pulitzer Prize for Music")]),
 (29, "What band that was a support act", "retain_candidate", ["other_listed_bands_dates_not_established"],
  "The tour explicitly lists All Time Low among first-leg support acts and its biography explicitly dates formation to 2003. This supports All Time Low as a satisfying answer. The packet does not prove no other listed band meets the same date, so do not claim uniqueness beyond the question's non-exhaustive 'what band' formulation or use this case as a uniquely exhaustive comparison benchmark.", None,
  [("before_action","California Tour","the All-American Rejects and All Time Low on certain dates."),("before_action","All Time Low","All Time Low is an American rock band from Towson, Maryland, a suburb of Baltimore, formed in 2003.")]),
 (30, " Tori Kelly voiced the character", "retain_candidate", [],
  "The prior Kelly biography establishes Sing as the 2016 Meena film but lacks its writer. The full question query uses only prior entities and retrieves the Sing article, which identifies Garth Jennings as writer. The distractor song's writers are correctly irrelevant.", None,
  [("before_action","Tori Kelly","She voiced a shy teenage elephant named Meena in the 2016 animated film \"Sing\"."),("retrieved_after_search","Sing (2016 American film)","It was directed and written by Garth Jennings")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-003.jsonl")
