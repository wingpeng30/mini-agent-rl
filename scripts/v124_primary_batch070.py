"""Primary examination of v122 internal candidate states 241–250."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (241, "What is the nickname of the driver", "retain_candidate", ["query_copies_visible_driver_relation"],
  "The Brutus article names Jim Liberman as its driver. The search follows that visible relationship and retrieves Liberman's Jungle Jim nickname.", None,
  [("before_action", "Brutus (Funny Car)", "driven by Jim Liberman"), ("retrieved_after_search", "Jim Liberman", "known as \"Jungle Jim\"")]),
 (242, "What is the nickname of the driver", "retain_candidate", [],
  "Jim Liberman drove Brutus and his biography gives the nickname Jungle Jim. The answer is supported.", None,
  [("before_action", "Brutus (Funny Car)", "driven by Jim Liberman"), ("before_action", "Jim Liberman", "known as \"Jungle Jim\"")]),
 (243, "Who was the county named for", "repair_candidate", ["query_assumes_county_before_location_link"],
  "The initial Jackson County article says it was named for President Andrew Jackson, but no initial snippet links Cockrell to that county. The query inserts Jackson County as the answer-bearing county before the needed Cockrell-to-county relation is established; the search then retrieves Cockrell's article and confirms it. This is a lucky verification path, not a grounded missing-relation query.",
  "Search Cockrell, Missouri to establish its county first, then answer from the already visible Jackson County article; replay the pair.",
  [("before_action", "Jackson County, Missouri", "named for President Andrew Jackson"), ("retrieved_after_search", "Cockrell, Missouri", "unincorporated community in Jackson County")]),
 (244, "Who was the county named for", "repair_candidate", ["paired_query_assumes_county_before_location_link", "question_typo_call"],
  "The retrieved Cockrell article and initial Jackson County article support Andrew Jackson. But the preceding query preselected the county without visible grounds, and the question says 'community call Cockrell' instead of 'called'.",
  "Retain the answer only with a Cockrell-targeted search and corrected wording after replay.",
  [("before_action", "Cockrell, Missouri", "unincorporated community in Jackson County"), ("before_action", "Jackson County, Missouri", "named for President Andrew Jackson")]),
 (245, "What is the full name of the sculptor", "retain_candidate", ["query_copies_visible_sculptor_relation"],
  "The Comb of the Wind article credits Eduardo Chillida with the sculptures and Luis Peña Ganchegui with their architectural arrangement. The search retrieves Chillida's full name, preserving the artist-architect distinction.", None,
  [("before_action", "The Comb of the Wind", "three sculptures by Eduardo Chillida"), ("retrieved_after_search", "Eduardo Chillida", "Eduardo Chillida Juantegui")]),
 (246, "What is the full name of the sculptor", "retain_candidate", [],
  "Eduardo Chillida is the credited sculptor, and his biography expands his name to Eduardo Chillida Juantegui. The answer is supported.", None,
  [("before_action", "The Comb of the Wind", "three sculptures by Eduardo Chillida"), ("before_action", "Eduardo Chillida", "Eduardo Chillida Juantegui")]),
 (247, "Where is the university from which Glenn Kothmann", "retain_candidate", ["query_copies_visible_alma_mater"],
  "Glenn Kothmann's biography names Texas A&M University as his graduation institution. The search retrieves the university article placing it in College Station, Texas.", None,
  [("before_action", "Glenn Kothmann", "graduate from Texas A&M University"), ("retrieved_after_search", "Texas A&amp;M University", "in College Station, Texas, United States")]),
 (248, "Where is the university from which Glenn Kothmann", "retain_candidate", [],
  "Kothmann graduated from Texas A&M University, whose article directly states College Station, Texas, United States. The answer is supported.", None,
  [("before_action", "Glenn Kothmann", "graduate from Texas A&M University"), ("before_action", "Texas A&amp;M University", "in College Station, Texas, United States")]),
 (249, "What was the birthday of the man", "repair_candidate", ["query_omits_named_missing_person", "retrieval_happens_to_find_target"],
  "The initial Don Melnick snippet explicitly names the Thomas Hunt Morgan professorship. The query copies only a generic sentence calling Melnick a biologist, omitting Morgan; retrieval happens to surface Morgan's birth date. The answer is supported but this query does not target the missing person's birthday.",
  "Search Thomas Hunt Morgan from the named professorship, then replay the pair.",
  [("before_action", "Don Melnick", "Thomas Hunt Morgan Professor of Conservation Biology"), ("retrieved_after_search", "Thomas Hunt Morgan", "September 25, 1866")]),
 (250, "What was the birthday of the man", "repair_candidate", ["paired_query_omits_named_missing_person"],
  "September 25, 1866 is supported for Thomas Hunt Morgan, the person named in Melnick's professorship. The preceding query does not mention him or the professorship and should be corrected before the pair is reused.",
  "Keep the date only after a Morgan-targeted query is replayed and reviewed.",
  [("before_action", "Don Melnick", "Thomas Hunt Morgan Professor of Conservation Biology"), ("before_action", "Thomas Hunt Morgan", "September 25, 1866")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-070.jsonl", cohort="v122_internal_candidates")
