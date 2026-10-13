"""Primary examination of v122 internal candidate states 31–40."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (31, "When was the American actor who was among", "repair_candidate", ["question_does_not_uniquely_identify_cast_member", "query_selects_secondary_cast_sentence"],
  "Flawless names multiple American actors, including its leads, but the question provides no distinguishing role or name. The query happens to select the secondary-cast sentence and retrieves Wilson Jermaine Heredia's birth date. Search success does not make the original target unique.", None,
  [("before_action", "Flawless (1999 film)", "stars Robert De Niro and Philip Seymour Hoffman"), ("before_action", "Flawless (1999 film)", "Daphne Rubin-Vega and Wilson Jermaine Heredia"), ("retrieved_after_search", "Wilson Jermaine Heredia", "born December 2, 1971")]),
 (32, "When was the American actor who was among", "repair_candidate", ["question_does_not_uniquely_identify_cast_member"],
  "December 2, 1971 is supported for Wilson Jermaine Heredia, a Flawless cast member, but the question's generic 'American actor' can describe other cast members. Add a distinguishing role or name before treating the pair as unambiguous.", None,
  [("before_action", "Flawless (1999 film)", "Daphne Rubin-Vega and Wilson Jermaine Heredia"), ("before_action", "Wilson Jermaine Heredia", "born December 2, 1971")]),
 (33, "How long is the river in which Fort Watuaga", "retain_candidate", ["question_misspells_watauga", "query_repeats_correct_visible_title"],
  "The question misspells Watauga, but the visible fort article identifies the Watauga River. Repeating the correct fort title retrieves the river article and its 78.5-mile length. The typo should be cleaned in any export.", None,
  [("before_action", "Fort Watauga", "Sycamore Shoals of the Watauga River"), ("retrieved_after_search", "Watauga River", "78.5 mi long")]),
 (34, "How long is the river in which Fort Watuaga", "repair_candidate", ["answer_omits_length_unit", "question_misspells_watauga"],
  "The river article gives 78.5 miles, while the answer action emits only 78.5. For a 'how long' question the unit is material; repair the answer to include miles and clean the fort-name typo.", None,
  [("before_action", "Fort Watauga", "Sycamore Shoals of the Watauga River"), ("before_action", "Watauga River", "78.5 mi long")]),
 (35, "How many episodes were there of the sitcom", "repair_candidate", ["episode_count_snapshot_only", "question_lacks_as_of_date", "query_copies_visible_context"],
  "The actor article names Last Man Standing. Retrieval reports 130 episodes only for an ABC run ending March 31, 2017; the question asks a timeless total without specifying that run or date. The query is grounded, but the paired question needs temporal scope.", None,
  [("before_action", "Jordan Masterson", "Ryan Vogelson on the sitcom \"Last Man Standing\""), ("retrieved_after_search", "Last Man Standing (U.S. TV series)", "aired on ABC from October 11, 2011 to March 31, 2017 with 130 episodes")]),
 (36, "How many episodes were there of the sitcom", "repair_candidate", ["episode_count_snapshot_only", "question_lacks_as_of_date"],
  "130 is supported for the show's cited ABC period through March 2017. The unqualified question could be read as a full-series total; retain the historical count only in a date- and network-anchored rewrite.", None,
  [("before_action", "Jordan Masterson", "Ryan Vogelson on the sitcom \"Last Man Standing\""), ("before_action", "Last Man Standing (U.S. TV series)", "aired on ABC from October 11, 2011 to March 31, 2017 with 130 episodes")]),
 (37, "How many BAFTA awards did the actress", "repair_candidate", ["unanchored_award_count", "query_copies_visible_context"],
  "Run identifies Olivia Colman as Carol and the retrieved biography says she has won three BAFTAs, but the question omits an as-of point for an accumulating award count. A snapshot-bound version would be needed before reuse.", None,
  [("before_action", "Run (TV series)", "Carol (Olivia Colman)"), ("retrieved_after_search", "Olivia Colman", "won three BAFTA Awards")]),
 (38, "How many BAFTA awards did the actress", "repair_candidate", ["unanchored_award_count"],
  "Three is the count stated in the historical Olivia Colman snippet. As a timeless question it is unstable; the answer can only be retained with the source snapshot's temporal scope.", None,
  [("before_action", "Run (TV series)", "Carol (Olivia Colman)"), ("before_action", "Olivia Colman", "won three BAFTA Awards")]),
 (39, "What is the land area of the state", "retain_candidate", ["query_copies_visible_context"],
  "Whitington's article identifies South Australia as his state. Its copied sentence retrieves South Australia's article with a land area of 983482 square kilometres.", None,
  [("before_action", "A. O. Whitington", "businessman and sportsman in South Australia"), ("retrieved_after_search", "South Australia", "total land area of 983482 km2")]),
 (40, "What is the land area of the state", "retain_candidate", [],
  "Whitington is placed in South Australia and that state's article gives 983482 km2. The answer includes the unit and is supported.", None,
  [("before_action", "A. O. Whitington", "businessman and sportsman in South Australia"), ("before_action", "South Australia", "total land area of 983482 km2")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-049.jsonl", cohort="v122_internal_candidates")
