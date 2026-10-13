"""Primary examination of v122 internal candidate states 271–280."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (271, "Which song on Eminem's album", "retain_candidate", ["query_uses_visible_chart_song", "source_counting_inconsistency"],
  "The chart snippet names The Monster by Eminem among songs also in the 2013 New Zealand top 50. The search retrieves the song article confirming its Marshall Mathers LP 2 album membership. The chart prose says 'Four songs' while listing five titles, but the named song's inclusion is explicit.", None,
  [("before_action", "New Zealand top 50 singles of 2014", "\"The Monster\" by Eminem"), ("retrieved_after_search", "The Monster (song)", "taken from Eminem's album \"The Marshall Mathers LP 2\"")]),
 (272, "Which song on Eminem's album", "retain_candidate", ["source_counting_inconsistency_does_not_erase_named_song"],
  "The Monster is explicitly named in the cross-year chart list and in the album-linked song article. Although that list's stated count is internally inconsistent, the individual song relation is supported.", None,
  [("before_action", "New Zealand top 50 singles of 2014", "\"The Monster\" by Eminem"), ("before_action", "The Monster (song)", "taken from Eminem's album \"The Marshall Mathers LP 2\"")]),
 (273, "What historic province is the Sacred Heart Church", "repair_candidate", ["query_guesses_county_before_church_link", "church_name_not_unique_in_question"],
  "The initial County Tyrone article states Ulster, but the initial state does not yet link the asked Sacred Heart Church to County Tyrone. Searching County Tyrone preselects the county and retrieves Sacred Heart Church, Plumbridge, confirming the link. The question omits Plumbridge, leaving the church name potentially ambiguous.",
  "Specify Sacred Heart Church in Plumbridge and search that church or town first, then replay the county-to-province chain.",
  [("before_action", "County Tyrone", "historic province of Ulster"), ("retrieved_after_search", "Sacred Heart Church, Plumbridge", "village of Plumbridge, County Tyrone")]),
 (274, "What historic province is the Sacred Heart Church", "repair_candidate", ["paired_query_guesses_county", "church_name_not_unique"],
  "Ulster is supported for County Tyrone, and the retrieved Plumbridge church article places that church there. The original question and query do not establish the specific church before the county is chosen.",
  "Retain Ulster only with a Plumbridge-specific, replayed pair.",
  [("before_action", "County Tyrone", "historic province of Ulster"), ("before_action", "Sacred Heart Church, Plumbridge", "Plumbridge, County Tyrone")]),
 (275, "Sir John Brunner, 1st Baronet formed", "retain_candidate", ["search_verifies_partner_relation"],
  "Ludwig Mond's birth date is already visible, and a statue article says he partnered with John Brunner. Searching Brunner's statue retrieves Brunner's biography, which explicitly ties him and Mond to Brunner Mond & Co., confirming the question's company-specific relation.", None,
  [("before_action", "Ludwig Mond", "7 March 1839"), ("retrieved_after_search", "Sir John Brunner, 1st Baronet", "partnership to create the chemical company Brunner Mond & Co.")]),
 (276, "Sir John Brunner, 1st Baronet formed", "retain_candidate", [],
  "Brunner's biography identifies chemist Ludwig Mond as his partner in Brunner Mond & Co.; Mond's biography supplies 7 March 1839. The full birth date is supported.", None,
  [("before_action", "Sir John Brunner, 1st Baronet", "partnership to create the chemical company Brunner Mond & Co."), ("before_action", "Ludwig Mond", "7 March 1839")]),
 (277, "What is the nationality of the author of Hero and Leander", "repair_candidate", ["same_title_multiple_poems", "query_focuses_marlowe_completion_not_disambiguation"],
  "The initial state contains Hero and Leander poems by Christopher Marlowe and Leigh Hunt, plus the underlying myth. The query follows Marlowe's completion sentence and retrieves his English biography, but the question does not specify which poem or author. The shared English answer does not resolve the referent ambiguity.",
  "Ask for the nationality of Christopher Marlowe, author of the posthumously published Hero and Leander poem, then replay.",
  [("before_action", "Hero and Leander (poem)", "poem by Christopher Marlowe"), ("before_action", "Hero and Leander (1819 poem)", "poem by Leigh Hunt"), ("retrieved_after_search", "Christopher Marlowe", "was an English playwright")]),
 (278, "What is the nationality of the author of Hero and Leander", "repair_candidate", ["paired_same_title_multiple_poems"],
  "English is documented for Marlowe, but the input also shows a different Hero and Leander poem by Leigh Hunt. The question lacks an edition, year or author clue and is not a unique task despite the target's apparent correctness.",
  "Retain English only with a Marlowe-specific, replayed question.",
  [("before_action", "Hero and Leander (poem)", "poem by Christopher Marlowe"), ("before_action", "Hero and Leander (1819 poem)", "poem by Leigh Hunt"), ("before_action", "Christopher Marlowe", "was an English playwright")]),
 (279, "What Australian film actress starred in Newness", "retain_candidate", ["query_copies_visible_cast_relation"],
  "Newness names Courtney Eaton in the cast. The search retrieves her biography explicitly describing her as an Australian film actress.", None,
  [("before_action", "Newness", "Courtney Eaton"), ("retrieved_after_search", "Courtney Eaton", "is an Australian film actress")]),
 (280, "What Australian film actress starred in Newness", "retain_candidate", [],
  "Courtney Eaton is named in the film cast, and her biography supplies the Australian film-actress qualification. The answer is supported.", None,
  [("before_action", "Newness", "Courtney Eaton"), ("before_action", "Courtney Eaton", "is an Australian film actress")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-073.jsonl", cohort="v122_internal_candidates")
