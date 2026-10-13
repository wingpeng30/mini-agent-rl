"""Primary examination of v122 training-supplement candidate states 131–140."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (131, "Prelude in C-sharp minor was composed", "repair_candidate", ["same_title_multiple_composers", "query_selects_rachmaninoff_without_disambiguation"],
  "The initial input includes both a Scriabin Prelude in C-sharp minor and Rachmaninoff's Prelude in the same key. The question names no opus or composer qualifier. The query copies a Rachmaninoff-specific sentence and retrieves his 1873 birth year, selecting one of two possible works rather than disambiguating the question.",
  "Specify Op. 3 No. 2 or The Bells of Moscow for Rachmaninoff, then replay; do not assume every Prelude in C-sharp minor refers to him.",
  [("before_action", "Prelude in C-sharp minor, Op. 11, No. 10 (Scriabin)", "Prelude in C-sharp minor"), ("before_action", "Prelude in C-sharp minor (Rachmaninoff)", "Op.  3, No. 2"), ("retrieved_after_search", "Sergei Rachmaninoff", "1873")]),
 (132, "Prelude in C-sharp minor was composed", "repair_candidate", ["paired_same_title_multiple_composers"],
  "1873 is Rachmaninoff's birth year and his Op. 3 No. 2 is documented. But the input also contains Scriabin's same-key prelude, so the bare work title in the question does not uniquely determine that composer.",
  "Use a Rachmaninoff-opus-specific question and replay.",
  [("before_action", "Prelude in C-sharp minor, Op. 11, No. 10 (Scriabin)", "Prelude in C-sharp minor"), ("before_action", "Prelude in C-sharp minor (Rachmaninoff)", "Op.  3, No. 2"), ("before_action", "Sergei Rachmaninoff", "1873")]),
 (133, "What two sitcoms included appearances from an English actress", "retain_candidate", ["answer_visible_before_search", "search_checks_rush_hour_cast"],
  "Miranda Hart's initial biography names Hyperdrive and Not Going Out and calls her English actress/comedian. Searching her name retrieves the Rush Hour UK cast, verifying the remaining show clue. The two sitcom names are already visible.", None,
  [("before_action", "Miranda Hart", "English actress and comedian"), ("before_action", "Miranda Hart", "\"Hyperdrive\" and \"Not Going Out\""), ("retrieved_after_search", "Rush Hour (UK TV series)", "Miranda Hart")]),
 (134, "What two sitcoms included appearances from an English actress", "retain_candidate", ["answer_visible_before_paired_search"],
  "Miranda Hart is listed in Rush Hour's cast and her biography specifically names the two sitcoms Hyperdrive and Not Going Out. The answer is supported; the search verifies the Rush Hour condition.", None,
  [("before_action", "Rush Hour (UK TV series)", "Miranda Hart"), ("before_action", "Miranda Hart", "\"Hyperdrive\" and \"Not Going Out\"")]),
 (135, "\"I Believe I Can Fly\" is a song", "retain_candidate", ["query_uses_visible_soundtrack_film"],
  "The initial song article places I Believe I Can Fly on Space Jam's soundtrack. Searching that film link retrieves Space Jam's article, identifying Joe Pytka as film director. Animation segment directors in the same sentence do not replace the named overall director.", None,
  [("before_action", "I Believe I Can Fly", "soundtrack to the 1996 film \"Space Jam\""), ("retrieved_after_search", "Space Jam", "directed by Joe Pytka")]),
 (136, "\"I Believe I Can Fly\" is a song", "retain_candidate", [],
  "The song is tied to Space Jam's soundtrack and the film article credits Joe Pytka with direction. The answer is supported by the action-time sources.", None,
  [("before_action", "I Believe I Can Fly", "soundtrack to the 1996 film \"Space Jam\""), ("before_action", "Space Jam", "directed by Joe Pytka")]),
 (137, "Dawn French's Girls Who Do Comedy included", "repair_candidate", ["multiple_possible_american_comedian_hosts", "query_lists_many_candidates"],
  "The initial series article lists Whoopi Goldberg and Joan Rivers among interviewees, plus Phyllis Diller in its overview. The question asks for an American comedienne and television host without a unique clue, and the query repeats the full list. Retrieval finds Joan Rivers's biography, but cannot rule out another listed person meeting those attributes.",
  "Specify Joan Rivers's episode or a distinguishing fact before asking for her name; replay the pair.",
  [("before_action", "Dawn French's Girls Who Do Comedy", "Whoopi Goldberg, Catherine Tate, Kathy Burke, Julie Walters, Victoria Wood and Joan Rivers"), ("retrieved_after_search", "Joan Rivers", "American comedian, actress, writer, producer, and television host")]),
 (138, "Dawn French's Girls Who Do Comedy included", "repair_candidate", ["paired_interviewee_not_unique", "legal_name_answer_vs_stage_name"],
  "Joan Alexandra Molinsky is Joan Rivers's legal name, and Rivers appeared in the listed series. The original question does not uniquely identify her among multiple American female entertainers in the series; answering with her birth name further obscures the requested public identity.",
  "Use a uniquely identifying Joan Rivers question and prefer the public name unless birth name is asked; replay.",
  [("before_action", "Dawn French's Girls Who Do Comedy", "Whoopi Goldberg"), ("before_action", "Dawn French's Girls Who Do Comedy", "Joan Rivers"), ("before_action", "Joan Rivers", "Joan Alexandra Molinsky")]),
 (139, "When was the club which Johnny Mølby was most recently", "repair_candidate", ["most_recently_unanchored", "query_uses_visible_club_link"],
  "The initial Johnny Mølby article says he was 'most recently' manager of Viborg FF, an inherently time-dependent claim. Searching that club link retrieves its 1896 founding. The query is grounded, but the question needs a source-snapshot time or a timeless wording naming Viborg FF.",
  "Ask when Viborg FF, a club formerly managed by Johnny Mølby, was founded; replay.",
  [("before_action", "Johnny Mølby", "most recently the manager of Viborg FF"), ("retrieved_after_search", "Viborg FF", "club was founded in 1896")]),
 (140, "When was the club which Johnny Mølby was most recently", "repair_candidate", ["paired_most_recently_unanchored"],
  "Viborg FF's 1896 founding is supported, and Mølby's association with it is visible. The frozen question presents a time-sensitive 'most recently' status without a time anchor.",
  "Retain 1896 under a dated or timeless club association wording after replay.",
  [("before_action", "Johnny Mølby", "most recently the manager of Viborg FF"), ("before_action", "Viborg FF", "club was founded in 1896")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-092.jsonl", cohort="v122_train_candidates")
