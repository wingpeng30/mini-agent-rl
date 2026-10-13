"""Primary examination of v122 training-supplement candidate states 171–180."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (171, "What is the name of the Hudson's Bay Company fur trader", "repair_candidate", ["answer_visible_before_search", "query_selects_wrong_explorer"],
  "The initial Robert Campbell article explicitly states that he established Fort Frances on Frances Lake. The search instead copies a John Bell biography sentence; Bell shares the company and fur-trader role but is not tied here to founding that fort. Retrieved results do not repair this wrong-entity search action.",
  "Convert to direct-answer supervision or construct and replay a genuinely missing Campbell relation; do not train the John Bell query.",
  [("before_action", "Robert Campbell (fur trader)", "He established Fort Frances, Yukon on Frances Lake"), ("before_action", "John Bell (explorer)", "was born Isle of Mull, Scotland"), ("retrieved_after_search", "John Bell (explorer)", "Hudson's Bay Company as a fur trader and explorer")]),
 (172, "What is the name of the Hudson's Bay Company fur trader", "repair_candidate", ["paired_wrong_explorer_query", "answer_supported_before_search"],
  "Robert Campbell is directly identified in the initial document as the fort's founder. The answer is supported, but the paired search pursued John Bell, whose similar occupation is a distractor rather than the needed establishment link.",
  "Use Robert Campbell as a direct answer if the original question is retained; re-audit any modified pair.",
  [("before_action", "Robert Campbell (fur trader)", "He established Fort Frances, Yukon on Frances Lake"), ("before_action", "John Bell (explorer)", "Hudson's Bay Company as a fur trader and explorer")]),
 (173, "Who is the English singer and musician who wrote", "retain_candidate", ["entity_query_checks_song_credit"],
  "The initial Joe Cocker biography establishes the English singer/musician description but not his credit for Something to Say. Searching the visible candidate's name retrieves The Last Puff, which explicitly attributes the song to him.", None,
  [("before_action", "Joe Cocker", "an English singer and musician"), ("retrieved_after_search", "The Last Puff", '"Something to Say" was written by Joe Cocker')]),
 (174, "Who is the English singer and musician who wrote", "retain_candidate", [],
  "The biography calls Joe Cocker an English singer and musician, and the retrieved album article directly credits him as writer of Something to Say. The answer is supported at this action time.", None,
  [("before_action", "Joe Cocker", "an English singer and musician"), ("before_action", "The Last Puff", '"Something to Say" was written by Joe Cocker')]),
 (175, "The college football team T. J. Ward played for", "retain_candidate", ["query_uses_visible_oregon_link", "retrieval_adds_first_team_year"],
  "Ward's initial biography says he played college football at Oregon. The query retains that Oregon link and retrieves the Ducks football article stating when the first team was fielded. It could be shorter, but the missing year is supplied by an action-time source.", None,
  [("before_action", "T. J. Ward", "He played college football at Oregon"), ("retrieved_after_search", "Oregon Ducks football", "The first football team was fielded in 1894")]),
 (176, "The college football team T. J. Ward played for", "retain_candidate", [],
  "T. J. Ward's Oregon college-football affiliation and the Oregon Ducks program's first team in 1894 establish the requested year. The answer is supported by separately bound documents.", None,
  [("before_action", "T. J. Ward", "He played college football at Oregon"), ("before_action", "Oregon Ducks football", "The first football team was fielded in 1894")]),
 (177, "Who wrote the closing night film", "retain_candidate", ["query_uses_visible_closing_film", "retrieval_adds_writers"],
  "The festival article names Snowpiercer as its 2013 closing-night film. Searching that visible title retrieves the film article, which attributes the screenplay to Bong and Kelly Masterson. The question asks who wrote the film, not the underlying graphic novel.", None,
  [("before_action", "2013 Deauville American Film Festival", '" Snowpiercer" by Bong Joon-ho was the closing night film'), ("retrieved_after_search", "Snowpiercer", "written by Bong and Kelly Masterson")]),
 (178, "Who wrote the closing night film", "retain_candidate", ["bong_short_form_in_source"],
  "The festival source identifies Snowpiercer, and the film source says it was written by Bong and Kelly Masterson, with Bong Joon-ho named in the same document as director. The frozen abbreviated answer follows the document's own wording; a final export should preserve unambiguous name normalization.", None,
  [("before_action", "2013 Deauville American Film Festival", '" Snowpiercer" by Bong Joon-ho was the closing night film'), ("before_action", "Snowpiercer", "written by Bong and Kelly Masterson")]),
 (179, "When was the singer of the band who made Out of Ashes", "retain_candidate", ["query_uses_visible_album_band_frontman", "retrieval_adds_birth_date"],
  "The initial album article identifies Dead by Sunrise and Chester Bennington as its lead vocalist. The query includes these visible identities; the resulting Bennington biography supplies his full birth date. This is a grounded, if verbose, query.", None,
  [("before_action", "Out of Ashes", "Linkin Park lead vocalist Chester Bennington"), ("retrieved_after_search", "Chester Bennington", "March 20, 1976")]),
 (180, "When was the singer of the band who made Out of Ashes", "retain_candidate", [],
  "Out of Ashes is attributed to Dead by Sunrise and names Chester Bennington among its members; the retrieved band article confirms him as lead singer, and his biography gives March 20, 1976. The answer is supported.", None,
  [("before_action", "Out of Ashes", "American rock band Dead by Sunrise"), ("before_action", "Dead by Sunrise", "lead singer Chester Bennington"), ("before_action", "Chester Bennington", "March 20, 1976")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-096.jsonl", cohort="v122_train_candidates")
