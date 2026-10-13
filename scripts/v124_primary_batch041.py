"""Primary examination of exposed historical internal states 51–60."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (51, "Ladislas de Hoyos was", "retain_candidate", ["query_copies_less_relevant_visible_sentence", "exposed_development_set"],
  "The visible journalist article names Klaus Barbie as the interviewee. The query copies a later trial sentence rather than the direct interview relation, but retrieval yields Barbie's SS biography and nickname.", None,
  [("before_action", "Ladislas de Hoyos", "first journalist to interview in 1972 the former Gestapo member Klaus Barbie"), ("retrieved_after_search", "Klaus Barbie", "known as the \"Butcher of Lyon\"")]),
 (52, "Ladislas de Hoyos was", "retain_candidate", ["exposed_development_set"],
  "The journalist article identifies Klaus Barbie, and Barbie's biography says he was an SS/Gestapo functionary known as the Butcher of Lyon. The answer is supported.", None,
  [("before_action", "Ladislas de Hoyos", "former Gestapo member Klaus Barbie"), ("before_action", "Klaus Barbie", "known as the \"Butcher of Lyon\"")]),
 (53, "Vimala Pons is", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "Vimala Pons's article lists Elle but does not label its genre or director. The copied film list retrieves Elle's article, identifying it as the psychological thriller directed by Paul Verhoeven.", None,
  [("before_action", "Vimala Pons", "and Elle (2016)"), ("retrieved_after_search", "Elle (film)", "psychological thriller film directed by Paul Verhoeven")]),
 (54, "Vimala Pons is", "retain_candidate", ["exposed_development_set"],
  "Pons's film list includes Elle, and its article describes a psychological thriller directed by Paul Verhoeven. The answer is supported.", None,
  [("before_action", "Vimala Pons", "and Elle (2016)"), ("before_action", "Elle (film)", "psychological thriller film directed by Paul Verhoeven")]),
 (55, "What show starred Canadian actor", "retain_candidate", ["query_repeats_visible_title", "birthdate_inferred_via_twin", "exposed_development_set"],
  "The initially visible show article identifies Conrad Bain alongside Coleman and Bridges, but lacks his birth date. A title-repeat search retrieves Bonar Bain's same-date biography and identifies Bonar as Conrad's identical twin. The birthdate link is an inference from twinhood, not an explicit Conrad birth citation; verify directly before any promotion.", None,
  [("before_action", "Diff'rent Strokes", "Phillip Drummond (Conrad Bain)"), ("retrieved_after_search", "Bonar Bain", "February 4, 1923"), ("retrieved_after_search", "Bonar Bain", "identical twin brother of actor Conrad Bain")]),
 (56, "What show starred Canadian actor", "retain_candidate", ["birthdate_inferred_via_twin", "exposed_development_set"],
  "Diff'rent Strokes explicitly lists Conrad Bain with Coleman and Bridges. The retrieved Bonar Bain article gives February 4, 1923 for Bonar and says Conrad is his identical twin; the date for Conrad is indirectly inferred and requires direct verification before reuse.", None,
  [("before_action", "Diff'rent Strokes", "stars Gary Coleman and Todd Bridges"), ("before_action", "Diff'rent Strokes", "Phillip Drummond (Conrad Bain)"), ("before_action", "Bonar Bain", "identical twin brother of actor Conrad Bain")]),
 (57, "Go Radio and Wire", "retain_candidate", ["question_grounded_disambiguation", "exposed_development_set"],
  "The question names both bands and the initial Wire article calls it a rock band; the query disambiguates Wire from electrical-wire hits and retrieves Go Radio's rock-band article.", None,
  [("before_action", "Wire (band)", "English rock band"), ("retrieved_after_search", "Go Radio", "American rock band")]),
 (58, "Go Radio and Wire", "retain_candidate", ["exposed_development_set"],
  "Both band articles explicitly use 'rock band', supporting the common genre answer while the antenna results are irrelevant distractors.", None,
  [("before_action", "Wire (band)", "English rock band"), ("before_action", "Go Radio", "American rock band")]),
 (59, "What GI Joe actor", "retain_candidate", ["query_copies_visible_cast", "retrieved_title_text_malformed", "exposed_development_set"],
  "The film's visible cast list includes Channing Tatum. Its copied list retrieves his biography, whose G.I. Joe film titles are partially malformed but which explicitly refers to both G.I. Joe films and his role Duke.", None,
  [("before_action", "Hail, Caesar!", "and Channing Tatum"), ("retrieved_after_search", "Channing Tatum", "portrayal of the character Duke"), ("retrieved_after_search", "Channing Tatum", "both \" G.I. Joe\" films")]),
 (60, "What GI Joe actor", "retain_candidate", ["retrieved_title_text_malformed", "exposed_development_set"],
  "Channing Tatum is listed in Hail, Caesar! and his biography identifies his Duke role plus G.I. Joe films. The connection is supported despite missing film-title text in the source extraction.", None,
  [("before_action", "Hail, Caesar!", "and Channing Tatum"), ("before_action", "Channing Tatum", "portrayal of the character Duke"), ("before_action", "Channing Tatum", "both \" G.I. Joe\" films")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-041.jsonl", cohort="v117_exposed_internal50")
