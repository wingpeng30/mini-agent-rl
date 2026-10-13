"""Primary examination of exposed historical internal states 11–20."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (11, "Bio is a studio album", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "The album article identifies Chuck Berry, but his pioneer status is absent before search. The copied album sentence retrieves his biography, which states that status.", None,
  [("before_action", "Bio (album)", "studio album by Chuck Berry"), ("retrieved_after_search", "Chuck Berry", "one of the pioneers of rock and roll music")]),
 (12, "Bio is a studio album", "retain_candidate", ["exposed_development_set"],
  "The album is attributed to Chuck Berry, and his biography gives the full name in the proposed answer and calls him a rock-and-roll pioneer. The answer is supported in this exposed set.", None,
  [("before_action", "Bio (album)", "studio album by Chuck Berry"), ("before_action", "Chuck Berry", "Charles Edward Anderson Berry"), ("before_action", "Chuck Berry", "one of the pioneers of rock and roll music")]),
 (13, "The Book of Other People contains", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "The anthology lists Colm Tóibín among its authors, but the Irish-novelist qualifier is missing before search. The copied author list retrieves his biography and supplies that qualifier.", None,
  [("before_action", "The Book of Other People", "Colm Tóibín"), ("retrieved_after_search", "Colm Tóibín", "an Irish novelist")]),
 (14, "The Book of Other People contains", "retain_candidate", ["exposed_development_set"],
  "The anthology names Colm Tóibín among its authors and the biography identifies him as an Irish novelist. Both relations are visible before this answer.", None,
  [("before_action", "The Book of Other People", "Colm Tóibín"), ("before_action", "Colm Tóibín", "an Irish novelist")]),
 (15, "When was the singer who performed", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "The song article identifies Kylie Minogue as performer but omits her birth date. Searching its copied opening sentence retrieves her biography with that date.", None,
  [("before_action", "I Should Be So Lucky", "performed by Australian recording artist and songwriter Kylie Minogue"), ("retrieved_after_search", "Kylie Minogue", "born 28 May 1968")]),
 (16, "When was the singer who performed", "retain_candidate", ["exposed_development_set"],
  "The song article identifies the performer as Kylie Minogue; her biography states the answer date, 28 May 1968.", None,
  [("before_action", "I Should Be So Lucky", "performed by Australian recording artist and songwriter Kylie Minogue"), ("before_action", "Kylie Minogue", "born 28 May 1968")]),
 (17, "One of Don Calfa's best known roles", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "Don Calfa's article links the hitman role to Weekend at Bernie's but lacks its director. The copied sentence retrieves the film article identifying Ted Kotcheff as director.", None,
  [("before_action", "Don Calfa", "the bumbling hitman in \"Weekend at Bernie's\""), ("retrieved_after_search", "Weekend at Bernie's", "directed by Ted Kotcheff")]),
 (18, "One of Don Calfa's best known roles", "retain_candidate", ["exposed_development_set"],
  "The actor article identifies the 1989 hitman film, and the film article names Ted Kotcheff as director. The answer is supported before the action.", None,
  [("before_action", "Don Calfa", "the bumbling hitman in \"Weekend at Bernie's\""), ("before_action", "Weekend at Bernie's", "directed by Ted Kotcheff")]),
 (19, "What year did the opera singer", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "The film description links the production to soprano Natalie Dessay but lacks her departure year. The copied sentence retrieves her biography with the 2013 departure date.", None,
  [("before_action", "Becoming Traviata", "coloratura soprano Natalie Dessay"), ("retrieved_after_search", "Natalie Dessay", "leaving the opera stage on 15 October 2013")]),
 (20, "What year did the opera singer", "retain_candidate", ["exposed_development_set"],
  "The film article connects Becoming Traviata to Natalie Dessay, and her biography says she left the opera stage in 2013. The question's word 'subject' is loose, but the intended singer and year are supported.", None,
  [("before_action", "Becoming Traviata", "coloratura soprano Natalie Dessay"), ("before_action", "Natalie Dessay", "leaving the opera stage on 15 October 2013")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-037.jsonl", cohort="v117_exposed_internal50")
