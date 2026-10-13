"""Primary case examination of original training rows 121–130."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (121, "When was one of the the notable", "retain_candidate", ["question_grammar"],
  "The Sugar Bowl article identifies Billy Vessels among Oklahoma's notable players, and his biography calls him a gridiron player and gives his death date as November 17, 2001. Frank Anderson's different date is a distractor.", None,
  [("before_action", "1951 Sugar Bowl", "Notable players for the two teams included Oklahoma's Billy Vessels"), ("before_action", "Billy Vessels", "November 17, 2001) was a gridiron football player")]),
 (122, "When was one of the co founder", "retain_candidate", ["question_grammar"],
  "The company article names Flamini as co-founder. His biography identifies him as a French professional footballer born 7 March 1984. This is the answer counterpart of search row 57.", None,
  [("before_action", "GF Biochemicals", "co-founded by Mathieu Flamini and Pasquale Granata"), ("before_action", "Mathieu Flamini", "born 7 March 1984) is a French professional footballer")]),
 (123, "Bugs Bunny's home", "retain_candidate", ["home_depiction_is_specific_movie"],
  "The Bugs Bunny/Road Runner Movie article says its bridging sequences depict Bugs at a home modeled on Fallingwater. The source supports this depiction; it should not be generalized to every portrayal of the character's home.", None,
  [("before_action", "The Bugs Bunny/Road Runner Movie", "show Bugs at his home, which is cantilevered over a carrot-juice waterfall (modeled on Frank Lloyd Wright's \"Fallingwater\" house")]),
 (124, "Name the 2005 romantic comedy", "retain_candidate", [],
  "The Wedding Date article gives its year, genre and Asking for Trouble adaptation. Dana Fox's biography links her writing credits for The Wedding Date and What Happens in Vegas. The title is supported by both clauses.", None,
  [("before_action", "The Wedding Date", "a 2005 American romantic comedy film"), ("before_action", "The Wedding Date", "Based on the novel \"Asking for Trouble\" by Elizabeth Young"), ("before_action", "Dana Fox", "writer of \"The Wedding Date\" (2005), \"What Happens in Vegas\" (2008)")]),
 (125, "What storm was noted", "repair_candidate", ["comparison_scope_overgeneralized"],
  "The December 1992 nor'easter article limits the less-severe comparison to the Maryland coast. The question states a global severity comparison without that scope; the Perfect Storm identity and No-Name nickname are supported, but the broader comparison is not.",
  "Proposed question: Which storm, nicknamed the No-Name Storm, was described as more severe than the December 1992 nor'easter along the Maryland coast? Recheck all same-question states and independently review the revised premise.",
  [("before_action", "December 1992 nor'easter", "Along the Maryland coast, the storm was less severe than the Perfect Storm in the previous year"), ("before_action", "1991 Perfect Storm", "also known as The No-Name Storm")]),
 (126, " Yuji Nomi", "retain_candidate", ["short_query_from_question_title"],
  "No prior documents are visible. 'Whisper' is a short term from the question's film title, and its actual top three includes Yuji Nomi's biography explicitly pairing Whisper of the Heart with The Cat Returns. The broad query may not be optimal, but its observed retrieval is relevant and not answer-derived.", None,
  [("retrieved_after_search", "Yuji Nomi", "His work includes the Studio Ghibli films \"Whisper of the Heart\" and \"The Cat Returns\"")]),
 (127, "In which American private", "retain_candidate", [],
  "Olmsted's biography states education at Clark University, and Clark's article explicitly describes it as an American private research university. The answer is supported.", None,
  [("before_action", "Andrew J. Olmsted", "educated at St. John's High School in Shrewsbury, MA, and Clark University"), ("before_action", "Clark University", "an American private research university")]),
 (128, "Which band was formed", "retain_candidate", ["same_name_singer_band_distractor"],
  "Walk the Moon is dated to 2006; the Marilyn Manson band, distinguished from the same-named singer biography, is dated to 1989. The earlier-band answer is correct.", None,
  [("before_action", "Walk the Moon", "started the band in 2006"), ("before_action", "Marilyn Manson (band)", "in Fort Lauderdale, Florida, in 1989")]),
 (129, "Kenneth Maurice Clarke", "retain_candidate", [],
  "Clarke's biography explicitly says he played in Super Bowl XV for Philadelphia. The Super Bowl XV article confirms the Oakland Raiders–Philadelphia Eagles matchup and 1980 season. Super Bowl XI is a different pairing.", None,
  [("before_action", "Ken Clarke (American football)", "He played in Super Bowl XV for the Philadelphia Eagles"), ("before_action", "Super Bowl XV", "champion Oakland Raiders and the National Football Conference (NFC) champion Philadelphia Eagles")]),
 (130, "Winnipeg Limited compted", "retain_candidate", ["question_typo_competed", "query_copies_partial_competitor_sentence"],
  "The Winnipeg Limited article already names Northern Pacific Railway's competing daytime train, but does not classify the railway as transcontinental. The copied competitor sentence retrieves the Northern Pacific article with that missing classification, supporting the intended railroad identification.", None,
  [("before_action", "Winnipeg Limited", "the Northern Pacific Railway’s unnamed daytime passenger train"), ("retrieved_after_search", "Northern Pacific Railway", "was a transcontinental railroad")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-013.jsonl")
