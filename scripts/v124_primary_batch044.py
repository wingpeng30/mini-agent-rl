"""Primary examination of exposed historical internal states 81–90."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (81, "The sister publication of The Journal", "retain_candidate", ["query_copies_less_relevant_visible_sentence", "exposed_development_set"],
  "The Journal article already names Evening Chronicle and Sunday Sun as sister publications. The query copies its generic opening sentence rather than Sunday Sun, but the retrieval supplies the latter's first-publication date.", None,
  [("before_action", "The Journal (newspaper)", "sister publications the \"Evening Chronicle\" and the \"Sunday Sun\""), ("retrieved_after_search", "Sunday Sun", "First published on 31 August 1919")]),
 (82, "The sister publication of The Journal", "retain_candidate", ["exposed_development_set"],
  "The Journal identifies Sunday Sun alongside Evening Chronicle as its sister publications, and Sunday Sun's article gives 31 August 1919. The answer is supported, though the wording 'sister publication of' is relationally compressed.", None,
  [("before_action", "The Journal (newspaper)", "sister publications the \"Evening Chronicle\" and the \"Sunday Sun\""), ("before_action", "Sunday Sun", "First published on 31 August 1919")]),
 (83, "The Anoaʻi family includes", "retain_candidate", ["query_copies_visible_family_list", "exposed_development_set"],
  "The initial family article lists Roman Reigns but not his other sport. The copied family list retrieves his biography, which identifies Canadian football as his former profession.", None,
  [("before_action", "Anoaʻi family", "Roman Reigns"), ("retrieved_after_search", "Roman Reigns", "former professional Canadian football player")]),
 (84, "The Anoaʻi family includes", "retain_candidate", ["exposed_development_set"],
  "Roman Reigns is listed as a family member, and his article calls him both a wrestler and former professional Canadian football player. The other sport is supported.", None,
  [("before_action", "Anoaʻi family", "Roman Reigns"), ("before_action", "Roman Reigns", "former professional Canadian football player")]),
 (85, "The composer of the opera", "retain_candidate", ["query_copies_visible_context", "exposed_development_set"],
  "The Zoo article names Arthur Sullivan as composer. The copied opening sentence retrieves his biography and composition list, both stating 13 major orchestral works.", None,
  [("before_action", "The Zoo", "music by Arthur Sullivan"), ("retrieved_after_search", "Arthur Sullivan", "13 major orchestral works")]),
 (86, "The composer of the opera", "retain_candidate", ["exposed_development_set"],
  "The opera's composer is Arthur Sullivan, whose biography explicitly counts 13 major orchestral works. The answer is supported.", None,
  [("before_action", "The Zoo", "music by Arthur Sullivan"), ("before_action", "Arthur Sullivan", "13 major orchestral works")]),
 (87, "Which of the stars in", "retain_candidate", ["query_copies_visible_cast", "exposed_development_set"],
  "The soundtrack article provides the film's cast including Hugh Grant. Its copied opening sentence retrieves Grant's biography, which gives the specified birth date.", None,
  [("before_action", "Sense and Sensibility (soundtrack)", "Emma Thompson, Hugh Grant, Kate Winslet and Tom Wilkinson"), ("retrieved_after_search", "Hugh Grant", "born 9 September 1960")]),
 (88, "Which of the stars in", "retain_candidate", ["exposed_development_set"],
  "The film cast list names Hugh Grant and his biography gives 9 September 1960. The answer is supported.", None,
  [("before_action", "Sense and Sensibility (soundtrack)", "Emma Thompson, Hugh Grant, Kate Winslet and Tom Wilkinson"), ("before_action", "Hugh Grant", "born 9 September 1960")]),
 (89, "Which of the stars in All the Wilderness", "retain_candidate", ["query_copies_visible_cast", "exposed_development_set"],
  "The film article names Kodi Smit-McPhee among its stars. Copying the cast list retrieves his biography, with the requested birth date.", None,
  [("before_action", "All the Wilderness", "stars Kodi Smit-McPhee"), ("retrieved_after_search", "Kodi Smit-McPhee", "born 13 June 1996")]),
 (90, "Which of the stars in All the Wilderness", "retain_candidate", ["exposed_development_set"],
  "All the Wilderness lists Kodi Smit-McPhee as a star and his article states 13 June 1996. The answer is supported.", None,
  [("before_action", "All the Wilderness", "stars Kodi Smit-McPhee"), ("before_action", "Kodi Smit-McPhee", "born 13 June 1996")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-044.jsonl", cohort="v117_exposed_internal50")
