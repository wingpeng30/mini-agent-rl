"""Primary examination of v122 internal candidate states 111–120."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (111, "The Book of Daniel was written", "retain_candidate", ["query_copies_visible_author_relation"],
  "The novel snippet names E. L. Doctorow as author. The grounded query retrieves his biography, which verifies the missing American-editor qualification rather than inventing an author.", None,
  [("before_action", "The Book of Daniel (novel)", "novel by E. L. Doctorow"), ("retrieved_after_search", "E. L. Doctorow", "American novelist, editor, and professor")]),
 (112, "The Book of Daniel was written", "retain_candidate", [],
  "The novel article credits E. L. Doctorow and the retrieved biography expands his name to Edgar Lawrence Doctorow while describing him as an American editor. The full-name answer is supported.", None,
  [("before_action", "The Book of Daniel (novel)", "novel by E. L. Doctorow"), ("before_action", "E. L. Doctorow", "Edgar Lawrence \"E. L.\" Doctorow"), ("before_action", "E. L. Doctorow", "American novelist, editor, and professor")]),
 (113, "What Football player for the USC Trojans", "retain_candidate", ["query_copies_visible_player_relation"],
  "The 2005 USC team snippet names Matt Leinart as quarterback. The search retrieves Leinart's biography, which supplies the requested May 11, 1983 birth date.", None,
  [("before_action", "2005 USC Trojans football team", "quarterback and 2004 Heisman Trophy winner Matt Leinart"), ("retrieved_after_search", "Matt Leinart", "born May 11, 1983")]),
 (114, "What Football player for the USC Trojans", "retain_candidate", [],
  "Matt Leinart is identified with the USC team, and his biography confirms the exact birth date. The answer follows the same person.", None,
  [("before_action", "2005 USC Trojans football team", "Matt Leinart"), ("before_action", "Matt Leinart", "born May 11, 1983")]),
 (115, "A symphony of Three Orchestras", "retain_candidate", ["query_copies_visible_dedication"],
  "The work is explicitly dedicated to Boulez and the New York Philharmonic. The query follows the named person, and the retrieved Pierre Boulez biography gives his 2016 death year.", None,
  [("before_action", "A Symphony of Three Orchestras", "dedicated to Boulez and the New York Philharmonic"), ("retrieved_after_search", "Pierre Boulez", "5 January 2016")]),
 (116, "A symphony of Three Orchestras", "retain_candidate", [],
  "Boulez is the human dedicatee, and Pierre Boulez's biography records his death on 5 January 2016. The answer is supported.", None,
  [("before_action", "A Symphony of Three Orchestras", "dedicated to Boulez and the New York Philharmonic"), ("before_action", "Pierre Boulez", "5 January 2016")]),
 (117, "On what day of the week did the Simpson", "repair_candidate", ["question_typos_and_imprecise_appearance", "query_retrieves_weekday"],
  "The initial episode article identifies Homer's Grim Reaper story and the November 2, 2003 original airing. Searching its title retrieves the season article that explicitly says Sunday. The answer is supported, but the question misspells Homer, calls The Simpsons 'the Simpson', and uses 'appear' instead of a clear original-airing date.",
  "Ask on which weekday Treehouse of Horror XIV originally aired, retaining the Homer clue if desired; replay the pair.",
  [("before_action", "Treehouse of Horror XIV", "Homer takes on the role of the Grim Reaper"), ("retrieved_after_search", "The Simpsons (season 15)", "began on Sunday, November 2, 2003")]),
 (118, "On what day of the week did the Simpson", "repair_candidate", ["paired_question_typos_and_imprecise_appearance"],
  "Sunday is explicitly supported by the retrieved season article for Treehouse of Horror XIV. The original wording still needs correction before the pair is a clean training or internal item.",
  "Retain Sunday only in a revised original-airing question with replayed input and independent review.",
  [("before_action", "Treehouse of Horror XIV", "originally aired on the Fox network"), ("before_action", "The Simpsons (season 15)", "began on Sunday, November 2, 2003")]),
 (119, "What body governs the campus", "repair_candidate", ["campus_vs_university_governance_wording"],
  "The stadium snippet locates Eck Stadium on Wichita State University's campus; the retrieved university article says the university is governed by the Kansas Board of Regents. The question asks what body governs the campus, while the source asserts university governance, so the institutional object should be made explicit.",
  "Ask which body governs Wichita State University, on whose campus Eck Stadium stands; replay the pair.",
  [("before_action", "Eck Stadium", "on the campus of Wichita State University"), ("retrieved_after_search", "Wichita State University", "governed by the Kansas Board of Regents")]),
 (120, "What body governs the campus", "repair_candidate", ["paired_campus_vs_university_governance_wording"],
  "Kansas Board of Regents is directly supported as Wichita State University's governing body. The original campus-level phrasing should be replaced with the university relation before the answer is treated as a clean paired label.",
  "Retain the answer with the university-specific wording after context replay and review.",
  [("before_action", "Eck Stadium", "on the campus of Wichita State University"), ("before_action", "Wichita State University", "governed by the Kansas Board of Regents")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-057.jsonl", cohort="v122_internal_candidates")
