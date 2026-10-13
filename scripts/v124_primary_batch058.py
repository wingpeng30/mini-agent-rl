"""Primary examination of v122 internal candidate states 121–130."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (121, "Which two cities does the motorway", "repair_candidate", ["query_guesses_unestablished_junction_motorway_link", "answer_already_visible_for_guessed_motorway"],
  "The initial M1 snippet already gives London and Leeds, but no visible snippet yet establishes that Sheffield Parkway meets the M1 at junction 33. The query inserts M1 as a candidate from retrieval ranking and asks the full question; the subsequent Parkway article supplies the missing identity link. It works as verification, but should not teach an unqualified guessed target as if it were grounded.",
  "Search for Sheffield Parkway and junction 33 without pre-committing to M1, then replay retrieval and answer under the same budget.",
  [("before_action", "M1 motorway", "connecting London to Leeds"), ("retrieved_after_search", "Sheffield Parkway", "junction 33 of the M1")]),
 (122, "Which two cities does the motorway", "repair_candidate", ["paired_query_guesses_unestablished_link"],
  "The final evidence supports the M1 identity and London-to-Leeds endpoints. The answer is correct, but the preceding search inserted the M1 before that junction link was observed, so the paired search example needs correction.",
  "Retain London and Leeds only after a revised grounded query is replayed and the whole pair reviewed.",
  [("before_action", "Sheffield Parkway", "junction 33 of the M1"), ("before_action", "M1 motorway", "connecting London to Leeds")]),
 (123, "Who portrays W'Kabi", "retain_candidate", ["answer_actor_visible_before_search", "search_verifies_director_clue"],
  "The W'Kabi snippet names Daniel Kaluuya and Black Panther, while the Ryan Coogler director clue is not yet confirmed. The search follows the visible film link and retrieves the film article verifying Coogler. This search checks the identifying condition rather than adding the actor name.", None,
  [("before_action", "W'Kabi", "portrayed by Daniel Kaluuya"), ("retrieved_after_search", "Black Panther (film)", "directed by Ryan Coogler")]),
 (124, "Who portrays W'Kabi", "retain_candidate", [],
  "W'Kabi's source identifies Kaluuya as the actor, and the retrieved film article confirms Black Panther was directed by Ryan Coogler. The answer follows the same film.", None,
  [("before_action", "W'Kabi", "portrayed by Daniel Kaluuya"), ("before_action", "Black Panther (film)", "directed by Ryan Coogler")]),
 (125, "When was the group founded", "retain_candidate", ["query_copies_visible_duo_relation"],
  "The song snippet names Brooks & Dunn as recording duo. The search copies that grounded relation and retrieves the duo article giving its 1990 foundation.", None,
  [("before_action", "Play Something Country", "recorded by American country music duo Brooks & Dunn"), ("retrieved_after_search", "Brooks &amp; Dunn", "duo was founded in 1990")]),
 (126, "When was the group founded", "retain_candidate", [],
  "Play Something Country is credited to Brooks & Dunn, and their article states the duo was founded in 1990. The year is supported.", None,
  [("before_action", "Play Something Country", "duo Brooks & Dunn"), ("before_action", "Brooks &amp; Dunn", "founded in 1990")]),
 (127, "What movie stars Chris Hemsworth", "repair_candidate", ["unnecessary_search_answer_already_visible", "remake_relation_already_visible"],
  "The initial Red Dawn (2012 film) snippet already names Chris Hemsworth and says its screenplay is based on the 1984 film of the same name. Searching the already-visible title adds neither the movie identity nor its remake basis.",
  "Answer directly from the initial state or exclude this redundant search pair.",
  [("before_action", "Red Dawn (2012 film)", "based on the 1984 film of the same name"), ("before_action", "Red Dawn (2012 film)", "film stars Chris Hemsworth")]),
 (128, "What movie stars Chris Hemsworth", "repair_candidate", ["paired_unnecessary_search", "answer_supported_before_search"],
  "Red Dawn is the supported film, and both the actor and earlier-film basis were visible before search. The frozen answer is correct, but the paired search action was unnecessary.",
  "Use as a direct-answer example only after replay and independent review.",
  [("before_action", "Red Dawn (2012 film)", "based on the 1984 film of the same name"), ("before_action", "Red Dawn (2012 film)", "film stars Chris Hemsworth")]),
 (129, "Which seminal figure of the Romantic Age", "retain_candidate", ["query_uses_visible_romantic_age_descriptor"],
  "The initial Blake article explicitly calls him a seminal figure of the Romantic Age. Searching his name retrieves Robert Bolt's play article, which supplies the missing title-inspiration relation.", None,
  [("before_action", "William Blake", "seminal figure in the history of the poetry and visual arts of the Romantic Age"), ("retrieved_after_search", "The Tiger and the Horse", "takes its title from William Blake's")]),
 (130, "Which seminal figure of the Romantic Age", "retain_candidate", [],
  "Robert Bolt's play article says its title comes from William Blake's work, and Blake's biography matches the Romantic Age descriptor. The answer is supported.", None,
  [("before_action", "William Blake", "seminal figure in the history of the poetry and visual arts of the Romantic Age"), ("before_action", "The Tiger and the Horse", "takes its title from William Blake's")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-058.jsonl", cohort="v122_internal_candidates")
