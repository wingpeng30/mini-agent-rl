"""Primary case examination of original training rows 271–280."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (271, "When is American actress who starred", "repair_candidate", ["duplicate_training_action"],
  "Fourth duplicate of the Gentlemen Broncos cast-list query after rows 268–270. It retrieves Jennifer Coolidge's same birth date and adds no independent target.",
  "Preserve frozen row, deduplicate the path and verify the paired answer.",
  [("before_action", "Gentlemen Broncos", "Jennifer Coolidge"), ("retrieved_after_search", "Jennifer Coolidge", "born August 28, 1963")]),
 (272, "What English playwright born in 1572", "retain_candidate", ["query_known_candidate", "duplicate_training_action_group"],
  "Jonson's 1572 birth and playwright role are visible, but his link to Guy Earl of Warwick is not. The name query retrieves the play article, which says he is a proposed author while authorship is unknown. The question's 'possible author' wording accurately preserves that uncertainty.", None,
  [("before_action", "Ben Jonson", "c. 11 June 1572"), ("retrieved_after_search", "The Tragical History of Guy Earl of Warwick", "author of \"Guy Earl of Warwick\" is not known, although Ben Jonson and Thomas Dekker have been proposed")]),
 (273, "What English playwright born in 1572", "repair_candidate", ["duplicate_training_action"],
  "Same Jonson query and proposed-authorship result as row 272, with irrelevant pre-action distractors. It is duplicate supervision, not a second authorship confirmation.",
  "Deduplicate after the answer state is checked; retain the play's explicit uncertainty.",
  [("before_action", "Ben Jonson", "c. 11 June 1572"), ("retrieved_after_search", "The Tragical History of Guy Earl of Warwick", "Ben Jonson and Thomas Dekker have been proposed")]),
 (274, "When was the American actress and model", "retain_candidate", ["query_known_candidate_verifies_other_film"],
  "Johnson's birthday and Crazy in Alabama debut are visible before search. The name query retrieves the Fifty Shades Freed film article naming Dakota Johnson, completing the two-film identity check. The search is verification rather than discovery of the already-visible date.", None,
  [("before_action", "Dakota Johnson", "born October 4, 1989"), ("before_action", "Dakota Johnson", "Crazy in Alabama"), ("retrieved_after_search", "Fifty Shades Freed (film)", "stars Dakota Johnson and Jamie Dornan")]),
 (275, "Paulo Coelho is best known", "retain_candidate", ["query_copies_visible_context"],
  "Coelho's excerpt names The Alchemist as the widely translated novel. Searching that visible sentence retrieves the novel article with its first-publication year 1988; the year, rather than an unrelated Coelho book, is supported.", None,
  [("before_action", "Paulo Coelho", "widely translated novel \"The Alchemist\""), ("retrieved_after_search", "The Alchemist (novel)", "first published in 1988")]),
 (276, "When did the person that reign", "retain_candidate", ["question_grammar"],
  "The Book of Common Prayer article dates the original publication to 1549 during Edward VI's reign. His biography says he became king on January 28, 1547. The grammar is poor but the requested date and two-hop identity are unambiguous here.", None,
  [("before_action", "Book of Common Prayer", "original book, published in 1549 in the reign of Edward VI"), ("before_action", "Edward VI of England", "King of England and Ireland from 28 January 1547")]),
 (277, "The Kentucky Democratic Presidential Primary", "retain_candidate", ["query_targets_known_districts", "duplicate_training_action_group"],
  "The pre-action districts article only establishes six districts and does not date the primary. Searching its title retrieves the 2008 Kentucky Democratic primary article, which includes the distinctive 34-district-delegate rule and May 20, 2008 date. The observed retrieval succeeds despite a query aimed at a known clue.", None,
  [("before_action", "Kentucky's congressional districts", "divided into 6 congressional districts"), ("retrieved_after_search", "Kentucky Democratic primary, 2008", "took place May 20, 2008"), ("retrieved_after_search", "Kentucky Democratic primary, 2008", "district's delegates, totaling 34")]),
 (278, "The Kentucky Democratic Presidential Primary", "repair_candidate", ["duplicate_training_action"],
  "Same question, Kentucky-districts query and 2008-primary result as row 277. Distractor differences do not add a separate date target.",
  "Deduplicate after paired answer review; preserve original row and date evidence.",
  [("before_action", "Kentucky's congressional districts", "divided into 6 congressional districts"), ("retrieved_after_search", "Kentucky Democratic primary, 2008", "took place May 20, 2008")]),
 (279, "The Kentucky Democratic Presidential Primary", "repair_candidate", ["duplicate_training_action"],
  "Third identical district-title query returns the same Kentucky primary date and 34-delegate rule.",
  "Preserve input, count only a justified representative after question-level deduplication.",
  [("before_action", "Kentucky's congressional districts", "divided into 6 congressional districts"), ("retrieved_after_search", "Kentucky Democratic primary, 2008", "took place May 20, 2008")]),
 (280, "The Kentucky Democratic Presidential Primary", "repair_candidate", ["duplicate_training_action"],
  "Fourth same-query copy; the date evidence is unchanged. Repeated target retrieval should not multiply independent training coverage.",
  "Keep original row for traceability and deduplicate the group only in a separately replayed export.",
  [("before_action", "Kentucky's congressional districts", "divided into 6 congressional districts"), ("retrieved_after_search", "Kentucky Democratic primary, 2008", "took place May 20, 2008")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-028.jsonl")
