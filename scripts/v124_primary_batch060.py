"""Primary examination of v122 internal candidate states 141–150."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (141, "What was Kate Garvey's husband's nickname", "repair_candidate", ["unanchored_spouse_relation"],
  "The Kate Garvey source names Jimmy Wales as her husband, and the retrieved Wales biography gives Jimbo as his nickname. The relation is supported in this frozen snapshot, but the question does not time-anchor a marital status that can change.",
  "Ask for the nickname of Jimmy Wales, whom the cited source identified as Kate Garvey's husband, or otherwise anchor the source period; replay the pair.",
  [("before_action", "Kate Garvey", "married to Wikipedia co-founder Jimmy Wales"), ("retrieved_after_search", "Jimmy Wales", "also known as Jimbo Wales")]),
 (142, "What was Kate Garvey's husband's nickname", "repair_candidate", ["paired_unanchored_spouse_relation"],
  "Jimbo is supported as Jimmy Wales's nickname, and the snapshot links him to Garvey. The unanchored husband reference should be made historically explicit before the pair is eligible.",
  "Retain the supported nickname only with the anchored question and replayed pair.",
  [("before_action", "Kate Garvey", "married to Wikipedia co-founder Jimmy Wales"), ("before_action", "Jimmy Wales", "also known as Jimbo Wales")]),
 (143, " Four Quartets is a set", "retain_candidate", ["query_copies_visible_author_relation"],
  "Four Quartets names T. S. Eliot as author. The search follows that visible author relation and retrieves Eliot's biography with his 1888 birth year.", None,
  [("before_action", "Four Quartets", "written by T. S. Eliot"), ("retrieved_after_search", "T. S. Eliot", "26 September 1888")]),
 (144, " Four Quartets is a set", "retain_candidate", [],
  "The work article names T. S. Eliot and his biography gives 26 September 1888. The year is supported.", None,
  [("before_action", "Four Quartets", "written by T. S. Eliot"), ("before_action", "T. S. Eliot", "26 September 1888")]),
 (145, "What nationality was the male star", "retain_candidate", ["query_copies_visible_cast_relation"],
  "The Scarlet Angel snippet stars Yvonne de Carlo and Rock Hudson, identifying the male actor. The query follows the cast relation and retrieves Hudson's biography, which calls him American.", None,
  [("before_action", "Scarlet Angel", "starring Yvonne de Carlo and Rock Hudson"), ("retrieved_after_search", "Rock Hudson", "was an American actor")]),
 (146, "What nationality was the male star", "retain_candidate", [],
  "Rock Hudson is the film's male star in the visible cast line; his biography explicitly calls him an American actor. The answer is supported.", None,
  [("before_action", "Scarlet Angel", "starring Yvonne de Carlo and Rock Hudson"), ("before_action", "Rock Hudson", "was an American actor")]),
 (147, "Davison's was the major competition", "retain_candidate", ["query_copies_visible_competitor_relation"],
  "Davison's is said to compete with Rich's. The query copies that relation and retrieves Rich's article giving its headquarters in Atlanta, Georgia.", None,
  [("before_action", "Davison's", "major competition to Rich's"), ("retrieved_after_search", "Rich's (department store)", "headquartered in Atlanta, Georgia")]),
 (148, "Davison's was the major competition", "retain_candidate", [],
  "The competing retail chain is Rich's, whose article directly states Atlanta, Georgia as headquarters. The answer includes the requested city with state disambiguation.", None,
  [("before_action", "Davison's", "major competition to Rich's"), ("before_action", "Rich's (department store)", "headquartered in Atlanta, Georgia")]),
 (149, "Robert Bozic fought and lost", "retain_candidate", ["query_copies_visible_opponent_relation"],
  "Bob Bozic's biography identifies Larry Holmes as the boxer he lost to. The search follows that named opponent and retrieves Holmes's Easton Assassin nickname.", None,
  [("before_action", "Bob Bozic", "fought and lost to Larry Holmes"), ("retrieved_after_search", "Larry Holmes", "nickname of the \"Easton Assassin\"")]),
 (150, "Robert Bozic fought and lost", "retain_candidate", [],
  "Bozic's opponent is Larry Holmes, and Holmes's biography supplies Easton Assassin. The nickname is supported.", None,
  [("before_action", "Bob Bozic", "fought and lost to Larry Holmes"), ("before_action", "Larry Holmes", "nickname of the \"Easton Assassin\"")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-060.jsonl", cohort="v122_internal_candidates")
