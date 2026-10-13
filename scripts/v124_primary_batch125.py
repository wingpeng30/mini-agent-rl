"""Primary examination of v123 internal candidate states 241–246."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (241, "What was the first professional short of the director", "repair_candidate", ["director_preselected_without_work_link", "work_format_underspecified"],
  "The initial Tom Hooper article supplies Painted Faces, but no initial passage connects him to Daniel Deronda. Searching Tom Hooper preselects the director before the serial-director link appears, and the question leaves Daniel Deronda's TV adaptation implicit beside the novel of the same name.",
  "Name the 2002 Daniel Deronda TV serial and search its director first; then use the short-film source.",
  [("before_action", "Tom Hooper", "first professional short, \"Painted Faces\""), ("retrieved_after_search", "Daniel Deronda (TV serial)", "directed by Tom Hooper")]),
 (242, "What was the first professional short of the director", "repair_candidate", ["paired_director_preselection", "answer_supported"],
  "Painted Faces is documented as Tom Hooper's first professional short, and the retrieved TV-serial passage establishes him as its director. The paired query selected him before that relationship was visible.",
  "Review after a Daniel Deronda director query replay and title disambiguation.",
  [("before_action", "Tom Hooper", "first professional short, \"Painted Faces\""), ("before_action", "Daniel Deronda (TV serial)", "directed by Tom Hooper")]),
 (243, "What computer magazine uses Alex Hughes", "repair_candidate", ["answer_visible_before_search", "query_copies_answer_bearing_sentence"],
  "Alex Hughes's initial biography already says his work has been used in PC Pro. The query copies that full sentence, and retrieval merely verifies that PC Pro is a computer magazine. The search does not discover which publication used his work.",
  "Either answer directly if PC Pro's category is sufficiently known, or search PC Pro's magazine type explicitly.",
  [("before_action", "Alex Hughes", "used in PC Pro"), ("retrieved_after_search", "PC Pro", "computer magazines")]),
 (244, "What computer magazine uses Alex Hughes", "repair_candidate", ["paired_redundant_search"],
  "PC Pro is visible in the original Alex Hughes biography, and its retrieved article confirms it is a computer magazine. The answer is supported, but the paired search copies the complete answer-bearing sentence.",
  "Rebuild as direct answer or explicit magazine-category verification.",
  [("before_action", "Alex Hughes", "used in PC Pro"), ("before_action", "PC Pro", "computer magazines")]),
 (245, "What is the nationality of the composer", "retain_candidate", ["query_uses_visible_work", "retrieval_adds_nationality"],
  "The initial Drumming article names Steve Reich as composer. Retrieval of his biography explicitly identifies him as an American composer. The work-title query retrieves the missing nationality without substituting one of the unrelated drummer articles.", None,
  [("before_action", "Drumming (Reich)", "piece by minimalist composer Steve Reich"), ("retrieved_after_search", "Steve Reich", "American composer")]),
 (246, "What is the nationality of the composer", "retain_candidate", [],
  "Steve Reich is the named composer of Drumming, and his retrieved biography states he is an American composer. The answer is supported.", None,
  [("before_action", "Drumming (Reich)", "piece by minimalist composer Steve Reich"), ("before_action", "Steve Reich", "American composer")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-125.jsonl", cohort="v123_internal_candidates")
