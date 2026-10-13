"""Primary case examination of original training rows 311–320."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (311, "The InterContinental Bucharest", "repair_candidate", ["ambiguous_plural_brand_scope", "duplicate_training_action"],
  "Same IHG operator sentence and group-brand list as row 310. The question does not specify whether it seeks the operator or which subset of other IHG brands, and this repeated search does not resolve the scope.",
  "Inspect the paired answer and source question before designing one precise, independently reviewed operator or brand-list target.",
  [("before_action", "InterContinental Bucharest", "operated by InterContinental Hotels Group"), ("retrieved_after_search", "InterContinental Hotels Group", "Its brands include Candlewood Suites, Crowne Plaza, EVEN Hotels, Holiday Inn")]),
 (312, "What is Jayson Anthony Paul", "retain_candidate", ["query_restates_question", "duplicate_training_action_group"],
  "The visible tag-team article lists JTG and Shad Gaspard but does not map the legal name. The query includes Jayson Anthony Paul from the question; retrieval of the JTG biography explicitly maps that name to ring name JTG.", None,
  [("before_action", "Cryme Tyme", "consisting of JTG and Shad Gaspard"), ("retrieved_after_search", "JTG", "Jayson Anthony Paul (born December 10, 1984), better known by the ring name JTG")]),
 (313, "What is Jayson Anthony Paul", "repair_candidate", ["duplicate_training_action"],
  "Same question, query and JTG name mapping as row 312 with fewer distractors. It is duplicate supervision.",
  "Keep original for audit and select at most one path after paired answer check.",
  [("before_action", "Cryme Tyme", "consisting of JTG and Shad Gaspard"), ("retrieved_after_search", "JTG", "better known by the ring name JTG")]),
 (314, "What kinds of citizenship", "retain_candidate", ["query_known_person_verifies_film"],
  "Seagal's American, Russian and Serbian citizenship is visible before the action; the name search retrieves the Kill Switch film article confirming he starred in it. This completes the identity link rather than discovering the already-visible citizenships.", None,
  [("before_action", "Steven Seagal", "holds American, Russian, and Serbian citizenship"), ("retrieved_after_search", "Kill Switch (2008 film)", "starring Steven Seagal")]),
 (315, "Khet and Quarto", "retain_candidate", ["query_known_game_fills_other", "duplicate_training_action_group"],
  "Khet is already identified as a chess-like abstract strategy board game. Querying its title retrieves Quarto as a board game, making the shared board-game category supported, although the Khet query is aimed at the known game.", None,
  [("before_action", "Khet (game)", "abstract strategy board game"), ("retrieved_after_search", "Quarto (board game)", "Quarto is a board game for two players")]),
 (316, "Khet and Quarto", "repair_candidate", ["duplicate_training_action"],
  "Exact duplicate of row 315's question, query and retrieved Quarto category.",
  "Preserve row, deduplicate after paired answer review.",
  [("before_action", "Khet (game)", "abstract strategy board game"), ("retrieved_after_search", "Quarto (board game)", "Quarto is a board game for two players")]),
 (317, "Khet and Quarto", "retain_candidate", ["query_known_game_fills_other", "duplicate_training_action_group"],
  "Quarto is known to be a board game before search. Querying its title retrieves Khet as an abstract strategy board game, completing the category join from the reverse evidence state.", None,
  [("before_action", "Quarto (board game)", "Quarto is a board game for two players"), ("retrieved_after_search", "Khet (game)", "abstract strategy board game")]),
 (318, "Khet and Quarto", "repair_candidate", ["duplicate_training_action"],
  "Exact duplicate of row 317's reverse-evidence query and Khet category result.",
  "Retain frozen evidence but exclude the redundant copy from any corrected export.",
  [("before_action", "Quarto (board game)", "Quarto is a board game for two players"), ("retrieved_after_search", "Khet (game)", "abstract strategy board game")]),
 (319, "Prudential Center in Newark", "retain_candidate", [],
  "The arena article identifies Seton Hall Pirates as a home basketball team. The program article locates Seton Hall University in South Orange, New Jersey and repeats the arena link. The requested town is supported.", None,
  [("before_action", "Prudential Center", "home of the New Jersey Devils of the National Hockey League (NHL) and the Seton Hall Pirates men's basketball team"), ("before_action", "Seton Hall Pirates men's basketball", "Seton Hall University in South Orange, New Jersey")]),
 (320, "Prudential Center in Newark", "repair_candidate", ["generic_geographic_query", "retrieval_hit_not_strategy_proof"],
  "From empty context, the query 'Jersey' is grounded only in a broad state name and retrieves many unrelated New Jersey entities alongside the correct Seton Hall program. The program excerpt supplies South Orange, but the one-word query provides no mechanism for targeting the arena-team relation; this hit should not be used as strong search supervision.",
  "Keep the observed result as diagnostic evidence; test a question-derived arena/team query in an isolated replay before proposing a replacement search label.",
  [("retrieved_after_search", "Seton Hall Pirates men's basketball", "Seton Hall University in South Orange, New Jersey"), ("retrieved_after_search", "Seton Hall Pirates men's basketball", "plays their home games in the Prudential Center in Newark, New Jersey")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-032.jsonl")
