"""Primary examination of v122 training-supplement candidate states 211–212."""
from scripts.v124_primary_batch001 import save_batch

SPECS = [
 (211, "Who wrote the theatrical comedy referenced", "retain_candidate", ["query_uses_visible_subtitle", "retrieval_adds_tartuffe_author"],
  "The initial Guilty Mother excerpt supplies the subtitle The Other Tartuffe. The query repeats that visible subtitle and retrieves the Tartuffe article, which calls the original theatrical comedy a work by Molière. The source also names Beaumarchais as author of The Guilty Mother, a distinct role not to confuse with Tartuffe's author.", None,
  [("before_action", "The Guilty Mother", 'subtitled "The Other Tartuffe"'), ("retrieved_after_search", "Tartuffe", "theatrical comedies by Molière")]),
 (212, "Who wrote the theatrical comedy referenced", "retain_candidate", [],
  "The Guilty Mother subtitle refers to Tartuffe, and the retrieved Tartuffe article attributes that theatrical comedy to Molière. The answer does not mistakenly use the separate playwright Pierre Beaumarchais.", None,
  [("before_action", "The Guilty Mother", 'subtitled "The Other Tartuffe"'), ("before_action", "Tartuffe", "theatrical comedies by Molière")]),
]

if __name__ == "__main__":
 save_batch(SPECS, "primary-batch-100.jsonl", cohort="v122_train_candidates")
