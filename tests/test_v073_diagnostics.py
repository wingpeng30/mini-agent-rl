from mini_agent_rl.evaluation import answer_metrics, token_f1


def test_normalized_answer_metrics_ignore_articles_and_punctuation():
    metrics = answer_metrics("The, City!", "city")
    assert metrics["strict_exact_match"] == 0.0
    assert metrics["normalized_exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0


def test_token_f1_partial_and_empty_cases():
    assert token_f1("red blue", "red green") == 0.5
    assert token_f1("", "") == 1.0
    assert token_f1("", "answer") == 0.0
