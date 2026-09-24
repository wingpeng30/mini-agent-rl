from mini_agent_rl.model.local import LocalModelConfig, TransformersModelClient
from mini_agent_rl.training.harness import preflight


def test_context_loss_counters_use_usage_not_document_detail_names():
    client = TransformersModelClient(LocalModelConfig())
    client._context_records = [
        {"phase": "rollout_sampling", "lossless_expression_changed": True},
        {"phase": "rollout_sampling", "lossy_body_trimmed": True,
         "documents_deleted": 2, "document_changes": [{"body_trimmed": True, "deleted": True}]},
    ]
    counts = client.context_usage_summary()["phases"]["rollout_sampling"]
    assert counts["lossless_expression_changed"] == 1
    assert counts["lossy_body_trimmed"] == 1
    assert counts["documents_deleted"] == 2


def test_harness_allows_long_and_unlimited_runs(tmp_path):
    for seconds in (0, 21600):
        config = tmp_path / f"run-{seconds}.toml"
        config.write_text(f'[experiment]\nrun_dir="new-{seconds}"\ntimeout_seconds={seconds}\n[command]\nargv=["python","--version"]\n')
        assert preflight(config)["passed"]
