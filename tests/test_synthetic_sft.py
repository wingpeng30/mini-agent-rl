import json

from mini_agent_rl.training.synthetic import generate_dataset, generate_split, validate_record


def test_synthetic_records_are_valid_and_deterministic():
    corpus = [{"id": "d1", "title": "北京", "text": "北京是中国的首都。"}]
    a = generate_split(corpus, 3, "train", seed=7)
    b = generate_split(corpus, 3, "train", seed=7)
    assert a == b
    assert all(validate_record(item) == [] for item in a)
    assert json.loads(a[0]["messages"][2]["content"])["action"] == "search"
    assert json.loads(a[0]["messages"][4]["content"])["action"] == "answer"


def test_generate_dataset_writes_manifest(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps([{"id": "d1", "title": "北京", "text": "北京是中国的首都。"}], ensure_ascii=False), encoding="utf-8")
    result = generate_dataset(corpus_path, tmp_path / "sft", train=2, validation=1, test=1)
    assert result["counts"] == {"train": 2, "validation": 1, "test": 1}
    assert (tmp_path / "sft" / "manifest.json").exists()
