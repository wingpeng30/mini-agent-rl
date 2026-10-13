import pytest

from scripts.v130_audit_blind_votes import _saved_vote_count


def test_saved_vote_count_uses_actual_review_rows_not_fixed_batch_size():
    assert _saved_vote_count({"reviews": [{"case_id": "a"}, {"case_id": "b"}]}) == 2


def test_saved_vote_count_rejects_malformed_vote_container():
    with pytest.raises(ValueError, match="saved_vote_reviews_not_array"):
        _saved_vote_count({"reviews": None})
