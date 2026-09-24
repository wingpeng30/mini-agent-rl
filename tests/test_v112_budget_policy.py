import json
from pathlib import Path
import pytest
from mini_agent_rl.evidence_sft_v112 import load_budget_policy


def test_no_amendment_keeps_prior_caps(tmp_path):
    policy=load_budget_policy(tmp_path)
    assert policy['cumulative_cap_cny']==39.600696
    assert policy['new_cap_cny']==12


def test_explicit_uncapped_authorization_is_fingerprinted(tmp_path):
    path=tmp_path/'configs/v112-budget-policy.json';path.parent.mkdir()
    path.write_text(json.dumps({'version':'v112','mode':'uncapped','authorization':'User instruction',
                               'cumulative_cap_cny':None,'new_cap_cny':None}))
    policy=load_budget_policy(tmp_path)
    assert policy['mode']=='uncapped' and policy['sha256']
    assert policy['new_cap_cny'] is None


def test_missing_authorization_fails_closed(tmp_path):
    path=tmp_path/'configs/v112-budget-policy.json';path.parent.mkdir()
    path.write_text('{"version":"v112","mode":"uncapped"}')
    with pytest.raises(ValueError,match='authorization'):
        load_budget_policy(tmp_path)
