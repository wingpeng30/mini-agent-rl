from scripts.v128_reconcile_deepseek_usage_csv import (
    cost_usage, masked_key_match, request_usage, token_usage,
)


def test_masked_key_match_reports_only_match_facts():
    result = masked_key_match("abcdefghABCDEFGHijklmnop1234", "abcdefgh********ijklmnop****")
    assert result["csv_key_is_masked"] is True
    assert result["key_lengths_equal"] is True
    assert result["visible_character_count"] == 16
    assert result["all_visible_characters_match"] is True
    assert "key" not in result


def test_daily_request_and_token_aggregates_do_not_emit_api_key():
    rows = [
        {"start_time_iso":"2026-09-26T00:00:00+08:00", "api_key_name":"AgentLightning",
         "api_key":"sk-hidden", "model":"deepseek-flash", "type":"request_count", "amount":"3"},
        {"start_time_iso":"2026-09-26T00:00:00+08:00", "api_key_name":"AgentLightning",
         "api_key":"sk-hidden", "model":"deepseek-flash", "type":"request_count", "amount":"4"},
        {"start_time_iso":"2026-09-26T00:00:00+08:00", "api_key_name":"AgentLightning",
         "api_key":"sk-hidden", "type":"output_tokens", "amount":"15"},
    ]
    requests = request_usage(rows)
    tokens = token_usage(rows)
    assert requests == [{"start_time_iso":"2026-09-26T00:00:00+08:00",
                         "api_key_name":"AgentLightning", "model":"deepseek-flash",
                         "request_count":7}]
    assert tokens[0]["tokens"] == 15
    assert "sk-hidden" not in str(requests) + str(tokens)
    assert all("api_key" not in row for row in requests + tokens)


def test_costs_are_daily_and_account_level():
    rows = [
        {"start_time_iso":"2026-09-26T00:00:00+08:00", "wallet_type":"Paid",
         "currency":"CNY", "cost":"1.25"},
        {"start_time_iso":"2026-09-26T00:00:00+08:00", "wallet_type":"Paid",
         "currency":"CNY", "cost":"0.75"},
    ]
    assert cost_usage(rows) == [{"start_time_iso":"2026-09-26T00:00:00+08:00",
                                 "wallet_type":"Paid", "currency":"CNY", "cost":"2.00"}]
