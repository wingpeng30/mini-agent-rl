# Goal AN seed42

探索通过：False。未发布。

```json
{
  "seed": 42,
  "route_passed": false,
  "gates": {
    "supported_at_least_23": false,
    "complete_at_least_22": false,
    "relative_supported_gain_two": false,
    "relative_complete_gain_two": false,
    "new_binding_repair": false,
    "errors_not_higher": true,
    "action_cases_not_higher": false,
    "parse_not_higher": true,
    "repeat_not_higher": true,
    "overbudget_not_higher": false,
    "protected_loss_at_most_one": true,
    "AM_gains_preserved": false,
    "partial_premature_not_higher": true,
    "partial_invalid_not_higher": true,
    "partial_repeat_not_higher": true,
    "partial_legal_search_not_lower": true
  },
  "scores": {
    "42-A": {
      "supported": 21,
      "success_ids": [
        "2wiki:3a3d60020bdd11eba7f7acde48001122",
        "2wiki:49d965aa0bde11eba7f7acde48001122",
        "2wiki:5dac1e8e0bdd11eba7f7acde48001122",
        "2wiki:67ecf3100bdc11eba7f7acde48001122",
        "2wiki:764039340bde11eba7f7acde48001122",
        "2wiki:7bda387a0bdb11eba7f7acde48001122",
        "2wiki:982f8e440bdb11eba7f7acde48001122",
        "2wiki:ee3233920bda11eba7f7acde48001122",
        "2wiki:ef9d40a40bdb11eba7f7acde48001122",
        "2wiki:f299dd600bdd11eba7f7acde48001122",
        "musique:2hop__109242_113442",
        "musique:2hop__151750_141308",
        "musique:2hop__158275_35739",
        "musique:2hop__418360_67370",
        "musique:2hop__435087_55984",
        "musique:2hop__478978_124498",
        "musique:2hop__506035_325512",
        "musique:2hop__530579_13529",
        "musique:2hop__69765_43945",
        "musique:2hop__796221_18803",
        "musique:2hop__864101_135373"
      ],
      "errors": 7,
      "complete": 20,
      "binding_repairs": 0,
      "protected_lost": [],
      "must_retain_lost": [],
      "premature": 0,
      "invalid": 0,
      "repeated": 0,
      "legal_search": 7,
      "categories": {
        "success": 21,
        "unsupported_by_visible_evidence": 3,
        "factual_error": 4,
        "incomplete_required_qualifier": 1
      },
      "complete_chain_visible": 26
    },
    "42-B": {
      "supported": 20,
      "success_ids": [
        "2wiki:3a3d60020bdd11eba7f7acde48001122",
        "2wiki:49d965aa0bde11eba7f7acde48001122",
        "2wiki:5dac1e8e0bdd11eba7f7acde48001122",
        "2wiki:67ecf3100bdc11eba7f7acde48001122",
        "2wiki:764039340bde11eba7f7acde48001122",
        "2wiki:7bda387a0bdb11eba7f7acde48001122",
        "2wiki:982f8e440bdb11eba7f7acde48001122",
        "2wiki:ee3233920bda11eba7f7acde48001122",
        "2wiki:ef9d40a40bdb11eba7f7acde48001122",
        "2wiki:f299dd600bdd11eba7f7acde48001122",
        "musique:2hop__109242_113442",
        "musique:2hop__151750_141308",
        "musique:2hop__158275_35739",
        "musique:2hop__418360_67370",
        "musique:2hop__435087_55984",
        "musique:2hop__478978_124498",
        "musique:2hop__506035_325512",
        "musique:2hop__530579_13529",
        "musique:2hop__69765_43945",
        "musique:2hop__864101_135373"
      ],
      "errors": 7,
      "complete": 20,
      "binding_repairs": 0,
      "protected_lost": [],
      "must_retain_lost": [
        "musique:2hop__796221_18803"
      ],
      "premature": 0,
      "invalid": 0,
      "repeated": 0,
      "legal_search": 7,
      "categories": {
        "success": 20,
        "unsupported_by_visible_evidence": 3,
        "factual_error": 4,
        "incomplete_required_qualifier": 2
      },
      "complete_chain_visible": 26
    }
  },
  "new_binding_repairs": 0,
  "metrics": {
    "42-A": {
      "EM": 26,
      "F1": 0.5339058510933511,
      "body_trimmed": 0,
      "calls": 192,
      "cases": 64,
      "invalid_or_repeated": 1,
      "length_stops": 0,
      "max_input": 2248,
      "max_output": 30,
      "mean_generation_seconds": 0.662820512157244,
      "mean_question_seconds": 2.0464525491697714,
      "overbudget_search_attempts": 0,
      "parse_invalid": 0,
      "repeated": 1,
      "searches": 128
    },
    "42-B": {
      "EM": 24,
      "F1": 0.5031137265512265,
      "body_trimmed": 0,
      "calls": 194,
      "cases": 64,
      "invalid_or_repeated": 2,
      "length_stops": 0,
      "max_input": 2248,
      "max_output": 30,
      "mean_generation_seconds": 0.7830762157466301,
      "mean_question_seconds": 2.4283524666097946,
      "overbudget_search_attempts": 1,
      "parse_invalid": 0,
      "repeated": 1,
      "searches": 130
    }
  },
  "released": false
}
```
