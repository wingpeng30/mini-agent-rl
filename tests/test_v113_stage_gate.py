import json
import unittest
from pathlib import Path

from scripts.check_v113_stage_gate import decide


class V113StageGateTests(unittest.TestCase):
    def test_current_zero_new_questions_blocks_p2_p3_and_p4(self):
        gate = json.loads(Path("reports/v113-p2-source-gate.json").read_text(encoding="utf-8"))
        for stage in ("p2", "p3", "p4"):
            result = decide(stage, gate)
            self.assertFalse(result["allowed"])
            self.assertTrue(any("source gate" in reason for reason in result["reasons"]))

    def test_p3_requires_all_internal_acceptance_gates(self):
        source = {"passed": True, "eligible_new_questions_after_historical_exclusion": 50,
                  "required_new_internal_questions": 50}
        self.assertFalse(decide("p3", source)["allowed"])
        self.assertTrue(decide("p3", source, {"all_internal_gates_passed": True})["allowed"])

    def test_p4_requires_frozen_internal_success_scheme(self):
        source = {"passed": True, "eligible_new_questions_after_historical_exclusion": 50,
                  "required_new_internal_questions": 50}
        p3 = {"all_internal_gates_passed": True}
        self.assertFalse(decide("p4", source, p3)["allowed"])
        self.assertTrue(decide("p4", source, p3, {"internal_scheme_frozen": True})["allowed"])


if __name__ == "__main__":
    unittest.main()
