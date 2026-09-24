# v114 isolation incident and clean rebuild

During the initial historical-exposure scan, the deny rule matched a directory named `final-test` but did not match a file named `final-test.jsonl`. The scan read records from `data/hotpot-agent-v072/final-test.jsonl` and included their IDs/questions in its historical exclusion set. No scoring or evaluation was run, but this was an access-policy violation.

All freezes, candidate states, and counts under `data/evidence-sft-v114/` are quarantined and must not be used. The builder was corrected to reject any path component containing `final-test`, `final_test`, `holdout`, or `independent100`, and to ignore all previous v114 outputs. A clean rebuild is being created under `data/evidence-sft-v114-clean/`; no further reads of the prohibited file are part of that build.

The clean build record distinguishes its own access scope from the incident history. The independent100 content remains unread and unevaluated. The final-test file was read during the initial attempt, so this incident is retained in the project record.
