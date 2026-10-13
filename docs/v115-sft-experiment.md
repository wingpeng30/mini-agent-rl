# v115: evidence-constrained SFT experiment

## Preregistration and scope

This run tests whether one small evidence-constrained SFT adapter improves the local v101 agent on the frozen v114 internal set. It preserves v101 as the primary comparator and v112 as a reference only. The v111 78% figure remains a training-side API-assisted hybrid result and is not a local-model score.

The training export contains 300 states from 182 training-source questions. The internal export contains 100 states over the frozen 50-question list. These states are programmatically constructed from the reviewed HotpotQA training source; they are not claimed to be authentic model trajectories. Only the training export enters the trainer. Internal labels, questions and task data are used only for local evaluation and are not used for gradient updates.

The v114 review gate and export hashes must pass before execution. The expected export SHA-256 values are `56c32bf103ec4fd51e800c50eec4a64efd9233160729bf00712e9ccec407872f` for train and `9e40e3cd996cabb99ebaad96bf5e033e5a922e360e44faac932df6ee6da59b8c` for internal. Frozen v101 adapter SHA-256 is `3b759be69a0c082fd5274eb99b696bf6a60662b991c2234aac59d0646ff08768`.

The internal retrieval corpus is built only from the 50 frozen questions' official training-source contexts. Documents are deduplicated by exact title and text; all three agents use the same 492-document BM25-v2 corpus, top 3 retrieval, direct-v090 prompt, compact-v3 observations, 2,048 input-token limit, 128 output-token limit, temperature 0, seed 42 and at most three tool steps. Evaluation processes set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTHONHASHSEED=42` before importing the model runtime; the evaluator rejects runs missing either setting.

Local and AutoDL tokenizers must produce equal per-record hashes for prompt IDs, complete sequence IDs and the assistant-only target mask. Maximum allowed sizes are 2,048 prompt tokens, 128 target tokens and 2,176 total tokens. A mismatch blocks training.

Training consists of a five-update smoke phase and one 50-update formal phase, each copied afresh from the frozen v101 adapter. Settings: BF16, learning rate `5e-6`, batch size 1, accumulation 8, seed 42, no thinking, no gradient checkpointing. The final checkpoint is used; no intermediate checkpoint selection or configuration search is allowed.

## Internal acceptance gates

The new adapter is compared against v101 on the complete 100-state and 50-question internal sets. v112 is retained as a descriptive reference. Acceptance requires all of the following:

- Protocol validity at least 98%.
- Supported-answer correctness at least 90%.
- Search on insufficient-evidence states at least 90%.
- At least 95% retention of v101's previously correct task answers.
- Net gain of at least three correct answers across 50 questions.
- Mean token F1 no lower than v101.
- Two independent blinded semantic audits with no unresolved support dispute or confirmed regression.

If any numerical or semantic gate fails, the run stops. Internal errors do not authorize additional training. The independent100 set is not opened before all internal gates pass.

## Conditional independent validation

Only after internal acceptance will the complete independent 100-question set be opened, with v101 and the v115 final checkpoint evaluated on the frozen 986-document environment. No interim-score stopping or tuning is permitted. Required gates are net correct gain of at least three, no F1 decrease, at least 95% retention, protocol at least 98%, and no unresolved semantic issue or confirmed regression. Question-paired bootstrap uses seed 42 and 10,000 draws; the EM-difference interval's lower bound must exceed zero. A pass authorizes only a GRPO proposal, not GRPO execution.

## Execution record

| Item | Status | Evidence |
|---|---|---|
| v114 review and export reconstruction | Passed | `reports/v114-review-gate.json`, `reports/v114-sft-export-manifest.json` |
| Frozen internal set | Passed | v3 freeze: 50 questions; selected export: 100 states |
| Internal corpus | Passed | 492 exact-deduplicated documents; manifest at `data/evidence-sft-v115/internal-environment-manifest.json` |
| Local token audit | Passed | 300 train + 100 internal; maxima input 1,673/1,366, target 74/72, total 1,691/1,377 |
| AutoDL token equivalence | Passed | Remote dry-run matched local prompt, sequence and target-mask hashes for every record |
| AutoDL / source checkpoints | Passed preflight | RTX 4090 D, 24,564 MiB; v101 and v112 adapter hashes match frozen values |
| Initial internal evaluation attempt | Superseded | Preserved but excluded from the primary result after CUDA reported a nondeterministic CuBLAS operation; all three agents were rerun under the fixed deterministic environment |
| Smoke training | Passed | 5 optimizer updates; finite loss/gradients; adapter reload passed; adapter SHA `bf02469dff9714b0110b9650a9fd2461ce05e972d0bc2b8b6ed66d92d1080d07`; peak allocated GPU memory 10.862 GiB |
| Formal training | Passed | Fresh v101 copy; 50 optimizer updates; final `checkpoint-50` reload passed; adapter SHA `1717d28e37c4fc9ab14a9ed48dbed2eb6de95958d74f7151a9905176b5b5c76e`; peak allocated GPU memory 14.048 GiB; training loss 0.4642; source v101 hash unchanged |
| Deterministic internal three-model evaluation | Complete | Full 100-state/50-task coverage for v101, v112 and v115 on the same corpus, prompt and decode settings; CUDA nondeterminism warning absent |
| v101 vs v115 task result | Numeric gate failed | EM `28/50 → 31/50` (+3 net: 4 gains, 1 loss); F1 `0.6205 → 0.6785`; retention `27/28` (96.4%); protocol 100% |
| v115 state diagnostics | Numeric gate failed | Protocol 100%; supported-answer correctness `28/44` (63.6%, required 90%); insufficient-evidence search `50/56` (89.3%, required 90%) |
| v112 task reference | Descriptive only | EM 30/50, F1 0.6647, net gain +2 over v101, retention 26/28; not used as the acceptance baseline |
| Independent semantic review | Incomplete; no semantic pass claimed | Reviewer A completed all 100 states and 17 changed-task cases (351 candidate judgments; 236 supported, 65 unsupported, 50 uncertain). Reviewer B's artifact remains a partial semantic upgrade with untouched records provisional. Since two complete independent reviews are required, the semantic gate is not established. |
| Independent100 | Not opened | Blocked because two internal state gates failed; final-test remains untouched |
| Backup | Passed | AutoDL archive `backups/v115-final/v115-run.tar.gz`, SHA-256 `34245dd8bca6dc399a276b5c18b6a1d5412632c832ec234672aece648af8ee15`; 80 files checked against the included SHA-256 manifest |
| GitHub publication | Passed | Source and documentation pushed as commit `11260f2896bf1be8e747f3b0eb73fb468fa5b050` on branch `codex/v115-controlled-sft`, with tag `v0.8.7`. Private datasets, reports, mapping and model weights are excluded. |
| AutoDL shutdown | Invoked | GPU was idle (1 MiB, 0% utilization); only TensorBoard/Jupyter services were running. `/usr/bin/shutdown --help` caused the SSH session to close and later SSH attempts were refused; at the user's direction, no further console verification was performed. |

## Data and runtime limitations

The internal questions are sourced from the HotpotQA training partition, so this is an internal diagnostic and not a document-disjoint or truly unseen generalization estimate. The state-level targets have passed the documented dual review and citation gate; reviewer agreement does not replace independent ground truth. Any reported improvement must be described within these limits.

Because the internal gate failed, this result does not authorize opening independent100, choosing another checkpoint, tuning the recipe, adding training steps, or starting GRPO. The measured gain is a small internal result with a material supported-answer behavior deficit; it is not evidence of general accuracy improvement.
