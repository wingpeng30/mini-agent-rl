# v133 Goal C-Run result

**Decision: stop this run; do not promote the SFT adapter.** The one authorized SFT completed, and the frozen paired evaluation completed within its time limit. However, all 32 generations violated the answer-action protocol, so this run does not provide an interpretable answer-accuracy comparison. The precommitted continuation gate (+2/16 EM with no evidence-support regression) was not met. No blind review calls were made.

## Execution

- Host: user-selected AutoDL instance `connect.cqa1.seetacloud.com:44249`; GPU: NVIDIA GeForce RTX 4090 D, 23.643 GiB, BF16 available.
- Runtime: torch 2.5.1+cu124, transformers 5.17.0, PEFT 0.20.0, datasets 5.0.1, accelerate 1.15.0, Pillow 12.3.0, tokenizers 0.23.2, safetensors 0.8.0, huggingface-hub 1.33.0. `pip check` and the frozen CPU preflight passed.
- SFT: fixed v118 base adapter, 25 train rows, 16 validation rows, seed 42, 7 optimizer steps, LR 5e-6, one run. Training finished in 14.653 s; the two-step estimate was 21.84 s versus the 3600 s cap; peak GPU memory was 12.185 GiB.
- Evaluation: the same 16 frozen prompts for v118 and SFT, 32 generations total. The estimate after the first two outputs was 32.655 s versus the 3600 s cap; total elapsed time was 28.965 s.
- There was no retry, parameter/seed change, pool expansion, smoke run, or external review call.

## Paired results and interpretation

| Measure | v118 | SFT |
|---|---:|---:|
| Prompts | 16 | 16 |
| Protocol-valid answer actions | 0/16 | 0/16 |
| EM emitted by the evaluator | 0/16 | 0/16 |
| Mean F1 emitted by the evaluator | 0.000 | 0.000 |

The evaluator's paired EM accounting is 0 newly correct, 0 lost correct, and 0 retained correct. Because every output is protocol-invalid, these are protocol-gated counts and do not estimate latent answer correctness.

The evaluator requires a JSON `{"action":"answer","answer":"..."}` response. Every generation was parsed as a `ValueError` because it emitted a `search` action instead. Thus the zero EM/F1 values are protocol-failure outputs, not evidence that the two systems have equal answer accuracy. Twelve of sixteen paired raw generations were identical; the four changed generations were also invalid. The case-level predictions and raw text are preserved in `autodl-run-20260928/paired-evaluation.json`.

This is a technical stop for the frozen answer-extraction setup. The run does not show a benefit, does not meet the continuation gate, and does not support a claim of accuracy regression or equivalence. Changed-answer support review was correctly skipped because the +2/16 gate was not reached. A future attempt would need a separately authorized, preregistered protocol correction; this run was not modified or repeated to obtain a passing result.

## Traceability

The authoritative run files are mirrored from AutoDL into this directory. Local and remote SHA-256 values matched for the evaluation and training-start records.

- Config SHA-256: `016ddcb51d2916f4a9ff87ca40baf09c5f265e3b3b188f7ee71effa23444309f`
- Freeze manifest SHA-256: `fa4ba22387e3c8f6511bfc8f808d6a97211bd0a91af3b515f0eaabeed099022a`
- Train data SHA-256: `b4d38e1e99886e3a254a6e881a28b7a36194f7f185b686a64dc7dbdf449ee9df`
- Dev data SHA-256: `23cde9d8f25376d469c91128742ca647fc4d84afe04b2009e38e571198780e58`
- Training metrics SHA-256: `a564722c519fc9c2b47a59fe838328f4e558603efd9e9b743aed7388e1072570`
- Training-start record SHA-256: `90c9ecacc473c776c8c25afc5ecbb4fd6d856f25c4a1a0b1bc020dc69efe81ac`
- Paired evaluation SHA-256: `86cc52318ccf07afb1839971585e1efbc7a497e6caf0995b115814b3b5fc120c`
- SFT adapter weights SHA-256: `0a80d82918e301cac05f0cc2892a7ccd007138996e121549b95dbcbf7f8919c6`
- SFT adapter config SHA-256: `c418c817b5811f8f0d13fda7edf30fc6ee5fde8f4fed7721a8605a21477b3090`

`kasa_config` in the v118 adapter is `null`; PEFT warned that it ignored this unknown null field. The trained and base adapter LoRA rank, alpha, dropout, and target-module set match. Transformers used reference implementations for `causal_conv1d` and `flash-linear-attention`; no packages were added for these optional kernels, and both runs completed under the frozen time caps.
