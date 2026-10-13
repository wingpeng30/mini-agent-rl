# v133 Goal D execution record

This record freezes Goal D before its SFT and paired evaluation. It follows the requested remote AutoDL route and keeps Goal C artifacts unchanged.

## Evidence that supports this trial

- The first SFT completed 7 optimizer steps on 25 reviewed examples. Its first step had learning rate 0 because the one-step warmup precedes optimizer updates; six subsequent updates used a decaying rate up to 5e-6.
- CPU comparison of the v118 and first SFT adapter files found the same 192 tensors and 3,194,880 scalar parameters. 3,194,829 scalars changed; max absolute delta was 1.7580e-5 and relative L2 delta was 0.0010614. This rules out a no-op save, while showing the parameter change was small.
- All 25 training answers were present literally in visible training evidence. No new labels or rows are needed for this dosage diagnostic.
- On the context-aligned 16-prompt evaluation, v118 and first SFT both scored 12/16 strict EM and 0.8417 mean F1, with 16/16 protocol-valid answers and identical raw outputs. The preceding frozen run remains separately preserved.

## Frozen experiment

Continue from the existing SFT adapter, which is itself based on v118. Use the same 25 approved train rows and same 16 validation rows, seed 42, batch 1, gradient accumulation 4, BF16, max length 2176, no gradient checkpointing, no thinking, and LR 5e-6. Train exactly 14 more optimizer steps (two further dataset passes), one warmup step, one seed, one output adapter. This is a dosage diagnostic only; it does not claim the 25 cases provide broad coverage. The 16 dev labels remain evaluation-only.

Compare v118 against the resulting cumulative adapter on the same 16 deterministically derived prompts in `scripts/evaluate_v133_goal_c_aligned_context.py`: question, search(question), first existing evidence, search(first evidence text), second existing evidence, answer. This is a synthetic trace and each tool result has one fact, while training tool results have three; no claim of full context-distribution equality or end-to-end search performance will be made.

Primary metric: strict normalized EM. Secondary: token F1. Also report protocol validity and exact lists of new, lost, and retained correct cases. Greedy decoding, same evidence and context for both adapters, max 128 new tokens. Training hard cap 60 minutes and evaluation hard cap 60 minutes; use the first two optimizer steps and first two baseline outputs for forward time projections. Stop if either projection exceeds its cap. Do not retry, adjust settings, train on dev cases, expand the pool, or modify the prompt after seeing outputs.

Stop this recipe if the new adapter gains fewer than 2/16 strict EM. If it gains at least 2/16, treat that as exploratory and require evidence-support adjudication of changed answers before any promotion claim. This execution will not make a positive claim from EM alone.

## Frozen identities

- Config SHA-256: `dcb7886c8a92511cbab3405ee75412f3890105b43130f11fe65bbe0e411b638b`
- Training runner SHA-256: `94389099b184af5e92dd55c9727841049d19fc8890b91b0a7a2443fec4999bee`
- Paired evaluation runner SHA-256: `f492560fa4a6b0a126e4a386a91141c83d7c296aa083f282edad4eecf2b43f48` (updated after a CPU-preflight-only guard correction; no inference had started)
- Context builder SHA-256: `1308fb0c4aaa4b7dbccaefb7c583fb277c83f543ef941ab7331825a28d8c04a3`
- Train SHA-256: `b4d38e1e99886e3a254a6e881a28b7a36194f7f185b686a64dc7dbdf449ee9df`
- Dev SHA-256: `23cde9d8f25376d469c91128742ca647fc4d84afe04b2009e38e571198780e58`
- Parent adapter SHA-256: `0a80d82918e301cac05f0cc2892a7ccd007138996e121549b95dbcbf7f8919c6`
- v118 adapter SHA-256: `ea894d235b6779bc8d15d1327a953773f946c347dfa7fa55ee3d2c2d5bd4a0a5`
- Prior aligned evaluation SHA-256: `fe8ee1c6e0a6c4633a6dcf586115277c7b352aa99a0b428b9102b9413b49d203`

Remote output paths are new: `runs/v133-goal-d-sft-once/adapter/` and `runs/v133-goal-d-sft-once/paired-evaluation.json`. The run marker is written before loading weights. Existing Goal C files will not be overwritten.
