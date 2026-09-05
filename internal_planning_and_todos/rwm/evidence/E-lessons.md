# E — Lessons that changed the design (sealed, dated)

| lesson | evidence | rule now |
|---|---|---|
| Serve-consistency (H14.11) | P1/P2/P3 non-evaluable; trunk amplifies lang deviation 30–80× | conditioning statistic must be causally computable at serve; D7 `--lang-mode stored`; `encoder_step` check |
| Zero-init decaying GDN read erases the oldest window slots | at `pos_decay_bias` 0.0 the 8-slot prefix moves the conditioning 7e-6 (< fp16 floor); at −4.0, 0.586 | effective horizon ≪ window at init — bears on sealed dnw8/w16 arms; init −4.0 on all M-arms |
| Grid alignment train↔serve | store grid arange(0,n,8) vs serve frames ≡ exec_start_idx mod 8 → live coverage 0 on 81 % of demo episodes | serve-aligned grid for the policy ω store; one shared `window_for_step` both sides import |
| Gates are code — pre-flight them like what they gate | parity pointed at `encoder_best.pt` while ω came from `encoder.pt` → cos 0.13 on a good encoder | `encoder_step` recorded + refused on mismatch |
| Silent success is the enemy | `--tasks all` → 0 to do / exit 0; index dropping an absent domain; judge shipping an empty store; 3-tap loader training on 2 domains; retrieval gate on n_anchors 0 | fatal + count assertions (994/8,869; per-domain non-emptiness; homogeneity) |
| Selection metrics vs validity floors | ctrl-0b passes G1b/eff-rank/bevf at chance retrieval | never select on them |
| Pre-register the stratum | ctrl-1Db changed the gate population | whole-gate Δ withdrawn |
| Node first-run landmines (vLLM on H100) | 9 attempts: bare `python`; reserved-capacity key; `enforce_eager`; FlashInfer cubin via FP8 DeepGEMM (fix `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0`) + GDN prefill triton; bare `wait`; `set -e` on `wait $pid`; missing pandas/pyarrow/av; log shadowing | canary any node entry on a real node before building a chain; per-attempt log prefixes |
| RoboMME renderer | SAPIEN dies every 25 episodes deterministically | keep `--max-renderer-restarts 64`; ≥31 needed per 800 |
| Node venv deps are per-entry | rmme tap FAILED 2026-09-03: `wsm_robocasa_configs` imports `robocasa` at import time; entry never installed it | mirror the proven install block across entries; import the exact module in the preflight |
| Queue governance | org SCP denies untagged submits; terminate works only on tagged jobs; 24 h ⇒ 600 rule relaxed to a 48 h standard class (user, 2026-09-02) | tags always; 400 up to 48 h |
| Budget pairing beats equivalence arguments | rcb 15k chosen to pair EXACTLY with H12 | now superseded by the saturation protocol (H14.9) with base curves |
| Executors and tokens | Opus executors stalled on session limits twice; Fable ones twice more; ~1.1M tokens in an afternoon | one executor at a time, critical path only; resume by message, never respawn |
