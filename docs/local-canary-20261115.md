# Local DeepSeek canary — 2026-11-15

Cluster: Node 1 `promaxgb10-0874`, Node 2 `spark-8485`; TensorFold TP=2, both RoCE rails, GID 3, path MTU 5.

## Pins and layout

- Recipe branch: `local-production-tp2` in the user's fork.
- Engine: `fbbc97132cd004237d9887b8d3185896efaedc5c` plus the recipe patches; image `tensorfold-dsv41:exl3-tp2-8a98ae503d85`, image id `a4792713e9a089179e54a02eb42a8d10c97fe25ce6ec0f103c9ee3f6ea19729f`, content-identical on both nodes.
- Checkpoint: `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` @ `45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b`.
- Checkpoint verification: 64/64 files, exact file set and sizes, Git blob ids and LFS SHA-256 values, 210,663,291,755 bytes. Receipt: `~/.local/state/llm-switch/downloads/pack-verification.json`.
- Engram: `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1088993b9b82de08f59de5be5a34e42170`; both official SHA-256 values passed on the head and again after the worker copy. Sizes: 101,535,150,936 and 101,537,926,640 bytes.
- Rank files: 101,191 MiB each. Rank 1 worker file is 106,120,890,008 bytes and matches the head's SHA-256.
- No NFS server, export, or mount was added. Worker runtime tree, Engram and rank cache are local. Head retains both generated rank caches.
- Runtime `vm.min_free_kbytes=45155` on both nodes. The persistent sysctl file remains unchanged at this stage. Swappiness remains 60.

## Startup

Command: `PREPARE=0 ./start.sh`.

- Exact served id: `DeepSeek-V4.1-Flash-1017`.
- Context 524,288, two streams, EAQ 2.9 bpw, drafts on, images on.
- Weight load 44.4 s; API answered after 49 s; built-in cross-rank smoke returned `OK`.
- Post-load memory: 6.1 GiB head, 7.0 GiB worker.
- Unauthenticated `/v1/models`: 401. `/health` under key authentication returns only `{"status":"ok"}`.

## Functional checks

`tools/smoke.py` passed health, plain reply, special-token leak check, max-token limit, stop string, streaming `[DONE]` and usage, array-valued tool arguments, tool-result round trip, greedy determinism, and thinking on/off. Receipt: `~/.local/state/llm-switch/canary/smoke.json`.

Authenticated `tools/validate.py` passed text, thinking, tool call, tool result, image, a 21,875-token long prompt in 9.6 s, and 8/8 concurrent replies equal to solo replies. Video was skipped because PyAV is not installed. Receipt: `~/.local/state/llm-switch/canary/validate.log` and `validate-replies.json`.

`scripts/canary_cancel_retry.py` cancelled a cold 30K-token prefill after 2 s, then a fresh request answered `4` with finish reason `stop`; health remained good. Receipt: `~/.local/state/llm-switch/canary/cancel-retry.json`.

## Short benchmark

One repetition, fixed seed, random prose, thinking off. This is a canary measurement, not a multi-repetition performance claim. Receipt: `~/.local/state/llm-switch/canary/bench.json`.

| Check | Result |
| --- | ---: |
| 2K prefill | 1,911.8 tok/s; TTFT 1.01 s |
| 8K prefill | 2,273.0 tok/s; TTFT 3.38 s |
| 30K prefill | 2,316.5 tok/s; TTFT 12.41 s |
| One-stream decode | 64.5 tok/s; TTFT 0.16 s |
| Two-stream aggregate decode | 87.4 tok/s; mean 46.1 tok/s/stream |
| Request errors | 0 |

Memory after validation and cancellation: 5.8 GiB head, 6.5 GiB worker. Node 1 swap in use was about 758 MiB; swappiness was not changed. This leaves limited headroom and requires the agreed mixed-workload/soak qualification before production promotion. No 512K needle or long soak has yet been run locally.
