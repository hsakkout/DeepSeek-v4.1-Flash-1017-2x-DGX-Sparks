# GLM / DeepSeek switch controller

`scripts/llm_use.py` is installed as `~/.local/bin/llm-use` on this pair. It uses the existing port and bearer-key file, but always preserves the real served model id.

```bash
llm-use status
llm-use stop
llm-use restart       # restart the saved selection
llm-use glm           # drain/stop the source, start and verify GLM
llm-use dsv41         # requires completed DeepSeek preparation
```

## Safety behavior

- A non-blocking state lock prevents concurrent switches.
- The actual `/v1/models` responder and both nodes' exact container states must agree. Unknown worker state is not treated as stopped.
- The source is drained when its health endpoint exposes `live.connections` or `requests_running`. If authentication hides that count, the controller logs an explicit 30-second quiet-period fallback; this is not represented as a measured zero-connection drain.
- The source recipe's own stop script runs first. Both models' containers must then be absent on both nodes.
- DeepSeek starts with `PREPARE=0`; switching never downloads or rebuilds it.
- Verification requires both ranks, `/health`, unauthenticated `/v1/models` returning 401, the exact authenticated model id, and a live 2+2 correctness response.
- Client changes use `--apply`, atomic writes, backups, and read-back checks. Only existing local-provider references move; cloud defaults and per-chat overrides are left alone.
- The selection is committed only after server and client verification. A failed switch from a running source stops the partial target, restores the source, and verifies it. A failed start from an intentionally stopped state remains stopped.
- `llm-use stop` leaves the saved selection unchanged, so `llm-use restart` intentionally restores it. A manual stop does not reset the existing once-per-boot handled marker.

## Current qualification boundary

The controller and client adapter have unit tests, live dry-runs, real stopped-state status checks, and a successful idempotent DeepSeek selection. Hermes's real resolver confirmed both profiles keep their cloud primary and use the DeepSeek local fallback. The Hindsight Node 2 adapter atomically updates the relocated `hindsight-api.service`, restores the prior file on verification failure, and verifies health plus the live process model. Its real DeepSeek update passed. DeepSeek has also passed the local canary, but live GLM ↔ DeepSeek switching, Open WebUI verification, and the longer stability qualification remain completion gates.

Boot-selector integration is installed behind the existing `glm53-tf-autostart.timer`, with the pre-selector script backed up. The selector reads the saved model, applies GLM's or DeepSeek's memory threshold, starts DeepSeek with `PREPARE=0`, and runs full verification. Its four tests cover invalid selection, handled-boot idempotence, and GLM/DeepSeek dry-runs. The current handled boot pair was also dry-run verified to take no action. A real reboot has not been performed.

Hindsight is relocated to Node 2. The adapter uses the LAN admin address for its service management, while Hindsight's own inference backend remains the ConnectX address `192.168.200.1:8888/v1`. The Node 1 service is not modified.
