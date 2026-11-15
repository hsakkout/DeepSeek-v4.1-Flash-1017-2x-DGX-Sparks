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

The controller and client adapter have unit tests and a real stopped-state status check. They have not yet passed live GLM ↔ DeepSeek switching. DeepSeek preparation, canary qualification, boot-selector integration, Open WebUI verification, and the Hindsight Node 2 adapter remain separate completion gates.

Hindsight is deliberately not modified while its service is being relocated to Node 2.
