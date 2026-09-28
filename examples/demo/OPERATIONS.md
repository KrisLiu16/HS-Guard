# Operating the streaming service

The supported deployment is one process on one L20, up to 16 simultaneous chats, in a trusted workspace. Model state lives in GPU memory and is not shared between replicas. Place the service behind a TLS ingress. The included Nginx configuration bounds connections and request rate and leaves SSE buffering disabled.

## Access and secrets

Set `GUARD_PRODUCTION=1` and supply `GUARD_ACCESS_TOKEN` through a secret store. Production mode refuses to start without a token; configured tokens must contain at least 24 characters. Browsers use HTTP Basic authentication with username `guard` and the token as password. API clients may instead send `Authorization: Bearer <token>`. Use HTTPS at the ingress; Basic authentication does not encrypt the credential.

This is shared workspace authentication. It does not provide separate tenant identities, billing quotas or per-user session ownership. Session IDs must be unpredictable and must not be reused between users. For multiple tenants, use separate service instances and credentials or add tenant isolation at an authenticated gateway.

The upstream key uses `GUARD_BUILTIN_API_KEY` and never appears in `/api/info`. Upstream response bodies and exception strings are not returned to clients. Custom upstreams remain disabled by default. If enabled, `GUARD_CUSTOM_MODEL_ORIGINS` must list administrator-controlled origins, including scheme and port; private hosts also need `GUARD_ALLOW_PRIVATE_HOSTS`. Enforce network egress policy for those endpoints.

## Bounds and failure handling

| Resource | Limit / behavior |
|---|---|
| HTTP connections | 64; excess sockets close before allocating a handler thread |
| Socket reads/writes | 30 s timeout |
| Request body | 4 MiB JSON; ambiguous lengths and transfer encoding rejected |
| Input text | 262,144 characters across the conversation |
| Concurrent chats | 16; excess requests receive 503 |
| Same session concurrently | 409; active sessions cannot be reset or evicted |
| Retained sessions | 64; idle sessions expire after 1 hour, swept every 30 s |
| GPU queue | 128 requests, 65,536 input tokens; at most 2,048 tokens per request |
| GPU request | 30 s deadline; configurable with `GUARD_GPU_TIMEOUT_SECONDS` |
| GPU scheduling | Up to 16 rows, 64 tokens per row per tick |
| Upstream queue | 128 events; slow consumers apply backpressure |
| Generated text | 196,608 characters across reasoning and content |
| Upstream idle read | 30 s |
| Conversation | 49,152 tokens; validated engine context range remains 8,192 |
| Chat duration | 30 minutes |
| SIGTERM | Reject new chats, cancel streams, wait up to 30 s, then allow 5 s for GPU shutdown |

The GPU owner checks deadlines between ticks. A running CUDA operation cannot be forcibly interrupted by Python. A queue with no progress for 60 s fails liveness; the container supervisor must restart it. Expired requests are discarded before their slot is reused. A failed stream discards its session checkpoint so a retry rebuilds from the full conversation rather than retaining a partial update. Model weights and thresholds do not change during recovery.

`/api/health` is readiness and returns 503 while loading, draining or unhealthy. `/api/live` is liveness and permits initial loading; pair it with a startup probe. Both are credential-free and return status flags only. `/api/info` requires authentication when configured, and reports queue occupancy, timeouts, rejections, batch size, latency and allocated/reserved GPU memory. Alert on failed/stalled workers, increasing rejections or timeouts, and sustained occupancy near the configured limit.

## Container and Kubernetes

From the repository root:

```bash
docker build -f examples/demo/deploy/Dockerfile -t hs-guard:local .
```

`deploy/deployment.yaml` references a model PVC named `hs-guard-model`, with the downloaded model at `/models/HS-Guard-v10`, and a Secret named `hs-guard-access` containing `token`. Substitute your image registry and storage names before applying. The manifest uses a read-only root filesystem, a non-root UID, writable temporary directories, startup/readiness/liveness probes and a 45 s termination allowance. The image and manifest are deployment templates; the local validation report records which deployment paths were exercised.

Keep one GPU process per card. With multiple replicas, configure session affinity or route each session to an assigned replica; there is no cross-replica cache. A restart loses server-side checkpoints. The client can resend the complete conversation to rebuild state, with prefill cost. The single-replica `Recreate` deployment has a warmup outage during upgrades. High availability requires additional GPUs and a rollout/routing plan.

## Validation and rollback

Run the CPU suite, then the fixed-case decision regression on the intended GPU. Run `benchmarks/run.py` for warm request latency and `benchmarks/soak.py` for HTTP replay under load. Treat timing scopes separately: one-token engine calls, full HTTP replay, and upstream generation have different costs. A five-minute soak is a release check, not proof of a long-term SLO.

Use an immutable image tag or versioned ConfigMap, keep the previous revision and model hash, and switch readiness only after the streaming self-check passes. Roll back the image/configuration if decision regression changes or liveness repeatedly fails. No automatic model or threshold updates occur at startup.
