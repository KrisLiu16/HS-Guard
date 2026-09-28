# Service validation — 2026-09-28

The GPU runtime still uses the released v10 weights and thresholds. This update adds resource limits and request deadlines, protects active sessions, bounds stream queues, discards failed session state, and exposes liveness for supervisor recovery. Deployment instructions are in [OPERATIONS.md](OPERATIONS.md).

## Request latency

On the same L20, with the same 2,048-token prefix and synchronized one-token arrivals as the previous service benchmark:

| Concurrent requests | 1 ms window P95 | 0.25 ms window P95 |
|---|---:|---:|
| 1 | 3.68 ms | **2.92 ms** |
| 4 | 4.86 ms | **4.08 ms** |
| 16 | **6.12 ms** | 6.15 ms |

Values are the worse of two 100-tick measured runs after 10 warmup ticks. The default window is now 0.25 ms: it reduces latency at low concurrency while 16-way latency is essentially unchanged. Timing includes queueing, checkpoint handling and record conversion; it excludes HTTP transport, initial prefill, tokenization and upstream generation.

Reproduce both windows from `examples/demo`:

```bash
GUARD_BATCH_WAIT_MS=1 python benchmarks/run.py --mode batch --bundle ./models/HS-Guard-v10 --output wait1.json
GUARD_BATCH_WAIT_MS=0.25 python benchmarks/run.py --mode batch --bundle ./models/HS-Guard-v10 --output wait025.json
python benchmarks/soak.py --url http://127.0.0.1:8080 --seconds 300 --output soak.json
```

## HTTP load

Sixteen clients created new session IDs throughout a five-minute replay load, with prompt lengths varied across three sizes. The run completed 10,397 HTTP requests with zero errors. The retained session count stayed at or below 64. After warmup, allocated GPU memory fluctuated with active work; reserved memory reached 7,392 MiB and stayed there over the final 100 seconds. There was no monotonic allocation growth in this run.

Full-request median/P95 were 331.39/948.54 ms. These include prompt prefill, text replay, tokenization, GPU scoring, SSE and response transfer; they are not one-token inference timings. The client and server ran on the same node. The client generated load for 300 seconds; polling and final completions brought recorded elapsed time to 310.32 seconds.

The 64-case, 16-session HTTP regression preserved all reference decisions. This regression set is not a new quality benchmark and is not used to adjust weights or thresholds.

The final server revision repeated the 64-case regression with zero changed decisions (maximum score deviation 0.005858). Live HTTP checks returned 409 for duplicate sessions and resets during generation, and 503 for a seventeenth active chat. Disconnected clients released their slots. SIGTERM during a running stream stopped the process in 1.603 seconds.

The deployed revision also passed all 64 decisions after the final queue-cancellation review; maximum score deviation from the reference was 0.003902. Deployed application file hashes were matched against the public source.

## Checked boundaries

Twenty service tests, five core contract tests and two tokenizer/session tests passed. CPU tests exercise queue/token capacity, invalid token input, pre-admission cancellation, queued and active deadlines, stalled workers, bounded shutdown, backend failure propagation, checkpoint branches, session eviction and reuse, stream backpressure, output caps, authentication, origin checks, HTTP parsing, connection limits, socket timeout, unhealthy probes and discarded partial sessions. Core model-contract tests remain unchanged.

Kubernetes templates passed client-side schema validation. The provided non-root Docker image and Nginx/TLS deployment templates have not been built or load-tested in this run. GPU tests use the existing PyTorch/CUDA deployment environment. No multi-node failover, long-duration SLO or multi-tenant isolation claim is made by these results.

[Machine-readable results](benchmarks/hardening_results.json). The original [serial-versus-batched baseline](BENCHMARK.md) remains available. A five-minute run does not establish long-term reliability; use the supplied soak runner against your own ingress, workload and hardware before assigning an SLO.
