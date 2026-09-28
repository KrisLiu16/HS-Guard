# Demo request batching

Measured on one NVIDIA L20 with PyTorch 2.7.1 and CUDA 12.8. Both paths use the same HS-Guard v10 weights, FP16 recurrent state and BF16 attention cache. The original runtime restores one workspace behind a lock and calls `step`; the batched runtime uses 16 workspaces, `step_packed`, a 64-token scheduling quantum and a 1 ms batch collection window.

Each stream starts from 2,048 tokens. Clients synchronize before each one-token request to measure overlapping arrivals. Each run has 10 warmup ticks and 100 measured ticks; two runs per concurrency. Values below are the worse of the two runs, with 200, 800 and 3,200 measured requests in total.

| Concurrent requests | Serial P50 (ms) | Batched P50 (ms) | Serial P95 (ms) | Batched P95 (ms) |
|---|---:|---:|---:|---:|
| 1 | 2.53 | 3.66 | 2.55 | 3.68 |
| 4 | 6.54 | 4.71 | 10.42 | 4.83 |
| 16 | 22.33 | 5.56 | 41.88 | 6.04 |

At 16 simultaneous requests, P95 is **6.9× lower**. Single-stream P95 increases by **1.13 ms**, primarily from waiting to form a batch. These timings include queueing, state restore/checkpoint copies and conversion to frontend records. They exclude HTTP transport, tokenization, initial prefill, upstream model generation and cold start. Synchronized arrivals are a contention test, not a measurement of production arrival patterns or maximum sustainable HTTP throughput.

Both runtimes preserved all 64 reference decisions (32 prompts, 32 responses) in the regression set. Floating-point scores are not bit-identical; the raw aggregates include their maximum deviations. Model SHA256: `01d07134b68f36352406bbdd3e3bc5dd72bef77006fde66ad4b7198eed2c4d4a`.

A separate HTTP/SSE replay test ran all 64 cases through 16 concurrent sessions, reusing session IDs across edited histories. It preserved all decisions; maximum score deviation from the reference was 0.00504. The same 64-case test passed again after deployment on a second L20, with zero changed decisions and maximum score deviation 0.00715. The startup streaming self-check also passed.

[Raw aggregates](benchmarks/results.json). Run the two modes sequentially on an otherwise idle GPU:

```bash
python benchmarks/run.py --mode serial --bundle ./models/HS-Guard-v10 --output serial.json
GUARD_BATCH_WAIT_MS=1 python benchmarks/run.py --mode batch --bundle ./models/HS-Guard-v10 --output batch.json
```

The benchmark uses a synthetic benign token stream. The included serial runtime is the original demo adapter retained for comparison. The 64-case regression data are not distributed with this repository.
