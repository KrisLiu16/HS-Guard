# Interactive streaming demo

Run HS-Guard on live model responses or replay pasted text. The page includes token scores, risk curves, configurable decision rules and conversation export. Vue 3.5.43 and TDesign 1.20.8 assets are included; their MIT licenses are in the repository's `third_party_licenses` directory.

Use Linux with an NVIDIA GPU. The tested environment is Python 3.11, NVIDIA L20 and CUDA 12.8. Install a C compiler for Triton, then from this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python download_model.py
cp .env.example .env
bash start.sh
```

Open http://127.0.0.1:8080/. Text replay needs no upstream API key. For live responses, set the `GUARD_BUILTIN_*` variables in `.env` with your endpoint, model and key. Supported protocols are `chat_completions`, `responses` and `anthropic_messages`. The upstream model generates text; HS-Guard scores it. The key stays on the server.

`/api/health` returns 200 after model verification, graph warmup and a streaming self-check. `/api/info` includes rolling request latency and scheduler batch statistics. Set `GUARD_ACCESS_TOKEN` for browser Basic authentication or API Bearer authentication. Production mode requires it. See [operations](OPERATIONS.md) for TLS, probes, resource bounds and deployment templates. Custom upstream URLs are disabled by default.

## GPU scheduling

One worker owns the mutable CUDA engine. Concurrent HTTP handlers enqueue immutable token requests; up to 16 requests occupy separate engine slots in each batch. Each tick processes at most 64 tokens per request through `step_packed`. Long prefills yield between ticks so newly admitted streams can make progress.

The worker restores a request's checkpoint once on admission, keeps its slot through all its ticks, and returns an immutable checkpoint on completion. Text edits, tokenizer-boundary changes and provisional tail scoring can branch from a checkpoint without modifying another request. Slots are reused only after completion. Checkpoints remain on GPU; this demo still incurs snapshot copies at request boundaries.

| Setting | Default | Meaning |
|---|---:|---|
| `GUARD_BATCH_SLOTS` | 16 | Maximum simultaneous GPU workspaces, 1–16 |
| `GUARD_BATCH_WAIT_MS` | 0.25 | Maximum idle-worker batch collection window, 0–10 ms |
| Queue capacity | 128 | Accepted requests, including active requests |
| Active HTTP chats | 16 | Excess chats receive HTTP 503 |

A shorter batch window reduces single-stream delay but can split arrivals into smaller batches. The worker drains accepted requests within their deadlines; `close(timeout=10)` bounds the caller’s wait. A backend failure rejects subsequent requests and fails all pending work; health becomes unavailable. A future may be cancelled before admission. Once admitted, deadline handling occurs between GPU ticks before slot reuse; `Future.cancel()` applies only before admission.

[Current service validation](HARDENING.md) and [original service latency](BENCHMARK.md) includes queueing, state copies and conversion to frontend records. The separate 352-stream engine benchmark is not an HTTP capacity measurement.

## Tests

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```

These CPU tests cover batch order, checkpoint branches, bounded queues, cancellation, slot reuse, prefill scheduling and failure propagation. GPU validation results and timing scope are in `BENCHMARK.md`.

Code is Apache-2.0. Model weights are CC BY-NC 4.0, for noncommercial research. [Model repository](https://huggingface.co/KrisLiu16/HS-Guard-v10).
