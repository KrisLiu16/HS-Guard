# Streaming engine

The engine maintains separate state for each slot: 18 gated-delta recurrent matrices, short-convolution histories and six attention key/value rings. The attention window contains 512 positions. State capacity is fixed by `max_slots`; it does not grow with the stream's token count.

CUDA Graphs are captured lazily for session/token buckets. BF16 is used for the backbone computation and key/value rings; recurrent-state storage is FP16 with FP32 arithmetic in its update kernel. Fused kernels reduce intermediate copies and state gathers. Inputs are copied from pinned host buffers, and `step` returns when the selected output probabilities are available on the host.

## Contract

- A slot belongs to a single live conversation. Do not reuse it without clearing its state.
- Each request contains nonempty, in-vocabulary integer token IDs. A tick cannot repeat a slot.
- Tokens must already be stable under tokenization. Appending raw text and retokenizing a previous suffix requires a snapshot/replay strategy outside this interface.
- Context tokens update state but must not be counted as response risk. `encode_messages` returns the target content positions.
- The complete-message scorer is the evaluation reference. Floating-point output is not guaranteed identical across arbitrary chunk sizes.
- Red-line thresholds are prompt 0.0761368871 and response 0.1117895246. General thresholds are 0.5.

## Measured performance

One L20, PyTorch 2.7.1 / CUDA 12.8. Single-token service P50 is approximately 2.38 ms after prefill. The arrival simulation uses 30-60 tokens/s per session, random start offsets, 10-second runs and two seeds. At 352 sessions, worst P95 across 512 / 2,048 / 8,000-token contexts is 18.368 ms. The 384-session point misses the 20 ms target.

Timing excludes tokenization, network transport, model loading and prefill. Session capacity is from short synthetic-arrival tests, not a production HTTP guarantee. Memory is preallocated for the configured maximum slots, so the allocation seen with one active slot is not a one-session memory measurement.

The comparison baseline is an earlier same-L20 SGLang 0.5.3rc0 measurement of Qwen3Guard-Stream-0.6B, support branch commit 9a06537, with radix cache enabled. It is not a statement about the fastest possible implementation of that model.
