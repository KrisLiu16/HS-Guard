"""Adapt stable-token inference to the demo's text snapshots and rollback."""
import collections
import os
import numpy as np
import threading
from batching import BatchScheduler
import time
from concurrent.futures import TimeoutError as FutureTimeout


class GuardRuntime:
    def __init__(self, bundle):
        import torch
        from hs_guard.runtime import load_model
        self.torch = torch
        self.model, self.tokenizer, metadata = load_model(bundle)
        self.backend = self.tokenizer
        self.backend.no_truncation()
        self.backend.no_padding()
        self.meta = dict(metadata, checkpoint_sha256=metadata['files']['model.safetensors'],
                         candidate='HS-Guard v10', device=torch.cuda.get_device_name(),
                         selection_status='released', engine='SlotStreamEngineV8')
        self.timings = collections.deque(maxlen=2000)
        self.timing_lock = threading.Lock()
        slots = int(os.environ.get('GUARD_BATCH_SLOTS', '16'))
        self.request_timeout = float(os.environ.get('GUARD_GPU_TIMEOUT_SECONDS', '30'))
        if not np.isfinite(self.request_timeout) or self.request_timeout <= 0:
            raise ValueError('GUARD_GPU_TIMEOUT_SECONDS must be positive and finite')
        wait_ms = float(os.environ.get('GUARD_BATCH_WAIT_MS', '0.25'))
        if not 1 <= slots <= 16:
            raise ValueError('GUARD_BATCH_SLOTS must be between 1 and 16')
        torch.cuda.synchronize()
        self.scheduler = BatchScheduler(lambda: _GPUBackend(self.model, slots), slots=slots,
                                        wait_ms=wait_ms)
        self.meta['scheduler'] = 'batched-64-token'

    def encode(self, text):
        encoded = self.backend.encode(text, add_special_tokens=False)
        return encoded.ids, encoded.offsets

    def forward(self, ids, cache):
        started = time.perf_counter()
        future = self.scheduler.submit(ids, cache, timeout=self.request_timeout)
        try:
            snapshot, parts = future.result(timeout=self.request_timeout + 1)
        except FutureTimeout:
            future.cancel()
            raise TimeoutError('GPU request timed out') from None
        probs = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        logs = np.log(np.maximum(probs, 1e-30)).tolist()
        records = [{'lu': user, 'la': assistant} for user, assistant in zip(*logs)]
        with self.timing_lock:
            self.timings.append((len(ids), time.perf_counter() - started))
        return snapshot, records

    def healthy(self):
        state = self.scheduler.stats()
        return not (state['failed'] or state['stalled'] or state['closed'])

    def close(self, timeout=10):
        return self.scheduler.close(timeout)

    def clone(self, cache):
        # Snapshots are immutable; restore copies into the engine's workspace.
        return cache

    def latency(self):
        with self.timing_lock:
            samples = list(self.timings)
        values = sorted(seconds * 1000 for _, seconds in samples)
        if not values:
            return {}
        pick = lambda q: round(values[min(len(values) - 1, int(q * len(values)))], 2)
        return {'ticks': len(values), 'p50_ms': pick(.5), 'p95_ms': pick(.95),
                'tokens_per_tick': round(sum(n for n, _ in samples) / len(samples), 1),
                'scheduler': self.scheduler.stats(),
                'gpu_allocated_mib': round(self.torch.cuda.memory_allocated() / 2**20, 1),
                'gpu_reserved_mib': round(self.torch.cuda.memory_reserved() / 2**20, 1)}


class _GPUBackend:
    def __init__(self, model, slots):
        import torch
        from hs_guard.engine import SlotStreamEngineV8
        self.torch = torch
        buckets = tuple(n for n in (1, 2, 4, 8, 16) if n <= slots)
        if slots not in buckets:
            buckets += (slots,)
        self.engine = SlotStreamEngineV8(model, slots, state_dtype='float16', ring='inplace',
                                        gdn_bv=32, gdn_warps=2, session_buckets=buckets)
        self.names = ('rec', 'conv', 'ring_k', 'ring_v')
        with torch.inference_mode():
            for rows in buckets:
                for size in (1, 2, 4, 8, 16, 32, 64):
                    self.engine.step_packed([(slot, [0] * size) for slot in range(rows)])
            self.engine.reset()

    def start(self, slot, cache):
        with self.torch.inference_mode():
            for name in self.names:
                state = getattr(self.engine, name)[:, slot]
                if cache is None:
                    state.zero_()
                else:
                    state.copy_(cache[name])
            if cache is None:
                self.engine.ring_pos[slot].fill_(-1)
                self.engine.length[slot] = 0
            else:
                self.engine.ring_pos[slot].copy_(cache['ring_pos'])
                self.engine.length[slot] = cache['length']

    def step(self, batch):
        probs, lengths = self.engine.step_packed(batch)
        return [probs[:, i, :n].copy() for i, n in enumerate(lengths)]

    def finish(self, slot):
        with self.torch.inference_mode():
            snapshot = {name: getattr(self.engine, name)[:, slot].clone() for name in self.names}
            snapshot.update(ring_pos=self.engine.ring_pos[slot].clone(),
                            length=int(self.engine.length[slot]))
            return snapshot


def presets(thresholds):
    result = {}
    for role, name in (('user', '提问'), ('assistant', '回答')):
        result[role] = [{'id': f'v10_{role}', 'name': f'v10 {name}默认规则',
                         'score': 'cut', 'T': 1.0, 'rule': 'endpoint' if role == 'user' else 'threshold',
                         'alpha': .1, 'k': 1, 'tau': thresholds[role],
                         'note': '按提问末位分数判定' if role == 'user' else '逐 token 审核，超过阈值截断'}]
    result['user'].append({'id': 'off', 'name': '不拦截提问', 'score': 'cut', 'T': 1.0,
                           'rule': 'off', 'alpha': .1, 'k': 1, 'tau': thresholds['user'],
                           'note': '提问仍打分'})
    return result
