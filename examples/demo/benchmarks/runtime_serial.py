"""Adapt stable-token inference to the demo's text snapshots and rollback."""
import collections
import math
import threading
import time


class GuardRuntime:
    def __init__(self, bundle):
        import torch
        from hs_guard.runtime import load_model
        from hs_guard.engine import SlotStreamEngineV8
        self.torch = torch
        self.model, self.tokenizer, metadata = load_model(bundle)
        self.backend = self.tokenizer
        self.backend.no_truncation()
        self.backend.no_padding()
        self.meta = dict(metadata, checkpoint_sha256=metadata['files']['model.safetensors'],
                         candidate='HS-Guard v10', device=torch.cuda.get_device_name(),
                         selection_status='released', engine='SlotStreamEngineV8')
        self.engine = SlotStreamEngineV8(self.model, 1, state_dtype='float16', ring='inplace',
                                        gdn_bv=32, gdn_warps=2, session_buckets=(1,))
        self.gpu = threading.Lock()
        self.timings = collections.deque(maxlen=2000)
        with torch.inference_mode():
            for size in (1, 2, 4, 8, 16, 32, 64):
                self.engine.step([(0, [0] * size)])
            self.engine.reset()

    def encode(self, text):
        encoded = self.backend.encode(text, add_special_tokens=False)
        return encoded.ids, encoded.offsets

    def _snapshot(self):
        return {**{name: getattr(self.engine, name)[:, 0].clone()
                   for name in ('rec', 'conv', 'ring_k', 'ring_v')},
                'ring_pos': self.engine.ring_pos[0].clone(), 'length': int(self.engine.length[0])}

    def _restore(self, cache):
        if cache is None:
            self.engine.reset()
            return
        for name in ('rec', 'conv', 'ring_k', 'ring_v'):
            getattr(self.engine, name)[:, 0].copy_(cache[name])
        self.engine.ring_pos[0].copy_(cache['ring_pos'])
        self.engine.length[0] = cache['length']

    def forward(self, ids, cache):
        started = time.perf_counter()
        with self.gpu, self.torch.inference_mode():
            self._restore(cache)
            outputs = self.engine.step([(0, ids)])
            snapshot = self._snapshot()
            records = [{key: [math.log(max(float(v), 1e-30)) for v in outputs[role][0][i]]
                        for role, key in (('user', 'lu'), ('assistant', 'la'))}
                       for i in range(len(ids))]
        self.timings.append((len(ids), time.perf_counter() - started))
        return snapshot, records

    def clone(self, cache):
        # Snapshots are immutable; restore copies into the engine's workspace.
        return cache

    def latency(self):
        samples = list(self.timings)
        values = sorted(seconds * 1000 for _, seconds in samples)
        if not values:
            return {}
        pick = lambda q: round(values[min(len(values) - 1, int(q * len(values)))], 2)
        return {'ticks': len(values), 'p50_ms': pick(.5), 'p95_ms': pick(.95),
                'tokens_per_tick': round(sum(n for n, _ in samples) / len(samples), 1)}


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
