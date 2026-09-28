"""Bounded GPU work queue. The worker alone owns mutable engine slots."""
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
import math
import threading
import time


@dataclass
class _Request:
    ids: tuple
    cache: object
    future: Future = field(default_factory=Future)
    deadline: float = float("inf")
    offset: int = 0
    parts: list = field(default_factory=list)


class BatchScheduler:
    """backend.start/step/finish run on one worker; returned caches must be immutable.

    step accepts (slot, token_ids) rows and returns one output per row.
    finish returns a checkpoint. Cancellation is supported before admission.
    close drains accepted work; a backend failure fails all pending work.
    """
    def __init__(self, backend_factory, slots=16, capacity=128, quantum=64, wait_ms=1,
                 max_tokens=2048, token_capacity=65536, stall_seconds=60):
        if not (1 <= slots <= capacity and 1 <= quantum <= 64 and 0 <= wait_ms <= 10):
            raise ValueError('invalid scheduler limits')
        if max_tokens < 1 or token_capacity < max_tokens or not math.isfinite(stall_seconds) or stall_seconds <= 0:
            raise ValueError('invalid token or stall limits')
        self.slots, self.capacity, self.quantum = slots, capacity, quantum
        self.wait_seconds = wait_ms / 1000
        self.max_tokens, self.token_capacity = max_tokens, token_capacity
        self.stall_seconds = stall_seconds
        self.last_progress = time.monotonic()
        self.tokens = self.expired = self.rejected = 0
        self.cv = threading.Condition()
        self.pending = deque()
        self.closed = False
        self.error = None
        self.outstanding = 0
        self.batches = self.rows = self.max_batch = 0
        self.ready = Future()
        self.thread = threading.Thread(target=self._run, args=(backend_factory,), daemon=True,
                                       name='guard-gpu')
        self.thread.start()
        self.ready.result()

    def submit(self, ids, cache=None, timeout=30):
        ids = tuple(ids)
        if not 1 <= len(ids) <= self.max_tokens:
            raise ValueError('token count exceeds request limits')
        if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in ids):
            raise ValueError('token IDs must be nonnegative integers')
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive and finite')
        with self.cv:
            if self.error is not None:
                raise RuntimeError('GPU worker failed') from self.error
            if self.closed:
                raise RuntimeError('GPU scheduler is closed')
            if self.outstanding >= self.capacity or self.tokens + len(ids) > self.token_capacity:
                self.rejected += 1
                raise RuntimeError('GPU queue is full')
            request = _Request(ids, cache, deadline=time.monotonic() + timeout)
            self.pending.append(request)
            if not self.outstanding:
                self.last_progress = time.monotonic()
            self.outstanding += 1
            self.tokens += len(ids)
            self.cv.notify()
        return request.future

    def stats(self):
        with self.cv:
            return dict(slots=self.slots, outstanding=self.outstanding, batches=self.batches,
                        mean_batch=round(self.rows / max(1, self.batches), 2),
                        max_batch=self.max_batch, failed=self.error is not None,
                        closed=self.closed, queued_tokens=self.tokens, expired=self.expired,
                        rejected=self.rejected, stalled=bool(self.outstanding and
                            time.monotonic() - self.last_progress > self.stall_seconds))

    def close(self, timeout=10):
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    def _run(self, factory):
        active = {}
        try:
            backend = factory()
            self.last_progress = time.monotonic()
            self.ready.set_result(True)
            while True:
                with self.cv:
                    while not active and not self.pending and not self.closed:
                        self.cv.wait()
                    if not active and not self.pending and self.closed:
                        return
                    if not active and not self.closed:
                        deadline = time.monotonic() + self.wait_seconds
                        while len(self.pending) < self.slots and not self.closed:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            self.cv.wait(remaining)
                    for slot, request in list(active.items()):
                        if time.monotonic() >= request.deadline:
                            del active[slot]
                            self.outstanding -= 1
                            self.tokens -= len(request.ids)
                            self.expired += 1
                            request.future.set_exception(TimeoutError('GPU request deadline exceeded'))
                    admitted = []
                    for slot in range(self.slots):
                        if slot in active:
                            continue
                        while self.pending:
                            request = self.pending.popleft()
                            if not request.future.set_running_or_notify_cancel():
                                self.outstanding -= 1
                                self.tokens -= len(request.ids)
                                continue
                            if time.monotonic() >= request.deadline:
                                self.outstanding -= 1
                                self.tokens -= len(request.ids)
                                self.expired += 1
                                request.future.set_exception(TimeoutError('GPU request deadline exceeded'))
                                continue
                            active[slot] = request
                            admitted.append((slot, request))
                            break
                for slot, request in admitted:
                    backend.start(slot, request.cache)
                    request.cache = None
                if not active:
                    continue
                batch = [(slot, r.ids[r.offset:r.offset + self.quantum])
                         for slot, r in active.items()]
                outputs = backend.step(batch)
                if len(outputs) != len(batch):
                    raise RuntimeError('GPU backend returned the wrong number of rows')
                with self.cv:
                    self.last_progress = time.monotonic()
                    self.batches += 1
                    self.rows += len(batch)
                    self.max_batch = max(self.max_batch, len(batch))
                for (slot, ids), output in zip(batch, outputs):
                    request = active[slot]
                    request.offset += len(ids)
                    request.parts.append(output)
                    if request.offset == len(request.ids):
                        if time.monotonic() >= request.deadline:
                            with self.cv:
                                self.outstanding -= 1
                                self.tokens -= len(request.ids)
                                self.expired += 1
                            del active[slot]
                            request.future.set_exception(TimeoutError('GPU request deadline exceeded'))
                            continue
                        result = backend.finish(slot), request.parts
                        with self.cv:
                            self.outstanding -= 1
                            self.tokens -= len(request.ids)
                        del active[slot]
                        request.future.set_result(result)
        except BaseException as error:
            with self.cv:
                self.error = error
                self.closed = True
                failed = list(active.values()) + list(self.pending)
                self.pending.clear()
                self.outstanding = 0
                self.tokens = 0
            if not self.ready.done():
                self.ready.set_exception(error)
            for request in failed:
                if not request.future.done():
                    request.future.set_exception(error)
