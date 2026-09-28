"""Bounded GPU work queue. The worker alone owns mutable engine slots."""
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
import threading
import time


@dataclass
class _Request:
    ids: tuple
    cache: object
    future: Future = field(default_factory=Future)
    offset: int = 0
    parts: list = field(default_factory=list)


class BatchScheduler:
    """backend.start/step/finish run on one worker; returned caches must be immutable.

    step accepts (slot, token_ids) rows and returns one output per row.
    finish returns a checkpoint. Cancellation is supported before admission.
    close drains accepted work; a backend failure fails all pending work.
    """
    def __init__(self, backend_factory, slots=16, capacity=128, quantum=64, wait_ms=1):
        if not (1 <= slots <= capacity and 1 <= quantum <= 64 and 0 <= wait_ms <= 10):
            raise ValueError('invalid scheduler limits')
        self.slots, self.capacity, self.quantum = slots, capacity, quantum
        self.wait_seconds = wait_ms / 1000
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

    def submit(self, ids, cache=None):
        ids = tuple(ids)
        if not ids:
            raise ValueError('token_ids must not be empty')
        with self.cv:
            if self.error is not None:
                raise RuntimeError('GPU worker failed') from self.error
            if self.closed:
                raise RuntimeError('GPU scheduler is closed')
            if self.outstanding >= self.capacity:
                raise RuntimeError('GPU queue is full')
            request = _Request(ids, cache)
            self.pending.append(request)
            self.outstanding += 1
            self.cv.notify()
        return request.future

    def stats(self):
        with self.cv:
            return dict(slots=self.slots, outstanding=self.outstanding, batches=self.batches,
                        mean_batch=round(self.rows / max(1, self.batches), 2),
                        max_batch=self.max_batch, failed=self.error is not None)

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        self.thread.join()

    def _run(self, factory):
        active = {}
        try:
            backend = factory()
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
                    admitted = []
                    for slot in range(self.slots):
                        if slot in active:
                            continue
                        while self.pending:
                            request = self.pending.popleft()
                            if request.future.set_running_or_notify_cancel():
                                active[slot] = request
                                admitted.append((slot, request))
                                break
                            self.outstanding -= 1
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
                    self.batches += 1
                    self.rows += len(batch)
                    self.max_batch = max(self.max_batch, len(batch))
                for (slot, ids), output in zip(batch, outputs):
                    request = active[slot]
                    request.offset += len(ids)
                    request.parts.append(output)
                    if request.offset == len(request.ids):
                        result = backend.finish(slot), request.parts
                        with self.cv:
                            self.outstanding -= 1
                        del active[slot]
                        request.future.set_result(result)
        except BaseException as error:
            with self.cv:
                self.error = error
                self.closed = True
                failed = list(active.values()) + list(self.pending)
                self.pending.clear()
                self.outstanding = 0
            if not self.ready.done():
                self.ready.set_exception(error)
            for request in failed:
                if not request.future.done():
                    request.future.set_exception(error)
