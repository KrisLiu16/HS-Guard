"""Exercise queue ordering, immutable checkpoints and failure propagation on CPU."""
import threading
import unittest
from batching import BatchScheduler


class Backend:
    def __init__(self):
        self.states = {}
        self.batches = []

    def start(self, slot, cache):
        self.states[slot] = cache or 0

    def step(self, batch):
        self.batches.append([(s, tuple(ids)) for s, ids in batch])
        result = []
        for slot, ids in batch:
            row = []
            for token in ids:
                self.states[slot] = self.states[slot] * 7 + token
                row.append(self.states[slot])
            result.append(row)
        return result

    def finish(self, slot):
        return self.states[slot]


class SchedulerTests(unittest.TestCase):
    def test_batch_order_checkpoint_and_branch(self):
        backend = Backend()
        scheduler = BatchScheduler(lambda: backend, slots=4, wait_ms=10, quantum=2)
        try:
            futures = [scheduler.submit([i, i + 1, i + 2]) for i in range(1, 5)]
            for i, future in enumerate(futures, 1):
                cache, parts = future.result(2)
                self.assertEqual(sum(parts, []), [i, 8*i + 1, 57*i + 9])
                self.assertEqual(cache, 57*i + 9)
            cache = futures[0].result()[0]
            a = scheduler.submit([9], cache)
            b = scheduler.submit([10], cache)
            self.assertEqual(a.result(2)[0], cache*7+9)
            self.assertEqual(b.result(2)[0], cache*7+10)
            self.assertEqual(scheduler.stats()['max_batch'], 4)
        finally:
            scheduler.close()
        self.assertFalse(scheduler.thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            scheduler.submit([1])

    def test_long_prefill_does_not_block_short_row(self):
        backend = Backend()
        scheduler = BatchScheduler(lambda: backend, slots=2, wait_ms=10, quantum=2)
        order = []
        try:
            a = scheduler.submit(list(range(30)))
            b = scheduler.submit([99])
            a.add_done_callback(lambda _: order.append('long'))
            b.add_done_callback(lambda _: order.append('short'))
            a.result(2)
            b.result(2)
            self.assertEqual(order, ['short', 'long'])
            self.assertLessEqual(max(len(ids) for batch in backend.batches for _, ids in batch), 2)
        finally:
            scheduler.close()

    def test_cancel_backpressure_and_slot_reuse(self):
        entered, release = threading.Event(), threading.Event()
        backend = Backend()
        step = backend.step
        def blocking(batch):
            entered.set()
            release.wait(2)
            return step(batch)
        backend.step = blocking
        scheduler = BatchScheduler(lambda: backend, slots=1, capacity=2, wait_ms=0)
        try:
            first = scheduler.submit([1])
            self.assertTrue(entered.wait(2))
            second = scheduler.submit([2])
            with self.assertRaisesRegex(RuntimeError, 'full'):
                scheduler.submit([3])
            self.assertTrue(second.cancel())
            release.set()
            self.assertEqual(first.result(2)[0], 1)
        finally:
            release.set()
            scheduler.close()
        self.assertTrue(second.cancelled())
        self.assertEqual(scheduler.stats()['outstanding'], 0)

    def test_failure_fails_active_and_pending(self):
        entered, release = threading.Event(), threading.Event()
        backend = Backend()
        def fail(batch):
            entered.set()
            release.wait(2)
            raise ValueError('backend failed')
        backend.step = fail
        scheduler = BatchScheduler(lambda: backend, slots=1, wait_ms=0)
        try:
            first = scheduler.submit([1])
            self.assertTrue(entered.wait(2))
            second = scheduler.submit([2])
            release.set()
            for future in (first, second):
                with self.assertRaisesRegex(ValueError, 'backend failed'):
                    future.result(2)
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                scheduler.submit([3])
        finally:
            release.set()
            scheduler.close()

    def test_startup_failure(self):
        def fail():
            raise ValueError('startup failed')
        with self.assertRaisesRegex(ValueError, 'startup failed'):
            BatchScheduler(fail)


if __name__ == '__main__':
    unittest.main()
