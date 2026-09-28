"""Request limits shared by the HTTP server and streaming readers."""
from collections import OrderedDict
from http.server import ThreadingHTTPServer
import queue
import threading
import time


class SessionBusy(RuntimeError):
    pass


class SessionStore:
    def __init__(self, factory, capacity=64, idle_seconds=3600):
        self.factory = factory
        self.capacity = capacity
        self.idle_seconds = idle_seconds
        self.lock = threading.Lock()
        self.sessions = OrderedDict()
        self.active = set()

    def _prune(self, now):
        for key, (_, touched) in list(self.sessions.items()):
            if key not in self.active and now - touched >= self.idle_seconds:
                del self.sessions[key]

    def acquire(self, key):
        with self.lock:
            self._prune(time.monotonic())
            if key in self.active:
                raise SessionBusy('会话正在生成回答')
            entry = self.sessions.pop(key, None)
            if entry is None:
                while len(self.sessions) >= self.capacity:
                    idle = next((k for k in self.sessions if k not in self.active), None)
                    if idle is None:
                        raise SessionBusy('会话容量已满')
                    del self.sessions[idle]
                session = self.factory()
            else:
                session = entry[0]
            self.active.add(key)
            self.sessions[key] = (session, time.monotonic())
            return session

    def release(self, key, discard=False):
        with self.lock:
            self.active.discard(key)
            if discard:
                self.sessions.pop(key, None)
            if key in self.sessions:
                session, _ = self.sessions.pop(key)
                self.sessions[key] = (session, time.monotonic())

    def drop(self, key):
        with self.lock:
            if key in self.active:
                raise SessionBusy('会话正在生成回答，结束后才能重置')
            self.sessions.pop(key, None)

    def prune(self):
        with self.lock:
            self._prune(time.monotonic())

    def stats(self):
        with self.lock:
            return {'sessions': len(self.sessions), 'active': len(self.active)}


class StreamQueue(queue.Queue):
    """Bound queued events and total generated text; cancellation unblocks producers."""
    def __init__(self, stop, maxsize=128, max_chars=196608):
        super().__init__(maxsize=maxsize)
        self.stop = stop
        self.max_chars = max_chars
        self.chars = 0

    def put(self, item, block=True, timeout=None):
        kind, value = item
        if kind in ('reasoning', 'content'):
            self.chars += len(value)
            if self.chars > self.max_chars:
                raise ValueError('模型输出超过字符上限')
        while not self.stop.is_set():
            try:
                super().put(item, timeout=.1)
                return
            except queue.Full:
                continue


class LimitedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, address, handler, max_connections=64, socket_timeout=30):
        self.connections = threading.BoundedSemaphore(max_connections)
        self.socket_timeout = socket_timeout
        super().__init__(address, handler)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(self.socket_timeout)
        return request, address

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()
