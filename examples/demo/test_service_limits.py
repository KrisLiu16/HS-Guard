"""CPU checks for session leases, producer backpressure and HTTP boundaries."""
import base64
import concurrent.futures
import http.client
import json
import queue
import socket
import threading
import time
import unittest
from unittest.mock import patch
from service_limits import SessionBusy, SessionStore, StreamQueue, LimitedHTTPServer
import server


class LimitsTests(unittest.TestCase):
    def test_active_sessions_cannot_be_reset_evicted_or_reentered(self):
        store = SessionStore(object, capacity=2, idle_seconds=.01)
        first = store.acquire('a')
        with self.assertRaises(SessionBusy): store.acquire('a')
        with self.assertRaises(SessionBusy): store.drop('a')
        second = store.acquire('b')
        with self.assertRaises(SessionBusy): store.acquire('c')
        time.sleep(.02)
        store.prune()
        self.assertEqual(store.stats(), {'sessions': 2, 'active': 2})
        store.release('b')
        third = store.acquire('c')
        self.assertIs(store.sessions['a'][0], first)
        self.assertIsNot(third, second)
        store.release('a')
        self.assertIs(store.acquire('a'), first)
        store.release('a')
        store.release('c')
        time.sleep(.02)
        store.prune()
        self.assertEqual(store.stats()['sessions'], 0)

    def test_cancel_releases_blocked_producer(self):
        stop = threading.Event()
        out = StreamQueue(stop, maxsize=1)
        out.put(('content', 'a'))
        worker = threading.Thread(target=out.put, args=(('content', 'b'),))
        worker.start()
        time.sleep(.02)
        self.assertTrue(worker.is_alive())
        stop.set()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(out.qsize(), 1)

    def test_total_output_limit_includes_reasoning(self):
        out = StreamQueue(threading.Event(), max_chars=3)
        out.put(('reasoning', 'ab'))
        with self.assertRaises(ValueError): out.put(('content', 'cd'))


class HTTPTests(unittest.TestCase):
    def setUp(self):
        server.SHUTDOWN.clear()
        self.http = LimitedHTTPServer(('127.0.0.1', 0), server.Handler, socket_timeout=.2)
        self.worker = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.worker.start()
        self.port = self.http.server_port

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.worker.join(1)
        server.SHUTDOWN.clear()

    def request(self, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=2)
        try:
            conn.request('GET' if body is None else 'POST', path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def test_bearer_basic_and_unauthenticated_probe(self):
        with patch.object(server, 'ACCESS_TOKEN', 'test-access-token-24-characters'):
            self.assertEqual(self.request('/api/info')[0], 401)
            status, _ = self.request('/api/health')
            self.assertIn(status, (200, 503))
            self.assertEqual(self.request('/api/info', headers={'Authorization':'Bearer test-access-token-24-characters'})[0], 200)
            value=base64.b64encode(b'guard:test-access-token-24-characters').decode()
            self.assertEqual(self.request('/api/info', headers={'Authorization':'Basic '+value})[0], 200)
            self.assertEqual(self.request('/api/info', headers={'Authorization':'Basic !!!'})[0], 401)

    def test_chat_rejects_invalid_text_before_model_work(self):
        rule=server.v10_presets({'user':.1,'assistant':.1})
        runtime=type('Runtime',(),{'healthy':lambda self:True})()
        body={'session_id':'text','messages':[{'role':'user','content':{'nested':'object'}}],
              'mode':'replay','guard':{'action':'observe','user_rule':rule['user'][0],
                                     'assistant_rule':rule['assistant'][0]}}
        with patch.object(server.State,'runtime',runtime):
            self.assertEqual(self.request('/api/chat',json.dumps(body).encode(),{'Content-Type':'application/json'})[0],400)
            body['messages'][0]['content']='x'*262145
            self.assertEqual(self.request('/api/chat',json.dumps(body).encode(),{'Content-Type':'application/json'})[0],400)

    def test_body_and_origin_validation(self):
        self.assertEqual(self.request('/api/reset', b'[]', {'Content-Type':'application/json'})[0], 400)
        self.assertEqual(self.request('/api/reset', b'{}', {'Content-Type':'text/plain'})[0], 400)
        self.assertEqual(self.request('/api/reset', b'{}', {'Content-Type':'application/json', 'Origin':'https://example.com'})[0], 403)
        self.assertEqual(self.request('/api/reset', b'{}', {'Content-Type':'application/json'})[0], 200)

    def test_duplicate_and_negative_content_length(self):
        for headers in [b'Content-Length: -1\r\n', b'Content-Length: 2\r\nContent-Length: 2\r\n']:
            with socket.create_connection(('127.0.0.1', self.port), timeout=2) as conn:
                conn.sendall(b'POST /api/reset HTTP/1.0\r\nContent-Type: application/json\r\n'+headers+b'\r\n{}')
                self.assertIn(b'400', conn.recv(4096).split(b'\r\n')[0])

    def test_unready_and_draining(self):
        fake = type('Runtime', (), {'healthy': lambda self: False})()
        with patch.object(server.State, 'runtime', fake):
            self.assertEqual(self.request('/api/health')[0], 503)
            self.assertEqual(self.request('/api/live')[0], 503)
        server.SHUTDOWN.set()
        self.assertEqual(self.request('/api/reset', b'{}', {'Content-Type':'application/json'})[0], 503)

    def test_failed_stream_discards_partial_session(self):
        class FailedSession:
            def sync(self, *args, **kwargs):
                raise RuntimeError('injected GPU failure after partial commit')
        runtime=type('Runtime',(),{'healthy':lambda self:True})()
        store=SessionStore(FailedSession)
        rule=server.v10_presets({'user':.1,'assistant':.1})
        body={'session_id':'partial','messages':[{'role':'user','content':'Hello'}],
              'mode':'replay','guard':{'action':'observe','user_rule':rule['user'][0],
                                     'assistant_rule':rule['assistant'][0]}}
        with patch.object(server.State,'runtime',runtime), patch.object(server.State,'store',store):
            status,payload=self.request('/api/chat',json.dumps(body).encode(),{'Content-Type':'application/json'})
            self.assertEqual(status,200)
            self.assertIn(b'event: error',payload)
            self.assertEqual(store.stats(),{'sessions':0,'active':0})

    def test_connection_limit_and_read_timeout(self):
        srv = LimitedHTTPServer(('127.0.0.1',0), server.Handler, max_connections=1, socket_timeout=.1)
        worker=threading.Thread(target=srv.serve_forever,daemon=True); worker.start()
        try:
            with socket.create_connection(srv.server_address, timeout=2) as first:
                first.sendall(b'GET / HTTP/1.1\r\n')
                time.sleep(.02)
                with socket.create_connection(srv.server_address,timeout=2) as second:
                    self.assertEqual(second.recv(1),b'')
                time.sleep(.15)
            conn=http.client.HTTPConnection(*srv.server_address,timeout=2)
            conn.request('GET','/api/live');self.assertIn(conn.getresponse().status,(200,503));conn.close()
        finally:
            srv.shutdown();srv.server_close();worker.join(1)


class UpstreamTests(unittest.TestCase):
    def test_output_cap_cannot_be_overridden_by_extra_body(self):
        import httpx
        captured = []
        class Response:
            status_code = 200
            extensions = {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def iter_lines(self): return iter(['data: [DONE]'])
        class Client:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def stream(self, *args, **kwargs):
                captured.append(kwargs)
                return Response()
        for protocol in ('chat_completions','responses','anthropic_messages'):
            builtin=dict(label='test',base_url='https://example.com',model='test',protocol=protocol,api_key='test-key')
            with patch.object(server,'BUILTIN',builtin), patch.object(server,'BUILTIN_MAX_TOKENS',32), patch.object(httpx,'Client',Client):
                out=StreamQueue(threading.Event())
                server.read_llm({'use_builtin':True,'max_tokens':999,'extra_body':{'max_tokens':99999,'max_output_tokens':99999,'max_completion_tokens':99999,'n':100,'model':'wrong'}}, [{'role':'user','content':'Hello'}], out, out.stop)
                body=captured[-1]['json']
                key='max_output_tokens' if protocol=='responses' else 'max_tokens'
                self.assertEqual(body[key],32)
                self.assertNotIn('max_completion_tokens',body)
                self.assertNotIn('n',body)
                self.assertEqual(body['model'],'test')
                events=[]
                while not out.empty(): events.append(out.get_nowait())
                self.assertEqual(events,[('end',None)])

    def test_custom_origin_allowlist_precedes_dns(self):
        with patch.object(server,'CUSTOM_MODEL_ORIGINS',{'https://allowed.example'}), patch.object(server.socket,'getaddrinfo') as dns:
            with self.assertRaisesRegex(ValueError,'origin'):
                server.check_model_url('https://untrusted.example/v1')
            dns.assert_not_called()
            with self.assertRaisesRegex(ValueError,'credentials'):
                server.check_model_url('https://user:password@allowed.example/v1')

if __name__ == '__main__':
    unittest.main()
