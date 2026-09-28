"""Interactive HS-Guard v10 service with token scoring and text rollback."""
from __future__ import annotations

import argparse
import base64
import hmac
import signal
import collections
import ipaddress
import json
import math
import os
import queue
import socket
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler
from service_limits import SessionStore, SessionBusy, StreamQueue, LimitedHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
HOLD_BACK = 3
# The UI marks context beyond the evaluated range.
VERIFIED_TOKENS = 8192
MAX_TOKENS = int(os.environ.get("GUARD_MAX_CONVERSATION_TOKENS", "49152"))
PREFILL_CHUNK = 2048
MAX_SESSIONS = 64
SESSION_IDLE_SECONDS = 3600
MAX_ACTIVE_CHATS = 16
SNAPSHOTS = 8
ACCESS_TOKEN = os.environ.get("GUARD_ACCESS_TOKEN", "")
MAX_CHAT_SECONDS = 1800
MAX_OUTPUT_CHARS = 196608
DRAIN_SECONDS = 30
MAX_BODY_BYTES = 4 * 1024 * 1024
SHUTDOWN = threading.Event()
# Custom model URLs may not target private networks (SSRF).
# Hosts listed here (comma-separated) may resolve to private addresses, e.g. an internal model gateway.
ALLOWED_PRIVATE_HOSTS = {h.strip().lower() for h in os.environ.get("GUARD_ALLOW_PRIVATE_HOSTS", "").split(",") if h.strip()}
# The built-in model key is read from the environment and never sent to the page.
BUILTIN = {"label": os.environ.get("GUARD_BUILTIN_LABEL", ""), "base_url": os.environ.get("GUARD_BUILTIN_BASE_URL", ""),
           "model": os.environ.get("GUARD_BUILTIN_MODEL", ""), "protocol": os.environ.get("GUARD_BUILTIN_PROTOCOL", "chat_completions"),
           "api_key": os.environ.get("GUARD_BUILTIN_API_KEY", "").strip()}
BUILTIN_MAX_TOKENS = int(os.environ.get("GUARD_BUILTIN_MAX_TOKENS", "4096"))
# Visitors' own endpoints (and the SSRF surface that comes with them) are off unless explicitly enabled.
CUSTOM_MODEL_ORIGINS = {u.strip().rstrip("/") for u in os.environ.get("GUARD_CUSTOM_MODEL_ORIGINS", "").split(",") if u.strip()}
ALLOW_CUSTOM_MODEL = os.environ.get("GUARD_ALLOW_CUSTOM_MODEL", "") == "1"
PROTECTED_BODY_KEYS = {"model", "messages", "input", "stream", "instructions", "system",
                       "max_tokens", "max_output_tokens", "max_completion_tokens", "n"}
STATIC_TYPES = {".js": "application/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
STATIC_DIR = HERE / "static"
ROLES = ("user", "assistant")
RISK_LABELS = ("safe", "unsafe", "controversial")
# ---- rules (mirrored in index.html) -------------------------------------------------------

def parse_rule(value):
    if not isinstance(value, dict):
        raise ValueError("rule must be an object")
    rule = {"rule": value.get("rule"), "score": value.get("score", "unsafe"),
            "T": float(value.get("T", 1.0)), "tau": float(value.get("tau", 0.5)),
            "k": int(value.get("k", 1)), "alpha": float(value.get("alpha", 0.1))}
    if rule["rule"] not in ("threshold", "ema", "argmax2", "endpoint", "off"):
        raise ValueError("unknown rule")
    if rule["score"] not in ("unsafe", "cut"):
        raise ValueError("score must be unsafe or cut")
    if not (0 < rule["T"] <= 20 and 0 < rule["tau"] < 1 and 1 <= rule["k"] <= 256 and 0 < rule["alpha"] <= 1):
        raise ValueError("rule parameters out of range")
    return rule


def probabilities(logits, temperature):
    scaled = [value / temperature for value in logits]
    top = max(scaled)
    weights = [math.exp(value - top) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def token_score(logits, rule):
    p = probabilities(logits, rule["T"])
    return p[1] if rule["score"] == "unsafe" else 1.0 - p[0]


def first_fire(rows, rule):
    """Index of the first firing row (content tokens of one message, in order) or -1."""
    kind = rule["rule"]
    if kind == "off" or not rows:
        return -1
    if kind == "endpoint":
        return len(rows) - 1 if token_score(rows[-1], rule) > rule["tau"] else -1
    if kind == "argmax2":
        positive = {1} if rule["score"] == "unsafe" else {1, 2}
        previous = None
        for i, row in enumerate(rows):
            label = max(range(3), key=row.__getitem__)
            if label in positive and label == previous:
                return i
            previous = label
        return -1
    if kind == "threshold":
        run = 0
        for i, row in enumerate(rows):
            run = run + 1 if token_score(row, rule) > rule["tau"] else 0
            if run >= rule["k"]:
                return i
        return -1
    bound, ema = math.log(rule["tau"] / (1 - rule["tau"])), None
    for i, row in enumerate(rows):
        s = min(max(token_score(row, rule), 1e-9), 1 - 1e-9)
        margin = math.log(s / (1 - s))
        ema = margin if ema is None else rule["alpha"] * margin + (1 - rule["alpha"]) * ema
        if ema > bound:
            return i
    return -1


# ---- conversation serialization (training format) ----------------------------------------

def guard_text(message):
    if message["role"] == "user":
        return message.get("content") or ""
    reasoning, content = message.get("reasoning") or "", message.get("content") or ""
    return reasoning + ("\n\n" if reasoning and content else "") + content


def serialize(messages):
    """'\\n\\n'.join(ROLE + ':\\n' + text) with (header_start, content_start, content_end) per message."""
    parts, spans, cursor = [], [], 0
    for index, message in enumerate(messages):
        if index:
            parts.append("\n\n")
            cursor += 2
        header = message["role"].upper() + ":\n"
        text = guard_text(message)
        spans.append((cursor, cursor + len(header), cursor + len(header) + len(text)))
        parts.append(header + text)
        cursor += len(header) + len(text)
    return "".join(parts), spans


def locate(offsets, spans, first=0):
    """(message index, content start, content end, header flag) for tokens first.. of the serialization."""
    result, m = [], 0
    for start, end in offsets[first:]:
        while m + 1 < len(spans) and start >= spans[m][2]:
            m += 1
        _, cs, ce = spans[m]
        a = max(start, cs) - cs
        b = max(min(end, ce), cs) - cs
        result.append((m, a, b, end <= cs))
    return result


def longest_common_prefix(left, right):
    n = min(len(left), len(right))
    if left[:n] == right[:n]:           # the usual case: pure append
        return n
    n = 0
    for old, new in zip(left, right):
        if old != new:
            break
        n += 1
    return n


# ---- model runtime and sessions ----------------------------------------------------------

from runtime_v10 import GuardRuntime, presets as v10_presets

class GuardSession:
    def __init__(self, runtime):
        self.rt = runtime
        self.lock = threading.Lock()
        self.ids, self.records, self.cache = [], [], None
        self.snapshots = collections.deque(maxlen=SNAPSHOTS)
        self.located, self.tail = [], []
        self.touched = time.time()

    def _commit(self, ids):
        for start in range(0, len(ids), PREFILL_CHUNK):
            chunk = ids[start:start + PREFILL_CHUNK]
            self.cache, records = self.rt.forward(chunk, self.cache)
            self.ids.extend(chunk)
            self.records.extend(records)
        self.snapshots.append((len(self.ids), self.rt.clone(self.cache)))

    def _rollback(self, position):
        while self.snapshots and self.snapshots[-1][0] > position:
            self.snapshots.pop()
        if self.snapshots:
            kept, snapshot = self.snapshots[-1]
            self.cache = self.rt.clone(snapshot)
        else:
            kept, self.cache = 0, None
        del self.ids[kept:]
        del self.records[kept:]

    def sync(self, messages, final):
        """Bring the state to the serialized conversation; returns the token payload from the first changed position."""
        self.touched = time.time()
        text, spans = serialize(messages)
        ids, offsets = self.rt.encode(text)
        if len(ids) > MAX_TOKENS:
            raise ValueError(f"对话已有 {len(ids)} 个 token，超过 {MAX_TOKENS}，请新建对话")
        previous = len(self.ids)
        common = longest_common_prefix(self.ids, ids)
        rolled_back = 0
        if common < len(self.ids):
            self._rollback(common)
            rolled_back = previous - len(self.ids)
        first = len(self.ids)
        target = max(len(self.ids), len(ids) - HOLD_BACK)
        if target > len(self.ids):
            self._commit(ids[len(self.ids):target])
        self.tail = []
        if final and target < len(ids):
            _, self.tail = self.rt.forward(ids[target:], self.rt.clone(self.cache))
        self.located = self.located[:first] + locate(offsets, spans, first)   # earlier tokens are unchanged
        records = self.records + self.tail
        tokens = []
        for i in range(first, len(records)):
            m, a, b, header = self.located[i]
            tokens.append({"i": i, "m": m, "a": a, "b": b, "h": int(header),
                           "p": int(i >= len(self.records)), **records[i]})
        return {"from": first, "tokens": tokens, "committed": len(self.records), "total": len(ids),
                "rolled_back": rolled_back}

    def evaluate(self, message_index, role, rule):
        key = "lu" if role == "user" else "la"
        records = self.records + self.tail
        positions = [i for i, (m, _, _, header) in enumerate(self.located[:len(records)])
                     if m == message_index and not header]
        rows = [records[i][key] for i in positions]
        best = max(range(len(rows)), key=lambda j: token_score(rows[j], rule), default=-1)
        result = {"n": len(rows), "max_score": round(token_score(rows[best], rule), 6) if rows else None,
                  "max_k": best, "fired": False}
        k = first_fire(rows, rule)
        if k >= 0:
            i = positions[k]
            result.update(fired=True, k=k, pos=i, char_end=self.located[i][2],
                          score=round(token_score(rows[k], rule), 6), provisional=i >= len(self.records))
        return result


class RuleTracker:
    """first_fire and the running maximum, fed one committed content token at a time (same rules)."""

    def __init__(self, rule):
        self.rule = rule
        self.bound = math.log(rule["tau"] / (1 - rule["tau"]))
        self.reset()

    def reset(self):
        self.count, self.run, self.ema, self.previous = 0, 0, None, None
        self.fired_k, self.best_k, self.best = -1, -1, -1.0

    def feed(self, logits):
        rule, k = self.rule, self.count
        s = token_score(logits, rule)
        if s > self.best:
            self.best, self.best_k = s, k
        self.count += 1
        if self.fired_k >= 0 or rule["rule"] in ("off", "endpoint"):
            return
        if rule["rule"] == "argmax2":
            label = max(range(3), key=logits.__getitem__)
            if label in ({1} if rule["score"] == "unsafe" else {1, 2}) and label == self.previous:
                self.fired_k = k
            self.previous = label
        elif rule["rule"] == "threshold":
            self.run = self.run + 1 if s > rule["tau"] else 0
            if self.run >= rule["k"]:
                self.fired_k = k
        else:
            c = min(max(s, 1e-9), 1 - 1e-9)
            margin = math.log(c / (1 - c))
            self.ema = margin if self.ema is None else rule["alpha"] * margin + (1 - rule["alpha"]) * self.ema
            if self.ema > self.bound:
                self.fired_k = k


# ---- upstream readers ------------------------------------------------------------------

def public_address(address):
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def check_model_url(url):
    """Reject model URLs that resolve into private, loopback, link-local or reserved space."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Base URL 需要以 http:// 或 https:// 开头")
    if parts.username or parts.password or parts.fragment:
        raise ValueError('model URL may not contain credentials or fragments')
    origin = f'{parts.scheme}://{parts.netloc}'
    if origin not in CUSTOM_MODEL_ORIGINS:
        raise ValueError('model origin is not in GUARD_CUSTOM_MODEL_ORIGINS')
    host = parts.hostname.lower()
    if host in ALLOWED_PRIVATE_HOSTS:
        return host
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except socket.gaierror as error:
        raise ValueError(f"解析不了模型地址 {host}：{error}") from None
    if not infos or not all(public_address(info[4][0]) for info in infos):
        raise ValueError(f"模型地址 {host} 解析到内网或保留地址，服务不允许访问（需要的话请管理员加白名单）")
    return host


def model_url(base, protocol):
    """Base URL -> endpoint. A bare host gets /v1 (OpenAI SDK convention); index.html mirrors this."""
    base = base.strip().rstrip("/")
    endpoint = {"responses": "/responses", "anthropic_messages": "/messages"}.get(protocol, "/chat/completions")
    if base.endswith(endpoint):
        return base
    return base + ("/v1" if urlsplit(base).path in ("", "/") else "") + endpoint


def builtin_available():
    return bool(BUILTIN["api_key"] and BUILTIN["base_url"] and BUILTIN["model"])


def read_llm(config, history, out, stop):
    import httpx
    try:
        trusted = bool(config.get("use_builtin"))
        if trusted:
            if not builtin_available():
                raise ValueError("服务端还没有配置内置模型的 key")
            requested = config.get("max_tokens")
            limit = max(1, min(int(requested), BUILTIN_MAX_TOKENS)) if requested not in (None, "") else BUILTIN_MAX_TOKENS
            config = {**config, **{k: BUILTIN[k] for k in ("base_url", "model", "protocol", "api_key")}, "max_tokens": limit}
        elif not ALLOW_CUSTOM_MODEL:
            raise ValueError("这个服务只开放内置模型")
        protocol = config.get("protocol") if config.get("protocol") in ("responses", "anthropic_messages") else "chat_completions"
        url = model_url(str(config.get("base_url") or ""), protocol)
        host = urlsplit(url).hostname if trusted else check_model_url(url)
        system = (config.get("system_prompt") or "").strip()
        messages = [{"role": "system", "content": system}] if system and protocol == "chat_completions" else []
        for message in history:
            if message["role"] == "user":
                messages.append({"role": "user", "content": message.get("content") or ""})
            else:
                messages.append({"role": "assistant", "content": message.get("content") or "（回答被截断）"})
        if protocol == "responses":
            body = {"model": config.get("model") or "", "input": messages, "stream": True}
            if system:
                body["instructions"] = system
        elif protocol == "anthropic_messages":
            body = {"model": config.get("model") or "", "messages": messages, "stream": True, "max_tokens": 4096}
            if system:
                body["system"] = system
        else:
            body = {"model": config.get("model") or "", "messages": messages, "stream": True}
        if config.get("temperature") not in (None, ""):
            body["temperature"] = float(config["temperature"])
        if config.get("max_tokens") not in (None, ""):
            body["max_output_tokens" if protocol == "responses" else "max_tokens"] = int(config["max_tokens"])
        extra = config.get("extra_body") or {}
        if not isinstance(extra, dict):
            raise ValueError("额外参数需要是 JSON 对象")
        body.update({k: v for k, v in extra.items() if k not in PROTECTED_BODY_KEYS})
        output_key = 'max_output_tokens' if protocol == 'responses' else 'max_tokens'
        body[output_key] = max(1, min(int(body.get(output_key, BUILTIN_MAX_TOKENS)), BUILTIN_MAX_TOKENS))
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if protocol == "anthropic_messages":
            headers["anthropic-version"] = "2023-06-01"
            if config.get("api_key"):
                headers["x-api-key"] = str(config["api_key"]).strip()
        if config.get("api_key"):
            headers["Authorization"] = "Bearer " + str(config["api_key"]).strip()
        timeout = httpx.Timeout(connect=20.0, read=30.0, write=30.0, pool=20.0)
        with httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client, \
                client.stream("POST", url, json=body, headers=headers) as response:
            # re-check the address actually connected to (DNS may have changed since check_model_url)
            stream = response.extensions.get("network_stream")
            peer = stream.get_extra_info("server_addr") if stream is not None else None
            if not trusted and host not in ALLOWED_PRIVATE_HOSTS and (not peer or not public_address(str(peer[0]))):
                out.put(("error", f"模型地址 {host} 实际连到了内网或无法确认的地址，已中止"))
                return
            if response.status_code != 200:
                out.put(("error", f"模型接口返回 HTTP {response.status_code}"))
                return
            for line in response.iter_lines():
                if stop.is_set():
                    return
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("error") or chunk.get("type") in ("error", "response.failed"):
                    out.put(("error", "上游模型流返回错误"))
                    return
                if protocol == "anthropic_messages":
                    kind = chunk.get("type", "")
                    delta = chunk.get("delta") or {}
                    if kind == "content_block_delta" and delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                        out.put(("content", delta["text"]))
                    elif kind == "content_block_delta" and delta.get("type") == "thinking_delta" \
                            and isinstance(delta.get("thinking"), str):
                        out.put(("reasoning", delta["thinking"]))
                    elif kind == "message_delta" and delta.get("stop_reason"):
                        out.put(("finish", str(delta["stop_reason"])))
                    elif kind == "message_stop":
                        break
                    continue
                if protocol == "responses":
                    kind = chunk.get("type", "")
                    if kind == "response.output_text.delta" and isinstance(chunk.get("delta"), str):
                        out.put(("content", chunk["delta"]))
                    elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta") \
                            and isinstance(chunk.get("delta"), str):
                        out.put(("reasoning", chunk["delta"]))
                    elif kind in ("response.completed", "response.incomplete"):
                        out.put(("finish", kind.split(".")[-1]))
                        break
                    continue
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if isinstance(reasoning, str) and reasoning:
                        out.put(("reasoning", reasoning))
                    if isinstance(delta.get("content"), str) and delta["content"]:
                        out.put(("content", delta["content"]))
                    if choice.get("finish_reason"):
                        out.put(("finish", str(choice["finish_reason"])))
    except Exception as error:
        out.put(("error", f"上游请求失败：{type(error).__name__}"))
    finally:
        out.put(("end", None))


def read_replay(replay, out, stop):
    try:
        size = max(1, min(64, int(replay.get("chars", 2))))
        delay = max(0.0, min(2.0, float(replay.get("interval_ms", 40)) / 1000))
        for kind in ("reasoning", "content"):
            text = replay.get(kind) or ""
            for start in range(0, len(text), size):
                if stop.is_set():
                    return
                out.put((kind, text[start:start + size]))
                if stop.wait(delay):
                    return
        out.put(("finish", "stop"))
    except Exception as error:
        out.put(("error", str(error)))
    finally:
        out.put(("end", None))


# ---- HTTP ------------------------------------------------------------------------------

ACTIVE_CHATS = threading.BoundedSemaphore(MAX_ACTIVE_CHATS)


class State:
    runtime = None
    store = None
    error = None
    selfcheck = None
    loaded_at = None
    started_at = time.time()


def load_model(bundle):
    try:
        runtime = GuardRuntime(bundle)
        check = selfcheck(runtime)
        if not check["tokenizer_match"] or check["stream_positions"] != check["tokens"] or check["max_prob_diff_stream_vs_whole"] > .05:
            raise RuntimeError(f"Streaming self-check failed: {check}")
        State.selfcheck = check
        State.store = SessionStore(lambda: GuardSession(runtime), MAX_SESSIONS, SESSION_IDLE_SECONDS)
        State.loaded_at = time.time()
        if SHUTDOWN.is_set():
            runtime.close()
            return
        State.runtime = runtime
        print("model ready", json.dumps(State.selfcheck), flush=True)
    except Exception as error:
        if "runtime" in locals():
            runtime.close()
        State.error = f"{type(error).__name__}: {error}"
        traceback.print_exc()


def selfcheck(runtime):
    """Stream a benign two-turn text in small pieces and compare with one whole forward."""
    messages = [{"role": "user", "content": "请简单介绍一下光合作用。"},
                {"role": "assistant", "content": "光合作用是绿色植物利用光能，把二氧化碳和水合成有机物并释放氧气的过程。"
                 "它分为光反应和暗反应两个阶段：光反应在类囊体膜上进行，产生 ATP 和 NADPH；"
                 "暗反应在叶绿体基质中进行，把二氧化碳固定为糖类。Photosynthesis also sustains most food chains."}]
    text, _ = serialize(messages)
    ids, _ = runtime.encode(text)
    reference_ids = runtime.tokenizer.encode(text, add_special_tokens=False).ids
    _, whole = runtime.forward(ids, None)
    session = GuardSession(runtime)
    reply = messages[1]["content"]
    started = time.perf_counter()
    for cut in range(0, len(reply), 3):
        session.sync([messages[0], {"role": "assistant", "content": reply[:cut]}], final=False)
    session.sync(messages, final=True)
    seconds = time.perf_counter() - started
    streamed = session.records + session.tail
    diff = 0.0
    for got, want in zip(streamed, whole):
        for key in ("lu", "la"):
            p, q = probabilities(got[key], 1.0), probabilities(want[key], 1.0)
            diff = max(diff, max(abs(x - y) for x, y in zip(p, q)))
    return {"tokens": len(ids), "tokenizer_match": ids == reference_ids, "stream_positions": len(streamed),
            "max_prob_diff_stream_vs_whole": round(diff, 5), "stream_seconds": round(seconds, 2)}


def ready():
    return not SHUTDOWN.is_set() and State.runtime is not None and State.runtime.healthy()


class Handler(BaseHTTPRequestHandler):
    server_version = "GuardDemo/1"

    def authorized(self):
        if not ACCESS_TOKEN:
            return True
        header = self.headers.get('Authorization', '')
        credential = ''
        if header.startswith('Bearer '):
            credential = header[7:]
        elif header.startswith('Basic '):
            try:
                user, credential = base64.b64decode(header[6:], validate=True).decode().split(':', 1)
                if user != 'guard':
                    credential = ''
            except (ValueError, UnicodeError):
                credential = ''
        if hmac.compare_digest(credential.encode(), ACCESS_TOKEN.encode()):
            return True
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm="HS-Guard", charset="UTF-8"')
        self.send_header('Content-Length', '0')
        self.end_headers()
        return False

    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _json(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length', [])) != 1:
            raise ValueError('one Content-Length header is required')
        length = int(self.headers['Content-Length'])
        if not 0 <= length <= MAX_BODY_BYTES:
            raise ValueError('request body size is out of range')
        if self.headers.get_content_type() != 'application/json':
            raise ValueError('Content-Type must be application/json')
        content = self.rfile.read(length)
        if len(content) != length:
            raise ValueError('incomplete request body')
        body = json.loads(content or b'{}')
        if not isinstance(body, dict):
            raise ValueError('request body must be an object')
        return body

    def do_GET(self):
        if self.path in ('/api/health', '/api/live'):
            live = State.error is None and (State.runtime is None or State.runtime.healthy())
            status = ready() if self.path == '/api/health' else live
            self._json(200 if status else 503, {'ready': ready(), 'live': live, 'draining': SHUTDOWN.is_set()})
            return
        if not self.authorized():
            return
        if self.path in ("/", "/index.html"):
            body = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/static/"):
            name = self.path[len("/static/"):].split("?", 1)[0]
            path = STATIC_DIR / name
            if "/" in name or name.startswith(".") or path.suffix not in STATIC_TYPES or not path.is_file():
                self._json(404, {"error": "not found"})
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", STATIC_TYPES[path.suffix])
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/info":
            self._json(200, info())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.authorized():
            return
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            self._json(403, {'error': 'cross-origin request rejected'})
            return
        if SHUTDOWN.is_set():
            self._json(503, {'error': 'service is draining'})
            return
        try:
            body = self._body()
        except Exception as error:
            self._json(400, {"error": str(error)})
            return
        if self.path == "/api/reset":
            if State.store:
                try:
                    State.store.drop(str(body.get("session_id")))
                except SessionBusy as error:
                    self._json(409, {'error': str(error)})
                    return
            self._json(200, {"ok": True})
        elif self.path == "/api/chat":
            if not ready():
                self._json(503, {"error": State.error or "模型还在加载"})
                return
            self.chat(body)
        else:
            self._json(404, {"error": "not found"})

    def send_event(self, name, payload):
        self.wfile.write(f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode())
        self.wfile.flush()

    def chat(self, body):
        try:
            session_id = body['session_id']
            if not isinstance(session_id, str) or not 1 <= len(session_id) <= 128:
                raise ValueError('session_id must be 1–128 characters')
            if not isinstance(body['messages'], list) or not 1 <= len(body['messages']) <= 512:
                raise ValueError('messages must contain 1–512 items')
            if any(not isinstance(m, dict) or any(m.get(k) is not None and not isinstance(m.get(k), str)
                   for k in ('content', 'reasoning')) for m in body['messages']):
                raise ValueError('message content and reasoning must be strings')
            if sum(len(m.get(k) or '') for m in body['messages'] for k in ('content', 'reasoning')) > 262144:
                raise ValueError('conversation text exceeds 262144 characters')
            replay = body.get('replay') or {}
            if not isinstance(replay, dict) or any(replay.get(k) is not None and not isinstance(replay.get(k), str)
                                                 for k in ('content', 'reasoning')):
                raise ValueError('replay content and reasoning must be strings')
            if sum(len(replay.get(k) or '') for k in ('content', 'reasoning')) > MAX_OUTPUT_CHARS:
                raise ValueError('replay text exceeds output character limit')
            history = [{"role": m["role"], "content": str(m.get("content") or ""),
                        "reasoning": str(m.get("reasoning") or "")} for m in body["messages"]]
            if not history or history[-1]["role"] != "user" or any(m["role"] not in ROLES for m in history):
                raise ValueError("messages 必须以一条 user 消息结尾")
            guard = body.get("guard") or {}
            action = guard.get("action", "cut")
            user_rule, assistant_rule = parse_rule(guard["user_rule"]), parse_rule(guard["assistant_rule"])
            mode = body.get("mode", "llm")
            if mode not in ('llm', 'replay') or action not in ('cut', 'observe', 'mark'):
                raise ValueError('invalid mode or guard action')
        except Exception as error:
            self._json(400, {"error": f"请求格式不对：{error}"})
            return
        if not ACTIVE_CHATS.acquire(blocking=False):
            self._json(503, {"error": f"同时进行的对话已达上限 {MAX_ACTIVE_CHATS}，请稍后再试"})
            return
        try:
            try:
                session = State.store.acquire(session_id)
            except SessionBusy as error:
                self._json(409, {'error': str(error)})
                return
            try:
                self.stream_chat(session, history, action, user_rule, assistant_rule, mode, body)
            finally:
                State.store.release(session_id, discard=getattr(session, 'failed', False))
        finally:
            ACTIVE_CHATS.release()

    def stream_chat(self, session, history, action, user_rule, assistant_rule, mode, body):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        stop = threading.Event()
        try:
            self.converse(session, history, action, user_rule, assistant_rule, mode, body, stop)
        except (BrokenPipeError, ConnectionResetError):
            session.failed = True
            stop.set()
        except Exception as error:
            session.failed = True
            stop.set()
            traceback.print_exc()
            try:
                self.send_event("error", {"message": f"审核中止：{type(error).__name__}"})
            except OSError:
                pass
        finally:
            stop.set()

    def converse(self, session, history, action, user_rule, assistant_rule, mode, body, stop):
        started = time.perf_counter()
        user_index = len(history) - 1
        payload = session.sync(history, final=True)
        self.send_event("tokens", payload)
        verdict = session.evaluate(user_index, "user", user_rule)
        verdict["ms"] = round((time.perf_counter() - started) * 1000, 1)
        self.send_event("user_verdict", verdict)
        if verdict["fired"] and action == "cut":
            self.send_event("done", {"blocked": True, "latency": State.runtime.latency()})
            return

        out = StreamQueue(stop, max_chars=MAX_OUTPUT_CHARS)
        if mode == "replay":
            reader = threading.Thread(target=read_replay, args=(body.get("replay") or {}, out, stop), daemon=True)
        else:
            reader = threading.Thread(target=read_llm, args=(body.get("llm") or {}, history, out, stop), daemon=True)
        reader.start()
        reply = {"role": "assistant", "reasoning": "", "content": ""}
        cut, ended, finish, first_delta = None, False, None, None
        tracker, positions, scanned = RuleTracker(assistant_rule), [], 0   # committed content tokens of the reply
        stream_started = time.perf_counter()
        deadline = time.monotonic() + MAX_CHAT_SECONDS
        while not ended:
            if time.monotonic() >= deadline or SHUTDOWN.is_set():
                raise TimeoutError('stream deadline or shutdown')
            try:
                items = [out.get(timeout=10)]
            except queue.Empty:
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()
                continue
            while len(items) < 128:
                try:
                    items.append(out.get_nowait())
                except queue.Empty:
                    break
            delta = {"reasoning": "", "content": ""}
            for kind, value in items:
                if kind in ("reasoning", "content"):
                    reply[kind] += value
                    delta[kind] += value
                    first_delta = first_delta or time.perf_counter()
                elif kind == "finish":
                    finish = value
                elif kind == "error":
                    session.failed = True
                    self.send_event("error", {"message": value})
                    stop.set()
                    return
                elif kind == "end":
                    ended = True
            if delta["reasoning"] or delta["content"]:
                self.send_event("delta", delta)
            if cut is not None and action == "cut":
                continue
            if not (delta["reasoning"] or delta["content"]):
                continue
            payload = session.sync(history + [reply], final=False)
            self.send_event("tokens", payload)
            if cut is None:
                if payload["from"] < scanned:              # BPE rollback into scored tokens: rescore
                    tracker.reset()
                    positions.clear()
                    scanned = 0
                for i in range(scanned, len(session.records)):
                    m, _, _, header = session.located[i]
                    if m == len(history) and not header:
                        positions.append(i)
                        tracker.feed(session.records[i]["la"])
                scanned = len(session.records)
                if tracker.fired_k >= 0:
                    i = positions[tracker.fired_k]
                    cut = {"n": tracker.count, "max_score": round(tracker.best, 6), "max_k": tracker.best_k,
                           "fired": True, "k": tracker.fired_k, "pos": i, "char_end": session.located[i][2],
                           "score": round(token_score(session.records[i]["la"], assistant_rule), 6), "provisional": False,
                           "at_ms": round((time.perf_counter() - stream_started) * 1000),
                           "received_chars": len(guard_text(reply)), "action": action}
                    self.send_event("assistant_verdict", cut)
                    if action == "cut":
                        stop.set()
                        break
        if cut is None or action != "cut":
            payload = session.sync(history + [reply], final=True)
            self.send_event("tokens", payload)
            if cut is None and guard_text(reply):
                verdict = session.evaluate(len(history), "assistant", assistant_rule)
                verdict.update(at_ms=round((time.perf_counter() - stream_started) * 1000),
                               received_chars=len(guard_text(reply)), action=action, at_end=True)
                self.send_event("assistant_verdict", verdict)
        self.send_event("done", {"finish_reason": finish, "cut": cut is not None and action == "cut",
                                 "first_delta_ms": round((first_delta - stream_started) * 1000) if first_delta else None,
                                 "total_ms": round((time.perf_counter() - started) * 1000),
                                 "latency": State.runtime.latency()})


def info():
    runtime = State.runtime
    meta = runtime.meta if runtime else {}
    thresholds = meta.get("thresholds", {})
    return {
        "ready": ready(), "error": State.error,
        "loading_seconds": None if State.loaded_at else round(time.time() - State.started_at),
        "checkpoint_sha256": meta.get("checkpoint_sha256"), "candidate": meta.get("candidate"),
        "selection_status": meta.get("selection_status"), "device": meta.get("device"),
        "canonical_quality_gate_pass": meta.get("canonical_quality_gate_pass"),
        "thresholds": thresholds, "hold_back": HOLD_BACK, "max_tokens": MAX_TOKENS, "verified_tokens": VERIFIED_TOKENS,
        "categories": None, "risk_labels": RISK_LABELS,
        "presets": v10_presets(thresholds) if runtime else None,
        "model_label": meta.get("candidate"), "engine": meta.get("engine"),
        "selfcheck": State.selfcheck, "latency": runtime.latency() if runtime else {},
        "sessions": State.store.stats()["sessions"] if State.store else 0,
        "active_chats": State.store.stats()["active"] if State.store else 0,
        "builtin": {"available": builtin_available(), "label": BUILTIN["label"] or BUILTIN["model"],
                    "model": BUILTIN["model"], "host": urlsplit(BUILTIN["base_url"]).hostname or "",
                    "max_tokens": BUILTIN_MAX_TOKENS},
        "custom_model_allowed": ALLOW_CUSTOM_MODEL,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--static", type=Path, default=STATIC_DIR, help="directory with vue/tdesign dist files")
    args = parser.parse_args()
    globals()["STATIC_DIR"] = args.static
    if os.environ.get('GUARD_PRODUCTION') == '1' and not ACCESS_TOKEN:
        raise ValueError('GUARD_PRODUCTION requires GUARD_ACCESS_TOKEN')
    if ACCESS_TOKEN and len(ACCESS_TOKEN) < 24:
        raise ValueError('GUARD_ACCESS_TOKEN must contain at least 24 characters')
    server = LimitedHTTPServer((args.host, args.port), Handler)
    threading.Thread(target=load_model, args=(args.bundle.resolve(),), daemon=True).start()
    def maintenance():
        while not SHUTDOWN.wait(30):
            if State.store:
                State.store.prune()
    threading.Thread(target=maintenance, daemon=True).start()
    def stop_server(signum, frame):
        if not SHUTDOWN.is_set():
            SHUTDOWN.set()
            threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    print(f"listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        SHUTDOWN.set()
        deadline = time.monotonic() + DRAIN_SECONDS
        while State.store and State.store.stats()['active'] and time.monotonic() < deadline:
            time.sleep(.05)
        server.server_close()
        if State.runtime:
            State.runtime.close(timeout=5)


if __name__ == "__main__":
    main()
