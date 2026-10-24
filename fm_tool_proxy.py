#!/usr/bin/env python3
"""fm_tool_proxy: OpenAI-compat shim between Goose (or any client) and Apple's `fm serve`.

  Goose --(OPENAI_HOST=http://127.0.0.1:1977)--> proxy --> fm serve (127.0.0.1:1976)

Fixes (stdlib only, Python 3.9+):
 1. Tool-call regression (fm 2.0.68.x): fm serve never sets `tool_calls`; when the model does
    call a tool it writes it into `content`, e.g.
      <start_of_turn>model\n{"tool_call": [{"name": "get_weather", "arguments": {...}}]}
    We strip template markers and parse that (plus `<tool_call>{..}</tool_call>`, bare
    {"name","arguments"} and `[default_api:fn{k:v}]` variants) into real OpenAI `tool_calls`,
    finish_reason "tool_calls". Unknown tool names are dropped (no hallucinated tools).
 2. Always asks upstream for stream:false (fm serve streams when `stream` is omitted) and
    re-synthesises SSE for clients that asked to stream (Goose does).
 3. Context overflow: fm serve answers HTTP 500 "The session's transcript exceeded the model's
    context size." Goose treats 5xx as a retryable server error and never compacts. We return
    400 {"error":{"code":"context_length_exceeded",...}} so Goose's ContextLengthExceeded path
    (auto-compaction) fires.
 4. Adds `meta.n_ctx` to /v1/models entries (Goose's openai provider reads it, llama.cpp style).
 5. Backfills missing tool `function.description` (fm serve 400s the whole request otherwise).
 6. tool_choice: "none" -> drop tools; "required"/named -> dropped (fm serve 500s on them).
Env: PROXY_PORT=1977 FM_UPSTREAM=http://127.0.0.1:1976 FM_CTX=4096 PROXY_LOG=logs/proxy.jsonl
"""
import json, os, re, sys, time, uuid, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PROXY_PORT", "1977"))
UP = os.environ.get("FM_UPSTREAM", "http://127.0.0.1:1976").rstrip("/")
CTX = int(os.environ.get("FM_CTX", "4096"))
LOG = os.environ.get("PROXY_LOG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "proxy.jsonl"))
TIMEOUT = float(os.environ.get("FM_TIMEOUT", "600"))

MARKERS = re.compile(r"<start_of_turn>(?:model|assistant)?\s*|<end_of_turn>|<ctrl\d+>|<\|[a-z_]+\|>")
DEC = json.JSONDecoder()

def log(**kw):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f: f.write(json.dumps({"ts": time.time(), **kw}) + "\n")

def _norm_call(obj, names):
    """obj -> (name, args_dict) or None."""
    if not isinstance(obj, dict): return None
    if "function" in obj and isinstance(obj["function"], dict): obj = obj["function"]
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
    if not isinstance(name, str): return None
    if isinstance(args, str):
        try: args = json.loads(args)
        except Exception: args = {"input": args}
    if not isinstance(args, dict): args = {}
    if name not in names:  # tolerate prefixes like default_api: / functions. / ext__
        cands = [n for n in names if name.split(":")[-1].split(".")[-1] == n or n.endswith("__" + name) or name.endswith("__" + n)]
        if len(cands) != 1: return None
        name = cands[0]
    return name, args

def _loose_obj(s):
    """'{count:3,sides:8}' -> dict (best effort)."""
    try: return json.loads(s)
    except Exception: pass
    out = {}
    for k, v in re.findall(r'([A-Za-z_][\w]*)\s*:\s*("(?:[^"\\]|\\.)*"|[^,}]+)', s):
        v = v.strip()
        try: out[k] = json.loads(v)
        except Exception: out[k] = v.strip("'")
    return out

def extract_tool_calls(text, tool_names):
    """Return (calls, leftover_text). calls = [(name, args)]."""
    names = set(tool_names)
    clean = MARKERS.sub("", text or "").strip()
    calls = []
    # A) JSON objects anywhere in the text
    i, spans = 0, []
    while i < len(clean):
        j = clean.find("{", i)
        if j < 0: break
        try:
            obj, end = DEC.raw_decode(clean, j)
        except ValueError:
            i = j + 1; continue
        found = []
        if isinstance(obj, dict):
            lst = obj.get("tool_call", obj.get("tool_calls", obj.get("function_call")))
            if lst is not None:
                for c in (lst if isinstance(lst, list) else [lst]):
                    n = _norm_call(c, names)
                    if n: found.append(n)
            else:
                n = _norm_call(obj, names)
                if n: found.append(n)
        if found:
            calls += found; spans.append((j, end))
        i = end if found else j + 1
    # B) [default_api:fn{...}] / [fn(...)] style
    if not calls:
        for m in re.finditer(r"\[(?:[\w]+:)?([A-Za-z_][\w.\-]*)\s*(\{.*?\})\]", clean, re.S):
            n = _norm_call({"name": m.group(1), "arguments": _loose_obj(m.group(2))}, names)
            if n: calls.append(n); spans.append(m.span())
    left = clean
    for a, b in sorted(spans, reverse=True): left = left[:a] + left[b:]
    left = re.sub(r"</?tool_call>", "", left).strip()
    return calls, left

def read_sse_as_message(body):
    """Accumulate an SSE body (in case upstream streams anyway) into one completion."""
    content, usage, fr, cid = "", None, "stop", None
    for line in body.decode("utf-8", "replace").splitlines():
        if not line.startswith("data:"): continue
        d = line[5:].strip()
        if d == "[DONE]": break
        try: o = json.loads(d)
        except Exception: continue
        if "error" in o: return None, o
        cid = cid or o.get("id")
        for ch in o.get("choices") or []:
            content += (ch.get("delta") or {}).get("content") or ""
            fr = ch.get("finish_reason") or fr
        usage = o.get("usage") or usage
    return {"id": cid or "chatcmpl-proxy", "object": "chat.completion", "created": int(time.time()), "model": "system",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": fr}],
            "usage": usage}, None

def is_ctx_overflow(msg):
    m = (msg or "").lower()
    return "context size" in m or "exceeded the model's context" in m or "exceeds the maximum allowed context" in m

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def _send(self, code, obj, ctype="application/json"):
        b = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def _upstream(self, method, path, body=None):
        req = urllib.request.Request(UP + path, data=body, method=method,
                                     headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()
        except Exception as e:
            return 502, "application/json", json.dumps({"error": {"type": "server_error", "code": "upstream_unreachable", "message": str(e)}}).encode()

    def do_GET(self):
        code, ct, body = self._upstream("GET", self.path)
        if code == 200 and self.path.rstrip("/").endswith("/v1/models"):
            try:
                j = json.loads(body)
                for m in j.get("data", []): m.setdefault("meta", {})["n_ctx"] = CTX
                body = json.dumps(j).encode()
            except Exception: pass
        self._send(code, body, ct or "application/json")

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if not self.path.rstrip("/").endswith("/chat/completions"):
            code, ct, body = self._upstream("POST", self.path, raw); return self._send(code, body, ct)
        try: req = json.loads(raw)
        except Exception: return self._send(400, {"error": {"type": "invalid_request_error", "message": "invalid JSON"}})
        client_stream = bool(req.get("stream", False))
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        req["stream"] = False
        req.pop("stream_options", None)
        tc = req.pop("tool_choice", None)
        if tc == "none": req.pop("tools", None)
        tools = req.get("tools") or []
        for t in tools:
            fn = t.setdefault("function", {})
            if not fn.get("description"): fn["description"] = fn.get("name", "tool")
        if not tools: req.pop("tools", None); req.pop("parallel_tool_calls", None)
        names = [t["function"].get("name") for t in tools]

        code, ct, body = self._upstream("POST", self.path, json.dumps(req).encode())
        err = None
        if code == 200 and "event-stream" in ct:
            resp, err = read_sse_as_message(body)
        elif code == 200:
            try: resp = json.loads(body)
            except Exception: code, err = 502, {"error": {"type": "server_error", "message": "bad upstream JSON"}}
        else:
            try: err = json.loads(body)
            except Exception: err = {"error": {"type": "server_error", "message": body.decode("utf-8", "replace")}}
        if err is not None:
            msg = (err.get("error") or {}).get("message", "") if isinstance(err.get("error"), dict) else str(err)
            if is_ctx_overflow(msg):
                code = 400
                err = {"error": {"type": "invalid_request_error", "code": "context_length_exceeded",
                                 "message": f"context length exceeded ({CTX}-token window): {msg}"}}
            elif code == 200: code = 500
            log(kind="error", status=code, message=msg, n_tools=len(tools))
            return self._send(code, err)

        ch = resp["choices"][0]; msg = ch.get("message") or {}
        raw_content = msg.get("content") or ""
        parsed = []
        if tools and not msg.get("tool_calls"):
            parsed, left = extract_tool_calls(raw_content, names)
            msg["content"] = left if left else (None if parsed else "")
            if parsed:
                msg["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                                      "function": {"name": n, "arguments": json.dumps(a)}} for n, a in parsed]
                ch["finish_reason"] = "tool_calls"
        log(kind="completion", n_tools=len(tools), raw_content=raw_content[:500], parsed=parsed,
            usage=resp.get("usage"), client_stream=client_stream)
        print(f"[proxy] tools={len(tools)} parsed={parsed} usage={resp.get('usage')}", file=sys.stderr, flush=True)

        if not client_stream:
            return self._send(200, resp)
        # synthesize SSE
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close"); self.end_headers()
        base = {"id": resp.get("id", "chatcmpl-proxy"), "object": "chat.completion.chunk",
                "created": resp.get("created", int(time.time())), "model": resp.get("model", "system")}
        def ev(o): self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")
        ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        if msg.get("content"):
            ev({**base, "choices": [{"index": 0, "delta": {"content": msg["content"]}, "finish_reason": None}]})
        for i, c in enumerate(msg.get("tool_calls") or []):
            ev({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, **c}]}, "finish_reason": None}]})
        ev({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": ch.get("finish_reason", "stop")}]})
        if include_usage and resp.get("usage"):
            ev({**base, "choices": [], "usage": resp["usage"]})
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        self.close_connection = True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        names = ["get_weather", "shell", "roll_dice"]
        cases = ['<start_of_turn>model\n{"tool_call": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]}<ctrl46>',
                 '<tool_call>{"name": "shell", "arguments": {"command": "ls"}}</tool_call>',
                 '[default_api:roll_dice{count:3,sides:8}]',
                 'Sure! {"name": "nope", "arguments": {}}', 'The weather is sunny.']
        for c in cases: print(repr(c[:50]), "->", extract_tool_calls(c, names))
        sys.exit(0)
    print(f"fm_tool_proxy on 127.0.0.1:{PORT} -> {UP} (n_ctx={CTX})", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
