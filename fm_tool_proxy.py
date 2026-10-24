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
 7. Request normalisation (normalize_request): fm serve decodes the body with strict Swift
    Codable types and answers 400 "...: The data couldn't be read because it is missing."
    for shapes OpenAI accepts. Tool `parameters` are rewritten to the flat subset fm serve
    decodes: `$ref`/`$defs` inlined, anyOf/oneOf/allOf collapsed, `type` arrays collapsed,
    every property gets a `type`, every object gets `properties` + `required`, and
    decorative/validation keys (title, default, $schema, format, minimum, ...) are dropped.
    Goose's developer `read_image` tool sends `crop: {"$ref": "#/$defs/CropParams"}` (no
    `type`), which is what broke the Goose smoke test on a real Mac. Messages: `developer`
    -> `system`, null/missing `content` -> "", text-part arrays -> one string, tool_call
    arguments -> JSON strings; `stop`, `reasoning_effort`, `strict` are not forwarded.
 8. Any upstream 4xx is logged with a field summary, and the exact forwarded body is
    saved to runs/last_upstream_400.json, so the next failure says which field it was.
Env: PROXY_PORT=1977 FM_UPSTREAM=http://127.0.0.1:1976 FM_CTX=4096 PROXY_LOG=runs/proxy.jsonl
"""
import json, os, re, sys, time, uuid, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PROXY_PORT", "1977"))
UP = os.environ.get("FM_UPSTREAM", "http://127.0.0.1:1976").rstrip("/")
CTX = int(os.environ.get("FM_CTX", "4096"))
HERE = os.path.dirname(os.path.abspath(__file__))
# Runtime logs go to runs/ (gitignored) so a running stack never dirties the git checkout.
LOG = os.environ.get("PROXY_LOG", os.path.join(HERE, "runs", "proxy.jsonl"))
BAD_BODY = os.path.join(os.path.dirname(LOG), "last_upstream_400.json")
PROXY_VERSION = "2026-10-24.1"   # reported by GET /proxy/version; setup restarts a stale proxy
TIMEOUT = float(os.environ.get("FM_TIMEOUT", "600"))

MARKERS = re.compile(r"<start_of_turn>(?:model|assistant)?\s*|<end_of_turn>|<ctrl\d+>|<\|[a-z_]+\|>")
DEC = json.JSONDecoder()

def log(**kw):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    try:   # keep an always-on stack's log bounded: one 5 MB generation + the current file
        if os.path.getsize(LOG) > 5_000_000: os.replace(LOG, LOG + ".1")
    except OSError: pass
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

# ---- request normalisation (fix 7) -------------------------------------------------
# Keys fm serve's tool-schema decoder understands. Everything else is dropped from
# property schemas: decorative keys cost tokens, and unknown/validation keys
# (`format: uint32`, `minimum`, `default: null`, `title` on a string) either 400 or
# are ignored [SRC fm-proxy AGENTS.md; future-architect 2026-06-24].
_PRIM_KEYS = ("type", "description", "enum")
_COMPOSITE = ("anyOf", "oneOf", "allOf")

def _inline_refs(node, defs, active=()):
    """Replace {"$ref":"#/$defs/X"} with a copy of X (siblings win). Cycles -> {"type":"string"}."""
    if isinstance(node, list): return [_inline_refs(n, defs, active) for n in node]
    if not isinstance(node, dict): return node
    ref = node.get("$ref")
    if isinstance(ref, str):
        name = ref.split("/")[-1]
        target = defs.get(name) if (ref.startswith("#/$defs/") or ref.startswith("#/definitions/")) else None
        if not isinstance(target, dict) or name in active:
            base = {"type": "string"}
        else:
            base = _inline_refs(target, defs, active + (name,))
        merged = dict(base)
        for k, v in node.items():
            if k != "$ref": merged[k] = _inline_refs(v, defs, active)
        return merged
    return {k: _inline_refs(v, defs, active) for k, v in node.items() if k not in ("$defs", "definitions")}

def _simplify(s):
    """One property schema -> fm-serve subset. Always returns a dict with a `type`."""
    if not isinstance(s, dict): return {"type": "string"}
    for key in _COMPOSITE:
        subs = [x for x in (s.get(key) or []) if isinstance(x, dict)]
        if subs:
            if key == "allOf":
                merged = {}
                for x in subs: merged.update(x)
            else:   # first branch that isn't plain null
                merged = dict(next((x for x in subs if x.get("type") not in (None, "null")), subs[0]))
            for k, v in s.items():
                if k != key and (key == "allOf" or k not in merged): merged[k] = v
            merged.pop(key, None)
            return _simplify(merged)
    t = s.get("type")
    if isinstance(t, list): t = next((x for x in t if x != "null"), "string")
    if t in (None, "null"):
        t = "object" if "properties" in s else "array" if "items" in s else \
            ("string" if not s.get("enum") else {str: "string", int: "integer", float: "number", bool: "boolean"}.get(type(s["enum"][0]), "string"))
    if t == "object":
        return _obj(s)
    out = {"type": t}
    if isinstance(s.get("description"), str) and s["description"]: out["description"] = s["description"]
    if t == "array":
        out["items"] = _simplify(s.get("items") if isinstance(s.get("items"), dict) else {"type": "string"})
    elif isinstance(s.get("enum"), list) and s["enum"] and all(v is not None for v in s["enum"]):
        out["enum"] = s["enum"]
    return out

def _obj(s):
    props = s.get("properties") if isinstance(s.get("properties"), dict) else {}
    out = {"type": "object", "properties": {k: _simplify(v) for k, v in props.items()}}
    req = s.get("required") if isinstance(s.get("required"), list) else []
    out["required"] = [r for r in req if isinstance(r, str) and r in out["properties"]]
    if isinstance(s.get("description"), str) and s["description"]: out["description"] = s["description"]
    return out

def normalize_tool_schema(params):
    """OpenAI/Goose `function.parameters` -> {"type":"object","properties":{..},"required":[..]}."""
    if not isinstance(params, dict): params = {}
    defs = {}
    for k in ("$defs", "definitions"):
        if isinstance(params.get(k), dict): defs.update(params[k])
    p = _inline_refs(params, defs)
    out = _obj(p)
    out.pop("description", None)   # root description lives on the function
    return out

def _text_of(content):
    if content is None: return ""
    if isinstance(content, str): return content
    if isinstance(content, list):
        return "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content
                         if not isinstance(x, dict) or x.get("type") in (None, "text", "input_text", "output_text"))
    return json.dumps(content)

def normalize_request(req):
    """Make an OpenAI chat request decodable by fm serve. Returns (req, changes:list[str])."""
    ch = []
    # `stop` 400s upstream; the handler applies it to the reply instead (req["_stop"]).
    if "stop" in req:
        s = req.pop("stop"); ch.append("stop applied by proxy")
        req["_stop"] = [s] if isinstance(s, str) else [x for x in (s or []) if isinstance(x, str) and x]
    for k in ("reasoning_effort", "stream_options", "store", "metadata", "user"):
        if k in req: req.pop(k); ch.append("dropped " + k)
    msgs = req.get("messages")
    if not isinstance(msgs, list): req["messages"] = msgs = []; ch.append("messages missing")
    for i, m in enumerate(msgs):
        if not isinstance(m, dict): msgs[i] = m = {"role": "user", "content": str(m)}; ch.append(f"messages[{i}] not an object")
        if m.get("role") == "developer": m["role"] = "system"; ch.append(f"messages[{i}].role developer->system")
        if m.get("role") not in ("system", "user", "assistant", "tool"):
            ch.append(f"messages[{i}].role {m.get('role')!r}->user"); m["role"] = "user"
        c = m.get("content")
        if c is None:
            m["content"] = ""; ch.append(f"messages[{i}].content null->\"\"")
        elif isinstance(c, list):
            has_image = any(isinstance(x, dict) and x.get("type") == "image_url" for x in c)
            if not (has_image and m["role"] == "user"):   # fm serve takes image_url parts on user turns
                m["content"] = _text_of(c); ch.append(f"messages[{i}].content array->string")
        elif not isinstance(c, str):
            m["content"] = json.dumps(c); ch.append(f"messages[{i}].content {type(c).__name__}->string")
        if m["role"] == "assistant":
            tcs = m.get("tool_calls")
            if tcs in (None, []): m.pop("tool_calls", None)
            elif isinstance(tcs, list):
                for j, t in enumerate(tcs):
                    t.setdefault("type", "function")
                    if not t.get("id"): t["id"] = "call_" + uuid.uuid4().hex[:24]; ch.append(f"messages[{i}].tool_calls[{j}].id filled")
                    fn = t.setdefault("function", {})
                    fn.setdefault("name", "")
                    a = fn.get("arguments")
                    if not isinstance(a, str):
                        fn["arguments"] = json.dumps(a if a is not None else {}); ch.append(f"messages[{i}].tool_calls[{j}].arguments->string")
        else:
            m.pop("tool_calls", None)
        if m["role"] == "tool" and not m.get("tool_call_id"):
            ch.append(f"messages[{i}] tool message without tool_call_id")
        if "refusal" in m and m["refusal"] is None: m.pop("refusal")
    tools = req.get("tools")
    if tools is not None:
        good = []
        for i, t in enumerate(tools if isinstance(tools, list) else []):
            if not isinstance(t, dict): ch.append(f"tools[{i}] dropped (not an object)"); continue
            fn = t.get("function") if isinstance(t.get("function"), dict) else {}
            if not fn.get("name"): ch.append(f"tools[{i}] dropped (no name)"); continue
            before = json.dumps(fn.get("parameters"), sort_keys=True)
            nfn = {"name": fn["name"],
                   "description": fn["description"] if isinstance(fn.get("description"), str) and fn["description"] else fn["name"],
                   "parameters": normalize_tool_schema(fn.get("parameters"))}
            if json.dumps(nfn["parameters"], sort_keys=True) != before: ch.append(f"tools[{fn['name']}].parameters normalised")
            if "strict" in fn: ch.append(f"tools[{fn['name']}].strict dropped")
            good.append({"type": "function", "function": nfn})
        if good: req["tools"] = good
        else: req.pop("tools", None); req.pop("parallel_tool_calls", None)
    return req, ch

def describe_body(req):
    """Compact shape summary for logs (no message text)."""
    return {"keys": sorted(req),
            "messages": [{k: (v if k == "role" else type(v).__name__) for k, v in m.items()} if isinstance(m, dict) else type(m).__name__
                         for m in req.get("messages") or []],
            "tools": {(t.get("function") or {}).get("name"): sorted(((t.get("function") or {}).get("parameters") or {}).get("properties", {}))
                      for t in req.get("tools") or [] if isinstance(t, dict)}}

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
        if self.path.rstrip("/") == "/proxy/version":
            return self._send(200, {"proxy": "fm_tool_proxy", "version": PROXY_VERSION, "upstream": UP, "n_ctx": CTX})
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
        req, changes = normalize_request(req)
        stops = req.pop("_stop", None) or []
        tools = req.get("tools") or []
        names = [t["function"]["name"] for t in tools]
        if changes:
            log(kind="normalised", changes=changes[:60])

        fwd = json.dumps(req).encode()
        code, ct, body = self._upstream("POST", self.path, fwd)
        if 400 <= code < 500:   # fix 8: keep what fm serve rejected so the field can be found
            try:
                os.makedirs(os.path.dirname(BAD_BODY), exist_ok=True)
                with open(BAD_BODY, "wb") as f: f.write(fwd)
            except Exception: pass
            log(kind="upstream_4xx", status=code, upstream=body.decode("utf-8", "replace")[:500],
                shape=describe_body(req), saved=BAD_BODY)
            print(f"[proxy] fm serve {code}: {body.decode('utf-8', 'replace')[:300]} (body saved to {BAD_BODY})",
                  file=sys.stderr, flush=True)
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
        if stops:
            cut = min([i for i in (raw_content.find(s) for s in stops) if i >= 0], default=-1)
            if cut >= 0: raw_content = raw_content[:cut]; msg["content"] = raw_content
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
        # normalisation: Goose developer read_image ($ref without type), nullable unions, nulls
        req = {"model": "system", "stop": ["X"], "messages": [
                   {"role": "developer", "content": "sys"},
                   {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                   {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "shell", "arguments": {"command": "ls"}}}]},
                   {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "ok"}]}],
               "tools": [{"type": "function", "function": {"name": "read_image", "description": "d", "strict": False,
                   "parameters": {"$schema": "x", "title": "P", "type": "object", "required": ["source"],
                       "properties": {"source": {"type": "string"},
                                      "crop": {"default": None, "$ref": "#/$defs/Crop"},
                                      "n": {"type": ["integer", "null"], "format": "uint32", "minimum": 0},
                                      "u": {"anyOf": [{"type": "null"}, {"type": "string"}]}},
                       "$defs": {"Crop": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}}}},
                   {"type": "function", "function": {"name": "noargs"}}]}
        out, ch = normalize_request(req)
        s = json.dumps(out)
        assert "$ref" not in s and "$defs" not in s and "null" not in s and "strict" not in s and "format" not in s, s
        ps = out["tools"][0]["function"]["parameters"]
        assert ps["properties"]["crop"] == {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}, ps
        assert ps["properties"]["n"] == {"type": "integer"} and ps["properties"]["u"] == {"type": "string"}, ps
        assert out["tools"][1]["function"] == {"name": "noargs", "description": "noargs",
                                               "parameters": {"type": "object", "properties": {}, "required": []}}
        m = out["messages"]
        assert m[0]["role"] == "system" and m[1]["content"] == "hi" and m[2]["content"] == "" and m[3]["content"] == "ok"
        assert m[2]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}' and "stop" not in out and out["_stop"] == ["X"]
        print("normalise: ok (%d changes)" % len(ch))
        sys.exit(0)
    print(f"fm_tool_proxy on 127.0.0.1:{PORT} -> {UP} (n_ctx={CTX})", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
