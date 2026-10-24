#!/usr/bin/env python3
"""Mock of Apple's `fm serve` (macOS 27, fm 2.0.68.x) for exercising Goose on Linux.

NOT the real thing. Behaviours are tagged:
  [SRC]  documented by a third-party live audit (URLs in README.md)
  [ASM]  assumed / invented for the mock (no source found)

Modes (FM_MOCK_MODE):
  regression (default)  tools present + last msg from user -> tool call JSON leaked into
                        message.content with template markers, NO tool_calls, finish_reason
                        "stop"  [SRC: fm-proxy AGENTS.md "Tool-call replies leak..."]
  chat                  never emits tool-call text; plain answer only
  invent                with tools present, answers as if it had called them (hallucinated
                        result)  [SRC: spacecorps RESEARCH.md §2]
Strict decoding (FM_MOCK_STRICT=1, the default; 0 turns it off): rejects request shapes that
real fm serve's Swift Codable decoder rejects, with its 400 "...: The data couldn't be read
because it is missing." See validate().
Env: FM_MOCK_PORT (1976), FM_MOCK_CTX (4096), FM_MOCK_LOG (runs/mock_requests.jsonl), FM_MOCK_STRICT (1)
"""
import json, os, sys, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("FM_MOCK_PORT", "1976"))
CTX = int(os.environ.get("FM_MOCK_CTX", "4096"))
MODE = os.environ.get("FM_MOCK_MODE", "regression")
LOG = os.environ.get("FM_MOCK_LOG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "mock_requests.jsonl"))
STRICT = os.environ.get("FM_MOCK_STRICT", "1") != "0"
MISSING = "The data couldn\u2019t be read because it is missing."
BADFMT = "The data couldn\u2019t be read because it isn\u2019t in the correct format."

def _schema_err(s, path):
    """First problem in a tool property schema, as (path, reason), or None.
    [SRC fm-proxy AGENTS.md / fm-proxy.js header, 27.0 RC]: $ref/$defs and anyOf/allOf/oneOf/
    if/then/else/not/patternProperties are not understood; `type` arrays are rejected; a titled
    string needs a non-empty enum; the root `required` must be present.
    [SRC future-architect 2026-06-24]: objects with properties need `required`.
    [ASM, Swift Codable]: a property schema without `type` is keyNotFound("type")."""
    if not isinstance(s, dict): return path, "schema is not an object"
    for k in ("$ref", "$defs", "definitions", "anyOf", "oneOf", "allOf", "if", "then", "else", "not", "patternProperties"):
        if k in s: return path, f"unsupported key {k}"
    t = s.get("type")
    if t is None: return path, "missing type"
    if not isinstance(t, str): return path, "type must be a string"
    if t == "string" and s.get("title") and not s.get("enum"): return path, "titled string without enum"
    if t == "object":
        if not isinstance(s.get("properties"), dict): return path, "object without properties"
        if s["properties"] and not isinstance(s.get("required"), list): return path, "object without required"
        for k, v in s["properties"].items():
            e = _schema_err(v, f"{path}.properties.{k}")
            if e: return e
    if t == "array":
        if not isinstance(s.get("items"), dict): return path, "array without items"
        return _schema_err(s["items"], path + ".items")
    return None

def validate(req):
    """None if fm serve would decode `req`, else (status, message, field)."""
    msgs = req.get("messages")
    if not isinstance(msgs, list) or not msgs: return 400, "Invalid request: " + MISSING, "messages"
    ids = set()
    for i, m in enumerate(msgs):
        role = m.get("role") if isinstance(m, dict) else None
        if role == "developer": return 400, "Invalid JSON: " + BADFMT, f"messages[{i}].role"   # [SRC fm-proxy.js]
        if role not in ("system", "user", "assistant", "tool"): return 400, "Invalid request: " + BADFMT, f"messages[{i}].role"
        c = m.get("content")
        if role != "assistant" and c is None: return 400, "Invalid request: " + MISSING, f"messages[{i}].content"   # [ASM]
        if isinstance(c, list):
            for j, part in enumerate(c):
                if not isinstance(part, dict) or part.get("type") not in ("text", "image_url"):
                    return 400, "Invalid request: " + BADFMT, f"messages[{i}].content[{j}]"   # [ASM]
                if part["type"] == "text" and not isinstance(part.get("text"), str):
                    return 400, "Invalid request: " + MISSING, f"messages[{i}].content[{j}].text"
        for j, t in enumerate(m.get("tool_calls") or []):
            fn = t.get("function") or {}
            if not t.get("id") or not isinstance(fn.get("name"), str) or not isinstance(fn.get("arguments"), str):
                return 400, "Invalid request: " + MISSING, f"messages[{i}].tool_calls[{j}]"   # [ASM]
            ids.add(t["id"])
        if role == "tool":
            tid = m.get("tool_call_id")
            if not tid: return 400, "Invalid request: " + MISSING, f"messages[{i}].tool_call_id"
            if tid not in ids: return 400, f"tool message references unknown tool_call_id: '{tid}'", f"messages[{i}].tool_call_id"  # [SRC]
    for i, t in enumerate(req.get("tools") or []):
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str): return 400, "Invalid tool definition: " + MISSING, f"tools[{i}].function.name"
        if not isinstance(fn.get("description"), str): return 400, "Invalid JSON: " + MISSING, f"tools[{i}].function.description"  # [SRC]
        ps = fn.get("parameters")
        if not isinstance(ps, dict) or ps.get("type") != "object" or not isinstance(ps.get("required"), list):
            return 400, "Invalid request: " + MISSING, f"tools[{fn['name']}].parameters"   # [SRC root required]
        e = _schema_err(ps, f"tools[{fn['name']}].parameters")
        if e: return 400, "Invalid request: " + MISSING, f"{e[0]} ({e[1]})"
    return None

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def ntok(s): return len(_enc.encode(s, disallowed_special=()))
    TOKENIZER = "tiktoken/cl100k_base"
except Exception:
    def ntok(s): return (len(s) + 3) // 4
    TOKENIZER = "chars/4"

def text_of(content):
    if content is None: return ""
    if isinstance(content, str): return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))

def measure(req):
    msgs = req.get("messages", [])
    sys_txt = "".join(text_of(m.get("content")) for m in msgs if m.get("role") in ("system", "developer"))
    other = "".join(text_of(m.get("content")) + json.dumps(m.get("tool_calls") or "") for m in msgs if m.get("role") not in ("system", "developer"))
    tools_json = json.dumps(req.get("tools") or [])
    t = {"system_tokens": ntok(sys_txt), "messages_tokens": ntok(other),
         "tools_tokens": ntok(tools_json) if req.get("tools") else 0,
         "n_tools": len(req.get("tools") or []), "n_messages": len(msgs),
         "system_chars": len(sys_txt), "tools_chars": len(tools_json) if req.get("tools") else 0}
    # [ASM] framing overhead ~ per-message; fm reports 57 prompt_tokens for "hello world" [SRC]
    t["prompt_tokens_est"] = t["system_tokens"] + t["messages_tokens"] + t["tools_tokens"] + 50 + 4 * len(msgs)
    t["tokenizer"] = TOKENIZER
    return t

def pick_tool_call(tools, user_text):
    """Deterministic fake 'model decision': prefer a tool taking a `command` string."""
    fns = [t.get("function", {}) for t in tools]
    for f in fns:
        props = (f.get("parameters") or {}).get("properties", {})
        if "command" in props:
            return f["name"], {"command": "echo hello-from-fm && ls | head -3"}
    for f in fns:
        if not (f.get("parameters") or {}).get("required"):
            return f["name"], {}
    f = fns[0]
    req = (f.get("parameters") or {}).get("required", [])
    return f["name"], {k: "x" for k in req}

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def _err(self, code, msg, typ=None):
        # [SRC] message strings; envelope {"error":{type,code,message}} as in fm-proxy test fixtures
        self._json(code, {"error": {"type": typ or ("server_error" if code >= 500 else "invalid_request_error"),
                                    "code": str(code), "message": msg}})

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            # [ASM] exact JSON shape; [SRC] `system` is the only model id
            return self._json(200, {"object": "list", "data": [{"id": "system", "object": "model", "owned_by": "apple"}]})
        if self.path == "/health":
            return self._json(200, {"status": "ok"})  # [ASM] body shape
        self._err(404, "Not found")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        try: req = json.loads(raw)
        except Exception: return self._err(400, "Invalid JSON")
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._err(404, "Not found")
        if self.headers.get("Origin") or self.headers.get("Referer"):
            return self._err(403, "Cross-site requests are not allowed")  # [SRC]
        m = measure(req)
        entry = {"ts": time.time(), "mode": MODE, "model": req.get("model"), "stream": req.get("stream"),
                 "tool_choice": req.get("tool_choice"), "auth": bool(self.headers.get("Authorization")),
                 "keys": sorted(req.keys()), **m, "tool_names": [t.get("function", {}).get("name") for t in req.get("tools") or []]}
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a") as f: f.write(json.dumps(entry) + "\n")
        with open(LOG.replace(".jsonl", "_last_body.json"), "w") as f: json.dump(req, f, indent=1)
        bd = os.path.join(os.path.dirname(LOG), "bodies"); os.makedirs(bd, exist_ok=True)
        with open(os.path.join(bd, f"{time.time():.3f}.json"), "w") as f: json.dump(req, f, indent=1)
        print(f"[mock] {m['prompt_tokens_est']} est tok (sys {m['system_tokens']}, tools {m['tools_tokens']} x{m['n_tools']}, msgs {m['messages_tokens']}) stream={req.get('stream')}", file=sys.stderr, flush=True)

        model = req.get("model") or "system"
        if model != "system":
            return self._err(400, f"Unknown model '{model}'. Available models: system.")  # [SRC]
        if (req.get("n") or 1) > 1:
            return self._err(400, f"n={req['n']} is not supported. Only a single completion per request is implemented.")  # [SRC]
        if "stop" in req and req["stop"] is not None:
            return self._err(400, "stop is not supported")  # [SRC: 400] [ASM: message text]
        for t in req.get("tools") or []:
            if "description" not in t.get("function", {}):
                return self._err(400, "Invalid JSON: " + MISSING)  # [SRC: fm-proxy AGENTS.md]
        bad = validate(req) if STRICT else None
        if bad:
            print(f"[mock] 400 at {bad[2]}: {bad[1]}", file=sys.stderr, flush=True)
            with open(LOG, "a") as f: f.write(json.dumps({"ts": time.time(), "rejected": bad[2], "message": bad[1]}) + "\n")
            return self._err(bad[0], bad[1])
        tc = req.get("tool_choice")
        if tc == "required" or isinstance(tc, dict):
            return self._err(500, "An unsupported generation guide was used.")  # [SRC]
        if m["prompt_tokens_est"] > CTX:
            return self._err(500, "The session's transcript exceeded the model's context size.")  # [SRC]

        msgs = req.get("messages", [])
        last = msgs[-1] if msgs else {}
        tools = req.get("tools") or []
        user_text = text_of(last.get("content"))
        if last.get("role") == "tool":
            content = "The tool returned: " + text_of(last.get("content"))[:200].strip() + "\nDone."
        elif tools and MODE == "regression":
            name, args = pick_tool_call(tools, user_text)
            # [SRC] shape seen live: <start_of_turn>model\n{"tool_call": [{"name": ..., "arguments": {...}}]}
            content = "<start_of_turn>model\n" + json.dumps({"tool_call": [{"name": name, "arguments": args}]}) + "<ctrl46>"
        elif tools and MODE == "invent":
            content = "I ran the command and everything looks fine: 3 files found."
        else:
            content = "PONG from mock fm serve. You said: " + user_text[:80]

        usage = {"prompt_tokens": m["prompt_tokens_est"], "completion_tokens": ntok(content),
                 "total_tokens": m["prompt_tokens_est"] + ntok(content)}
        cid = "chatcmpl-" + uuid.uuid4().hex[:12]
        stream = req.get("stream", True)  # [SRC] omitted stream => SSE
        if stream is False:
            return self._json(200, {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": "system",
                                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                                 "finish_reason": "stop"}], "usage": usage})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        def ev(o): self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n"); self.wfile.flush()
        base = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": "system"}
        ev({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        for i in range(0, len(content), 16):
            ev({**base, "choices": [{"index": 0, "delta": {"content": content[i:i+16]}, "finish_reason": None}]})
        ev({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        if (req.get("stream_options") or {}).get("include_usage"):  # [SRC] usage only with include_usage
            ev({**base, "choices": [], "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        self.close_connection = True

if __name__ == "__main__":
    print(f"mock fm serve on 127.0.0.1:{PORT} mode={MODE} ctx={CTX} strict={STRICT} tokenizer={TOKENIZER}", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
