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
 9. Goose context (prepare_context): Goose 1.56 only sends its `<turn-context>` block (cwd, time)
    when the context limit is >= 32k, but its system prompt always describes that block. At
    4096 the model never learns the working directory and echoes `<turn_context>` back. The
    proxy removes that section and Goose boilerplate from the system prompt, moves any real
    `<turn-context>` facts into one plain system line, and otherwise reads the working directory
    of the Goose process on the other end of the loopback connection (lsof on macOS, /proc on
    Linux; skipped for `goose acp`, whose sessions have their own cwd). Disable: FM_PEER_CWD=0.
    It then drops the oldest turns (never the latest user turn or what follows it) and trims
    long tool output so the request fits FM_CTX, and retries once, tighter, on overflow.
10. Reply cleanup (clean_reply): strips echoed `<turn_context>`/`<turn-context>` wrappers and
    their time/cwd lines, drops invented `<stdout>`/`<output>` sections, and turns text-form calls
    (`<tool_call id="shell:1">cmd`, JSON blobs, a lone ```bash block when `shell` is offered and
    the user spoke last) into real `tool_calls`. Text the model wrote after a call (a guessed
    result) is dropped. Plain prose is never turned into a call.
Env: PROXY_PORT=1977 FM_UPSTREAM=http://127.0.0.1:1976 FM_CTX=4096 PROXY_LOG=runs/proxy.jsonl
     FM_RESERVE=600 (tokens kept free for the reply) FM_PEER_CWD=1
"""
import json, os, re, subprocess, sys, time, uuid, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PROXY_PORT", "1977"))
UP = os.environ.get("FM_UPSTREAM", "http://127.0.0.1:1976").rstrip("/")
CTX = int(os.environ.get("FM_CTX", "4096"))
HERE = os.path.dirname(os.path.abspath(__file__))
# Runtime logs go to runs/ (gitignored) so a running stack never dirties the git checkout.
LOG = os.environ.get("PROXY_LOG", os.path.join(HERE, "runs", "proxy.jsonl"))
BAD_BODY = os.path.join(os.path.dirname(LOG), "last_upstream_400.json")
PROXY_VERSION = "2026-10-24.3"   # reported by GET /proxy/version; setup restarts a stale proxy
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

# ---- Goose context (fix 9) ---------------------------------------------------------
RESERVE = int(os.environ.get("FM_RESERVE", "600"))
PEER_CWD = os.environ.get("FM_PEER_CWD", "1") != "0"
TC_BLOCK = re.compile(r"<turn[-_]context>(.*?)</turn[-_]context>", re.S)
_BOILER = [
    re.compile(r"\n?# Turn Context\n.*?(?=\n# |\Z)", re.S),
    re.compile(r"goose is being developed as an open-source software project\.\n?"),
    re.compile(r"Extensions provide additional tools and context from different data sources and applications\.\n"
               r"You can dynamically enable or disable extensions as needed to help complete tasks\.\n?"),
    re.compile(r"Because you dynamically load extensions, your conversation history may refer\s+to interactions with "
               r"extensions that are not currently active\. The currently\s+active extensions are below\. Each of these "
               r"extensions provides tools that are\s+in your tool specification\.\n?"),
    re.compile(r"Make sure to use the tools \*efficiently\*.*?cost the user money\.\n?", re.S),
]

def _tag(name, text):
    m = re.search(r"<%s>(.*?)</%s>" % (name, name), text, re.S)
    return m.group(1).strip() if m else None

_peer_cache = {}
def peer_cwd(port):
    """Working directory of the local Goose process that owns the client end of this connection."""
    if not PEER_CWD or not port: return None
    if port in _peer_cache: return _peer_cache[port]
    cwd = None
    try:
        me = os.getpid()
        if sys.platform == "darwin":
            r = subprocess.run(["/usr/sbin/lsof", "-nP", "-t", "-iTCP:%d" % port, "-sTCP:ESTABLISHED"],
                               capture_output=True, text=True, timeout=3)
            for pid in [int(x) for x in r.stdout.split() if x.isdigit() and int(x) != me]:
                args = subprocess.run(["/bin/ps", "-o", "args=", "-p", str(pid)], capture_output=True, text=True, timeout=3).stdout
                if "goose" not in args.split(" ")[0] or re.search(r"\sacp(\s|$)", args): continue
                r2 = subprocess.run(["/usr/sbin/lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], capture_output=True, text=True, timeout=3)
                cwd = next((l[1:] for l in r2.stdout.splitlines() if l.startswith("n")), None)
                if cwd: break
        elif os.path.isdir("/proc"):
            inode = None
            for f in ("/proc/net/tcp", "/proc/net/tcp6"):
                try:
                    for line in open(f).read().splitlines()[1:]:
                        c = line.split()
                        if int(c[1].rsplit(":", 1)[1], 16) == port and c[3] == "01": inode = c[9]; break
                except OSError: pass
                if inode: break
            if inode:
                tgt = "socket:[%s]" % inode
                for pid in filter(str.isdigit, os.listdir("/proc")):
                    if int(pid) == me: continue
                    try:
                        if not any(os.readlink("/proc/%s/fd/%s" % (pid, fd)) == tgt for fd in os.listdir("/proc/%s/fd" % pid)): continue
                        argv = open("/proc/%s/cmdline" % pid, "rb").read().split(b"\0")
                        if b"goose" not in os.path.basename(argv[0]) or b"acp" in argv[1:3]: break
                        cwd = os.readlink("/proc/%s/cwd" % pid); break
                    except OSError: continue
    except Exception: cwd = None
    if len(_peer_cache) > 256: _peer_cache.clear()
    _peer_cache[port] = cwd
    return cwd

def est_tokens(req):
    """Conservative token estimate for fm's tokenizer (no exact count without forking `fm`)."""
    n = 60
    for m in req.get("messages") or []:
        n += 6 + len(_text_of(m.get("content"))) / 3.2 + len(json.dumps(m.get("tool_calls") or "")) / 2.8
    if req.get("tools"): n += len(json.dumps(req["tools"])) / 2.8
    return int(n)

def _trim_text(s, keep):
    if len(s) <= keep: return s
    head = int(keep * 0.7); tail = keep - head
    return s[:head] + "\n…[%d characters trimmed by fm_tool_proxy]…\n" % (len(s) - keep) + s[-tail:]

def prepare_context(req, cwd=None, budget=None):
    """Fix 9. Returns list of changes. Mutates req."""
    ch = []
    msgs = req.get("messages") or []
    facts = {}
    for m in msgs:   # real turn-context blocks (Goose sends them when its context limit is >= 32k)
        if m.get("role") == "user" and isinstance(m.get("content"), str) and TC_BLOCK.search(m["content"]):
            for blk in TC_BLOCK.findall(m["content"]):
                for k in ("working-directory", "current-time"):
                    v = _tag(k, blk)
                    if v: facts[k] = v
            m["content"] = TC_BLOCK.sub("", m["content"]).strip()
            ch.append("turn-context block moved to system line")
    before = len(msgs)
    msgs[:] = [m for m in msgs if not (m.get("role") == "user" and m.get("content") == "")]
    if len(msgs) != before: ch.append("dropped %d empty turn-context messages" % (before - len(msgs)))
    for m in msgs:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            s = m["content"]
            for rx in _BOILER: s = rx.sub("", s)
            s = re.sub(r"\n{3,}", "\n\n", s).strip()
            if s != m["content"]: m["content"] = s; ch.append("system prompt compacted")
    wd = facts.get("working-directory") or cwd
    lines = []
    if wd: lines.append("Working directory: %s (shell commands run here; use relative paths, or absolute paths the user gives)." % wd)
    lines.append("Current time: %s." % (facts.get("current-time") or time.strftime("%Y-%m-%d %H:%M %Z")))
    if req.get("tools"):
        lines.append("To use a tool, call it. Never write a tool's output yourself; wait for the real result.")
    note = {"role": "system", "content": "\n".join(lines)}
    first_sys = next((i for i, m in enumerate(msgs) if m.get("role") == "system"), None)
    if first_sys is None: msgs.insert(0, note)
    else: msgs[first_sys]["content"] = note["content"] + "\n\n" + msgs[first_sys]["content"]
    ch.append("context line: cwd=%s" % ("turn-context" if facts.get("working-directory") else "peer" if cwd else "unknown"))
    # fit the window
    budget = budget or (CTX - RESERVE)
    if est_tokens(req) > budget:
        last_user = max([i for i, m in enumerate(msgs) if m.get("role") == "user"] or [0])
        sys_msgs = [m for m in msgs[:last_user] if m.get("role") == "system"]
        hist = [m for m in msgs[:last_user] if m.get("role") != "system"]
        tail = msgs[last_user:]
        dropped = 0
        while hist and est_tokens({**req, "messages": sys_msgs + hist + tail}) > budget:
            nxt = next((i for i, m in enumerate(hist[1:], 1) if m.get("role") == "user"), len(hist))
            dropped += nxt; hist = hist[nxt:]
        if dropped:
            ch.append("dropped %d old messages to fit %d tokens" % (dropped, budget))
            sys_msgs[0:1] = [{**sys_msgs[0], "content": sys_msgs[0]["content"] + "\n(Earlier conversation omitted to fit the context window.)"}] if sys_msgs else []
        msgs[:] = sys_msgs + hist + tail
        for keep in (1500, 700, 300):
            if est_tokens(req) <= budget: break
            for m in msgs:
                if m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > keep:
                    m["content"] = _trim_text(m["content"], keep); ch.append("tool output trimmed to %d chars" % keep)
    req["messages"] = msgs
    return ch

# ---- reply cleanup (fix 10) ----------------------------------------------------------
_INFO = re.compile(r"<(current-time|working-directory|turn-budget)>.*?</\1>\s*", re.S)
_WRAP = re.compile(r"</?turn[-_]context[^>]*>\s*")
_FAKE_OUT = re.compile(r"<(stdout|stderr|output|tool_output|tool_result|tool_response|result)>.*?(?:</\1>|\Z)\s*", re.S)
_TAGGED = re.compile(r"<tool_call(\s[^>]*)?>(.*?)(?:</tool_call>|(?=<tool_call[\s>])|\Z)", re.S)
_FENCE = re.compile(r"```(bash|sh|shell|zsh|console|terminal)?[ \t]*\n(.*?)```", re.S)

def _clean_nul(s):
    return re.sub(r"\n{3,}", "\n\n", s.replace("\x00", "")).strip()

def _single_param(tool):
    ps = ((tool or {}).get("function") or {}).get("parameters") or {}
    req = ps.get("required") or []
    if len(req) == 1: return req[0]
    props = list((ps.get("properties") or {}).keys())
    return props[0] if len(props) == 1 else None

# Function-call syntaxes small models wrap around a command: shell:run("ls"), shell.run(command="ls"),
# shell("ls"), run_shell(command='ls'), developer__shell({"command": "ls"}), functions.shell(...).
_CALL_SYN = re.compile(r'^\s*`?\s*(?P<chain>[A-Za-z_][\w\-]*(?:\s*(?:[:.]|::)\s*[A-Za-z_][\w\-]*)*)\s*\((?P<args>.*)\)\s*;?\s*`?\s*$', re.S)
_SHELL_ALIASES = {"run", "exec", "execute", "bash", "sh", "zsh", "terminal", "cmd", "command", "run_shell", "shell_run",
                  "run_command", "execute_command", "exec_command", "run_shell_command", "execute_shell", "shell_exec",
                  "developer__shell", "default_api", "functions", "tools", "tool", "developer", "subprocess", "os", "system"}

def _resolve_chain(chain, names):
    """'shell:run' / 'developer__shell' / 'run_shell' -> a tool name that exists, else None."""
    segs = [x for x in re.split(r"\s*(?:::|[:.])\s*", chain) if x]
    for seg in segs:                                  # an exact tool name anywhere in the chain wins
        n = _norm_call({"name": seg, "arguments": {}}, names)
        if n: return n[0]
    if "shell" in names and segs and all(x.lower() in _SHELL_ALIASES or "shell" in x.lower() for x in segs):
        return "shell"
    return None

def _str_lit(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
        if s[0] == '"':
            try: return json.loads(s)
            except ValueError: pass
        body = s[1:-1]
        return re.sub(r"\\(['\"`\\])", r"\1", body)
    return None

def _parse_call_args(a):
    """Inside of (...) -> ('pos', str) | ('kw', dict) | ('obj', dict) | ('none', None) | None."""
    a = a.strip()
    if not a: return ("none", None)
    if a.startswith("{"):
        try:
            o = json.loads(a)
            if isinstance(o, dict): return ("obj", o)
        except ValueError:
            o = _loose_obj(a)
            if o: return ("obj", o)
        return None
    lit = _str_lit(a)
    if lit is not None: return ("pos", lit)
    kw = {}
    for m in re.finditer(r'\s*([A-Za-z_]\w*)\s*[=:]\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|-?\d+(?:\.\d+)?|true|false|True|False)\s*(?:,|$)', a):
        k, v = m.group(1), m.group(2)
        sv = _str_lit(v)
        if sv is not None: kw[k] = sv
        elif v in ("true", "True"): kw[k] = True
        elif v in ("false", "False"): kw[k] = False
        else: kw[k] = float(v) if "." in v else int(v)
    return ("kw", kw) if kw else None

def unwrap_call(text, names, by_name):
    """A whole string that is a function-call wrapper -> (tool, args) or None. Never guesses from prose."""
    m = _CALL_SYN.match(text or "")
    if not m: return None
    name = _resolve_chain(m.group("chain"), names)
    if not name: return None
    pa = _parse_call_args(m.group("args"))
    if pa is None: return None
    kind, val = pa
    p1 = _single_param(by_name.get(name)) or ("command" if name == "shell" else None)
    if kind == "none": return (name, {})
    if kind == "pos": return (name, {p1: val}) if p1 else None
    args = val
    if kind == "obj" and len(args) == 1:          # {"command": {"command": "ls"}} / {"arguments": {...}}
        (k, v), = args.items()
        if isinstance(v, dict) and (k in ("arguments", "args", "params", "parameters", "input") or k == p1): args = v
    if p1 and p1 not in args:                      # run_shell(cmd="ls") -> command
        alt = [k for k in ("cmd", "command", "input", "script", "code", "args") if isinstance(args.get(k), str)]
        if len(alt) == 1 and len(args) == 1: args = {p1: args[alt[0]]}
    return (name, args)

def fix_args(name, args, names, by_name):
    """Unwrap a wrapper left inside an argument: {"command": "shell:run(\"ls\")"} -> {"command": "ls"}."""
    if not isinstance(args, dict): return name, {}
    p1 = _single_param(by_name.get(name)) or ("command" if name == "shell" else None)
    for _ in range(3):
        v = args.get(p1) if p1 else None
        if isinstance(v, dict): args = {**{k: x for k, x in args.items() if k != p1}, **v}; continue
        if not isinstance(v, str): break
        u = unwrap_call(v.strip(), names, by_name)
        if not u or u[0] != name: break
        args = {**{k: x for k, x in args.items() if k != p1}, **u[1]}
    return name, args

def _clean_reply_raw(text, tools, last_role="user"):
    """Returns (calls, content). calls = [(name, args)]."""
    names = [t["function"]["name"] for t in tools or []]
    by_name = {t["function"]["name"]: t for t in tools or []}
    t = MARKERS.sub("", text or "")
    t = _INFO.sub("", t); t = _WRAP.sub("", t)
    had_fake = bool(_FAKE_OUT.search(t))
    t = _FAKE_OUT.sub("\n\x00\n", t).strip()   # \x00 marks where an invented result was
    if not names: return [], _clean_nul(t)
    m = _TAGGED.search(t)
    if m:
        attrs, body = m.group(1) or "", m.group(2)
        if not m.group(0).rstrip().endswith("</tool_call>"):   # unclosed: the call ends at the invented result or a blank line
            body = re.split(r"\x00|\n[ \t]*\n", body.strip(), maxsplit=1)[0]
        body = body.replace("\x00", "").strip()
        an = re.search(r'name\s*=\s*"([^"]+)"', attrs) or re.search(r'id\s*=\s*"([A-Za-z_][\w.\-]*?)(?:[:_]\d+)?"', attrs)
        calls, _ = extract_tool_calls(body, names) if "{" in body else ([], "")
        if not calls and an:
            n = _norm_call({"name": an.group(1), "arguments": {}}, names)
            if n:
                name = n[0]
                try: args = json.loads(body) if body.startswith("{") else None
                except ValueError: args = None
                if not isinstance(args, dict):
                    u = unwrap_call(body, names, by_name)
                    if u and u[0] == name: args = u[1]
                    else:
                        p1 = _single_param(by_name[name])
                        args = {p1: body} if (p1 and body) else {}
                calls = [(name, args)]
        if not calls:
            u = unwrap_call(body, names, by_name)
            if u: calls = [u]
        if calls: return calls, _clean_nul(t[:m.start()])
    t = _clean_nul(t)
    calls, left = extract_tool_calls(t, names)
    if calls:
        return calls, left
    if last_role == "user":   # a line that is exactly a call to an offered tool: shell:run("ls")
        for i, line in enumerate(t.splitlines()):
            u = unwrap_call(line.strip(), names, by_name) if "(" in line else None
            if u: return [u], "\n".join(t.splitlines()[:i]).strip()
    if "shell" in names and last_role == "user":
        fences = _FENCE.findall(t)
        if len(fences) == 1:
            lang, body = fences[0]
            body = "\n".join(l[2:] if l.startswith("$ ") else l for l in body.strip().splitlines())
            prose = _FENCE.sub("", t).strip()
            if body and len(body.splitlines()) <= 5 and len(prose) <= 200 and (lang or len(body.splitlines()) == 1 or had_fake):
                p1 = _single_param(by_name["shell"]) or "command"
                return [("shell", {p1: body})], t[:t.find("```")].strip()
    return [], t

def clean_reply(text, tools, last_role="user"):
    """Returns (calls, content). calls = [(name, args)] with wrapper syntax removed from the arguments."""
    names = [t["function"]["name"] for t in tools or []]
    by_name = {t["function"]["name"]: t for t in tools or []}
    calls, content = _clean_reply_raw(text, tools, last_role)
    return [fix_args(n, a, names, by_name) for n, a in calls], content

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

        pristine = json.loads(json.dumps(req))
        cwd = peer_cwd(self.client_address[1]) if self.client_address and self.client_address[0] in ("127.0.0.1", "::1") else None
        ctx_changes = prepare_context(req, cwd)
        log(kind="context", changes=ctx_changes[:20], est_tokens=est_tokens(req))
        fwd = json.dumps(req).encode()
        code, ct, body = self._upstream("POST", self.path, fwd)
        if code >= 500 and is_ctx_overflow(body.decode("utf-8", "replace")):   # our estimate was low: retry once, tighter
            req = pristine; req.pop("_stop", None)
            ctx_changes = prepare_context(req, cwd, budget=int((CTX - RESERVE) * 0.7))
            log(kind="context_retry", changes=ctx_changes[:20], est_tokens=est_tokens(req))
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
        last_role = (req.get("messages") or [{}])[-1].get("role", "user")
        if not msg.get("tool_calls"):
            parsed, left = clean_reply(raw_content, tools, last_role)
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
        # reply cleanup: the replies Chris got from real fm serve + Goose 1.56 (2026-10-23)
        T = [{"type": "function", "function": {"name": "shell", "parameters": {"type": "object",
              "properties": {"command": {"type": "string"}, "timeout_secs": {"type": "integer"}}, "required": ["command"]}}},
             {"type": "function", "function": {"name": "tree", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
        c, t = clean_reply("<turn_context>\nThe current operational context indicates no explicit reference to a specific directory. "
                           "Could you please clarify?\n</turn_context>", T)
        assert c == [] and t.startswith("The current operational context") and "turn_context" not in t, (c, t)
        chris3 = ('<turn_context>\nThe context does not indicate that this directory exists.\n</turn_context>\n<tool_call id="shell:1">\n'
                  'python3 -c "import os; print(os.listdir())"\n\n<stdout>\n[\'run.sh\', \'README.md\']\n</stdout>\n'
                  'The current working directory contains the following files:\n```\nrun.sh\nREADME.md\n```')
        c, t = clean_reply(chris3, T)
        assert c == [("shell", {"command": 'python3 -c "import os; print(os.listdir())"'})], c
        assert t == "The context does not indicate that this directory exists.", t
        assert clean_reply('<tool_call name="tree">{"path": "."}</tool_call>', T)[0] == [("tree", {"path": "."})]
        assert clean_reply("I'll list them.\n```bash\nls -la\n```", T)[0] == [("shell", {"command": "ls -la"})]
        assert clean_reply("I'll list them.\n```bash\nls -la\n```", T, "tool")[0] == []          # after a result: prose
        assert clean_reply("Use `ls` to list files. In Python:\n```python\nimport os\n```", T)[0] == []
        assert clean_reply("The weather is sunny.", T) == ([], "The weather is sunny.")
        c, t = clean_reply("<turn-context>\n<current-time>2026-10-23 22:00:00 -05:00</current-time>\n"
                           "<working-directory>/x</working-directory>\n</turn-context>\nThe files are a and b.", T)
        assert c == [] and t == "The files are a and b.", t
        # context: Goose's turn-context section removed, cwd line added, oversized history trimmed
        sysp = ("You are a general-purpose AI agent called goose.\ngoose is being developed as an open-source software project.\n\n"
                "# Turn Context\n\nEach turn may include a `<turn-context>` block.\n\n# Response Guidelines\n\nUse Markdown.")
        big = {"messages": [{"role": "system", "content": sysp}] +
               [m for i in range(30) for m in ({"role": "user", "content": "q%d " % i + "x" * 400}, {"role": "assistant", "content": "a" * 400})] +
               [{"role": "user", "content": "<turn-context>\n<working-directory>/w d</working-directory>\n</turn-context>"},
                {"role": "user", "content": "What files are here?"}], "tools": T}
        ch = prepare_context(big, cwd="/peer")
        s0 = big["messages"][0]["content"]
        assert "Working directory: /w d" in s0 and "# Turn Context" not in s0 and "open-source" not in s0, s0
        assert big["messages"][-1]["content"] == "What files are here?" and est_tokens(big) <= CTX - RESERVE, ch
        assert all("turn-context" not in (m.get("content") or "") for m in big["messages"][1:])
        # wrapper syntaxes seen from the on-device model (2026-10-23: `shell:run("ls")` ran as a literal command)
        LS = [("shell", {"command": "ls"})]
        for w in ['<tool_call id="shell:1">shell:run("ls")</tool_call>', '<tool_call id="shell:1">\nshell:run("ls")\n\n<stdout>x</stdout>',
                  '<tool_call name="shell">shell.run(command="ls")</tool_call>', "<tool_call>shell('ls')</tool_call>",
                  'I\'ll check the files.\nshell:run("ls")', 'shell("ls")', "run_shell(command='ls')", 'run_shell(cmd="ls")',
                  'developer__shell({"command": "ls"})', 'functions.shell({"command": "ls"})', 'default_api:shell(command="ls")',
                  '<tool_call name="shell">{"command": "shell:run(\\"ls\\")"}</tool_call>', '{"name": "shell", "arguments": {"command": {"command": "ls"}}}',
                  "```bash\nshell:run(\"ls\")\n```", '<tool_call id="shell:1">`shell.run("ls")`</tool_call>']:
            c, t = clean_reply(w, T)
            assert c == LS, (w, c)
        assert clean_reply('shell.run(command="ls -la", timeout_secs=30)', T)[0] == [("shell", {"command": "ls -la", "timeout_secs": 30})]
        assert clean_reply('tree(path=".")', T)[0] == [("tree", {"path": "."})]
        assert clean_reply('shell:run("echo \\"a b\\"")', T)[0] == [("shell", {"command": 'echo "a b"'})]
        assert clean_reply('print("hello")', T)[0] == []                     # not a tool
        assert clean_reply('In Python, call shell("ls") to list.', T)[0] == []  # prose, not a whole line
        assert clean_reply('subprocess.run(["ls"])', T)[0] == []              # unparseable args: leave it
        assert clean_reply('shell:run("ls")', T, "tool")[0] == []             # after a tool result: prose
        for n, a in [clean_reply(w, T)[0][0] for w in ['<tool_call id="shell:1">shell:run("ls")</tool_call>']]:
            assert "(" not in a["command"]
        print("goose context + reply cleanup: ok")
        sys.exit(0)
    print(f"fm_tool_proxy on 127.0.0.1:{PORT} -> {UP} (n_ctx={CTX})", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
