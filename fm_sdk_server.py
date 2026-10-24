#!/usr/bin/env python3
"""fm_sdk_server: OpenAI Chat Completions server backed by apple_fm_sdk with REAL tool calls.

*** UNTESTED: written from the apple_fm_sdk docs/source (github.com/apple/python-apple-fm-sdk,
commit 0249f35, 2026-10-07). Needs macOS 27 + Apple Intelligence; cannot run on Linux. ***

Why: fm serve (fm 2.0.68.x) never emits `tool_calls`. The SDK's native Tool protocol works
(spacecorps RESEARCH.md), but SDK tools run *in-process*: the model calls Tool.call() and waits
for a string. An OpenAI client (Goose) instead expects the server to return `tool_calls` and to
receive results in a later request. We bridge that by suspending Tool.call() on an
asyncio.Future keyed by tool_call_id; the next HTTP request carrying role:"tool" results resolves
it and the same LanguageModelSession continues. Same idea as spacecorps' OpenAppleModelsServer
(Swift). State is in-memory, one process, so run a single worker.

  pip install apple-fm-sdk fastapi uvicorn
  python fm_sdk_server.py            # listens on 127.0.0.1:1978
  Goose: OPENAI_HOST=http://127.0.0.1:1978  GOOSE_MODEL=system

Limitations (by design or unverified):
 - Prior turns are flattened into the prompt text for a *new* session (Transcript.from_dict's
   format is not documented enough to synthesise one).
 - JSON-Schema -> GenerationSchema mapping is shallow: str/int/float/bool; arrays/objects are
   passed as JSON strings. Required/optional nuance is not expressed.
 - No tool_choice control in the Python SDK (spacecorps RESEARCH.md §1).
"""
import asyncio, json, time, uuid
from typing import Any, Dict, List, Optional

import apple_fm_sdk as fm
from apple_fm_sdk.generation_property import Property
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

app = FastAPI()
MODEL = fm.SystemLanguageModel()
PENDING: Dict[str, "Turn"] = {}          # tool_call_id -> Turn awaiting its result
PY = {"string": str, "integer": int, "number": float, "boolean": bool}

class Turn:
    """One in-flight LanguageModelSession.respond() that may pause on external tool calls."""
    def __init__(self):
        self.events: asyncio.Queue = asyncio.Queue()
        self.futures: Dict[str, asyncio.Future] = {}
        self.task: Optional[asyncio.Task] = None

def make_tool(turn: Turn, spec: dict) -> fm.Tool:
    fn = spec.get("function", {})
    params = fn.get("parameters") or {}
    props = []
    for pname, p in (params.get("properties") or {}).items():
        t = PY.get(p.get("type"), str)
        desc = p.get("description") or pname
        if p.get("type") in ("array", "object"):
            desc += " (JSON-encoded)"
        props.append(Property(name=pname, type_class=t, description=desc))
    schema_cls = type(f"{fn['name']}_Args", (), {})
    schema = fm.GenerationSchema(type_class=schema_cls, description=fn.get("description") or fn["name"], properties=props)

    class ExternalTool(fm.Tool):
        name = fn["name"]
        description = fn.get("description") or fn["name"]
        @property
        def arguments_schema(self):
            return schema
        async def call(self, args: fm.GeneratedContent) -> str:
            try:
                a = json.loads(args.to_json())
            except Exception:
                a = {}
            cid = "call_" + uuid.uuid4().hex[:24]
            fut = asyncio.get_running_loop().create_future()
            turn.futures[cid] = fut
            PENDING[cid] = turn
            await turn.events.put(("tool_call", {"id": cid, "type": "function",
                                                 "function": {"name": fn["name"], "arguments": json.dumps(a)}}))
            return await fut            # resolved by the client's next request (role: tool)
    return ExternalTool()

def text_of(c):
    if c is None: return ""
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c if isinstance(p, dict))

async def run_turn(turn: Turn, instructions: str, prompt: str, tools: List[dict]):
    try:
        session = fm.LanguageModelSession(instructions=instructions or None, model=MODEL,
                                          tools=[make_tool(turn, t) for t in tools] or None)
        resp = await session.respond(prompt)
        await turn.events.put(("final", str(resp)))
    except fm.ExceededContextWindowSizeError as e:
        await turn.events.put(("error", (400, "context_length_exceeded", f"context length exceeded: {e}")))
    except fm.GuardrailViolationError as e:
        await turn.events.put(("error", (400, "content_filter", str(e))))
    except Exception as e:
        await turn.events.put(("error", (500, "internal_error", repr(e))))

async def next_step(turn: Turn):
    """Wait for the model's next step: final text, error, or a batch of tool calls."""
    kind, val = await turn.events.get()
    if kind != "tool_call":
        return kind, val
    calls = [val]
    while True:                         # gather parallel calls emitted together
        try:
            k, v = await asyncio.wait_for(turn.events.get(), timeout=0.05)
        except asyncio.TimeoutError:
            break
        if k == "tool_call": calls.append(v)
        else: turn.events.put_nowait((k, v)); break
    return "tool_calls", calls

def completion(content, tool_calls=None, usage=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls: msg["tool_calls"] = tool_calls
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion", "created": int(time.time()),
            "model": "system", "choices": [{"index": 0, "message": msg,
                                            "finish_reason": "tool_calls" if tool_calls else "stop"}], "usage": usage}

def sse(resp):
    async def gen():
        base = {k: resp[k] for k in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
        m = resp["choices"][0]["message"]
        yield "data: " + json.dumps(base | {"choices": [{"index": 0, "delta": {"role": "assistant"}}]}) + "\n\n"
        if m.get("content"):
            yield "data: " + json.dumps(base | {"choices": [{"index": 0, "delta": {"content": m["content"]}}]}) + "\n\n"
        for i, c in enumerate(m.get("tool_calls") or []):
            yield "data: " + json.dumps(base | {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": i} | c]}}]}) + "\n\n"
        yield "data: " + json.dumps(base | {"choices": [{"index": 0, "delta": {}, "finish_reason": resp["choices"][0]["finish_reason"]}]}) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "system", "object": "model", "owned_by": "apple",
                                        "meta": {"n_ctx": MODEL.context_size}}]}

@app.post("/v1/chat/completions")
async def chat(request: Request):
    req = await request.json()
    msgs = req.get("messages", [])
    tools = req.get("tools") or []
    if req.get("tool_choice") == "none": tools = []
    # 1) Continuation: trailing role:tool messages answer pending calls.
    trailing = []
    for m in reversed(msgs):
        if m.get("role") != "tool": break
        trailing.append(m)
    turn = None
    for m in trailing:
        t = PENDING.pop(m.get("tool_call_id"), None)
        if t:
            turn = t
            fut = t.futures.pop(m["tool_call_id"])
            if not fut.done(): fut.set_result(text_of(m.get("content")))
    # 2) New turn otherwise.
    if turn is None:
        turn = Turn()
        instructions = "\n".join(text_of(m.get("content")) for m in msgs if m.get("role") in ("system", "developer"))
        convo = [m for m in msgs if m.get("role") not in ("system", "developer")]
        history = "\n".join(f"{m['role']}: {text_of(m.get('content')) or json.dumps(m.get('tool_calls'))}" for m in convo[:-1])
        last = text_of(convo[-1].get("content")) if convo else ""
        prompt = (f"Conversation so far:\n{history}\n\nuser: {last}" if history else last)
        turn.task = asyncio.create_task(run_turn(turn, instructions, prompt, tools))
    kind, val = await next_step(turn)
    if kind == "error":
        status, code, msg = val
        return JSONResponse({"error": {"type": "invalid_request_error" if status == 400 else "server_error",
                                       "code": code, "message": msg}}, status_code=status)
    resp = completion(None, val) if kind == "tool_calls" else completion(val)
    return sse(resp) if req.get("stream") else JSONResponse(resp)

if __name__ == "__main__":
    ok, reason = MODEL.is_available()
    print(f"model available={ok} reason={reason} context_size={MODEL.context_size}")
    uvicorn.run(app, host="127.0.0.1", port=1978, workers=1)
