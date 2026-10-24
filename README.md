# fm serve ↔ Goose prototype

Hooking Goose (aaif-goose/goose, CLI **1.56.0**; source checked at commit `c3bfc5f`, 2026-10-23) up to Apple's `fm serve` (macOS 27, `fm` 2.0.68.1.402).

**How much of this was tested:** everything here was built and run on a **Linux** box against
`mock_fm_serve.py`. That mock copies the fm serve behaviour that third parties have documented, and it is not
Apple's server. Labels used below: **[TESTED-MOCK]** means real Goose against the mock. **[SRC]** means a published source says so.
**[ASSUMED]** means I found no source. **[UNTESTED]** means it needs a Mac running macOS 27.

## 1. Minimal Goose config

```bash
# env (wins over config.yaml; note OPENAI_HOST env overrides the config file)
export GOOSE_PROVIDER=openai
export GOOSE_MODEL=system                  # the only model fm serve accepts [SRC fm-proxy]
export OPENAI_HOST=http://127.0.0.1:1977   # fm_tool_proxy; :1976 = fm serve directly (chat only)
export OPENAI_BASE_PATH=v1/chat/completions
export GOOSE_CONTEXT_LIMIT=4096            # 8192 on Macs that get AFM 3 Core Advanced [SRC apfel, spacecorps]
# OPENAI_API_KEY: not needed. Goose 1.56 sends no auth when it is unset [TESTED-MOCK + source],
# and fm serve has no auth [SRC apfel, spacecorps]. Setting any dummy value is harmless.
goose run --no-profile --with-builtin developer -t "..."
```
`config.example.yaml` is the same thing as a persisted `~/.config/goose/config.yaml`. It turns off every
default platform extension, keeps only `developer`, and limits that to `available_tools: [shell]`.
Note that the Extension Manager's config key is `extensionmanager`.

## 2. Results [TESTED-MOCK] (`./run_mock_tests.sh`, output in `logs/test_results.txt`)
Token counts are cl100k estimates. Apple's tokenizer and its tool-rendering format are different, so on a Mac run `fm count-tokens` to get real numbers.

| # | Setup | Request size (est. tokens) | Outcome |
|---|---|---|---|
| A | direct :1976, Goose default extensions (17 tools) | **~4,280** (system 850 + tool schemas 3,321) | Mock returns 500 "transcript exceeded the model's context size". Goose treats it as a server error and retries 3x. No compaction. **Fails.** |
| B | direct :1976, `--no-profile --with-builtin developer` (6 tools) | ~1,770 | Fits. The regression shows up: Goose prints `<start_of_turn>model {"tool_call":[...]}<ctrl46>` as text and runs no tool. |
| C | via proxy :1977, same as B | ~1,720 → 1,790 | Proxy turns the content into `tool_calls`. Goose **runs `shell`**, sends the tool result back, and gets a final answer. **Works.** |
| D | via proxy, default extensions | ~4,230–4,305 | Proxy maps the error to 400 `context_length_exceeded`. Goose **auto-compacts and retries twice, then stops**. Tools plus system prompt alone are over 4096, so compaction can't fix it. |
| E | `config.example.yaml` only (shell + load_skill), no API key | **~1,008** first turn | Full tool loop works through the proxy. |
| F | E + `GOOSE_TOOLSHIM=true` | ~1,051 (no `tools` array; tool text moved into the system prompt) | Plain chat. Tool calls would need Goose's toolshim interpreter, which calls Ollama (`mistral-nemo` by default). Not tested. |
| G | multi-turn session (`-n fmtest -r`), lean config | 1,008 → 3,759 | When it passes 80% of 4096 (~3.3k), Goose compacts automatically. Its summarisation request was ~3.4k tokens, which is **close to overflowing by itself**. |

Other findings from the source and the runs:
- Every `goose run` first sends a ~440-token title-generation request with no tools.
- `load_skill` is always injected (SkillOperation) and costs ~170 tokens. I found no config switch that removes it.
- Goose streams (`stream:true` with `stream_options.include_usage`).

### 2a. Fixed: `400 Invalid request: The data couldn't be read because it is missing.` [TESTED-MOCK + real Mac report]
The first real Mac run (macOS 27, fm 2.0.68) failed the README smoke test with that 400 on Goose's first
request that carries tools. Cause: `--with-builtin developer` includes Goose's `read_image` tool, whose
parameters use a JSON Schema reference: `"crop": {"$ref": "#/$defs/CropParams", "default": null}`. fm serve
decodes tool schemas with strict Swift `Codable` types, doesn't understand `$ref`/`$defs` [SRC fm-proxy], and a
property with no `type` is a missing key, which is exactly Swift's "data couldn't be read because it is missing".
The lean `config.example.yaml` (shell + load_skill) never sends `read_image`, so it wasn't affected.

`fm_tool_proxy.py` now normalises every request (`normalize_request`): `$ref`/`$defs` inlined, anyOf/oneOf/allOf
and `type` arrays collapsed, every property gets a `type`, every object gets `properties` and `required`,
decorative and validation keys (`title`, `default`, `$schema`, `format`, `minimum`, `strict`) dropped,
`developer` mapped to `system`, null or array `content` turned into a string, and `stop` applied by the proxy.
On any 4xx from fm serve it logs a field summary to `runs/proxy.jsonl` and saves the exact forwarded body to
`runs/last_upstream_400.json`. The mock is now strict (`FM_MOCK_STRICT=1`, default) and `./test_goose_smoke.sh`
checks both halves: Goose straight to the mock reproduces the 400, and Goose through the proxy prints hello.
`GET :1977/proxy/version` reports the running proxy's version.

### 2b. Fixed: `<turn_context>` echoes, "working directory is not defined", text tool calls [TESTED-MOCK + real Mac report]
On a real Mac the interactive session answered "What files are in this directory?" with a `<turn_context>…</turn_context>`
wrapper asking which directory, and later wrote `<tool_call id="shell:1">python3 -c …` with an invented `<stdout>` as
plain text, which Goose never ran. Causes, from Goose 1.56.0 source (`crates/goose/src/agents/moim.rs`) and captured requests:
- Goose's system prompt always has a `# Turn Context` section describing a `<turn-context>` block (time, working
  directory, turn budget), but Goose only sends that block when the context limit is **at least 32,000**
  (`MIN_CONTEXT_FOR_MOIM`). At `GOOSE_CONTEXT_LIMIT=4096` nothing carries the working directory, so the model doesn't
  know it and copies the tag name into its reply.
- fm 2.0.68 never fills `tool_calls` (§3), and the old proxy only parsed JSON-shaped leaks.

`fm_tool_proxy.py` now:
- removes that section and some Goose boilerplate from the system prompt
- adds one plain line: `Working directory: …` (from a real `<turn-context>` block if present, else the cwd of the Goose
  process on the other end of the loopback connection), the current time, and "call tools, don't write their output"
- drops the oldest turns and trims long tool output to fit `FM_CTX`, and retries once, tighter, if fm still overflows
- strips echoed `<turn_context>`/`<turn-context>` tags, drops invented `<stdout>`/`<output>` blocks and anything written
  after a call, and turns `<tool_call …>` blocks, JSON blobs, and a lone ```` ```bash ```` block (only when `shell` is
  offered and the user spoke last) into real `tool_calls`

`./test_goose_session.sh` runs real Goose for two turns against `FM_MOCK_MODE=small`, which imitates these replies.
**Expect** the on-device model to still be weak: it picks the wrong command or path, misses tool calls, or answers
from guesswork, and each reply takes several seconds. The proxy can fix the format of what it says, not its judgement.

## 3. The tool-calling regression [SRC]
- In fm 2.0.68.x (Beta 5 onward, and in the RC), `tool_choice:"auto"` **never fills `tool_calls`**, and `finish_reason` is `stop`. Spacecorps saw 0 of 54. When the model does call a tool, the call text leaks into `content`, for example
  `<start_of_turn>model\n{"tool_call": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]}`. Spacecorps saw this in 13 of 45 replies, and fm-proxy reported markers in 7–9 of 10. Other replies **invent the tool result instead**. Forced `tool_choice` returns 500 "An unsupported generation guide was used." Tool calling reportedly worked in 2.0.62.
- Sources disagree on part of this. One teardown says any `tools` array gets 400 "Invalid tool definition", and apfel says it leaks template tokens. Check on your build.

## 4. Workarounds
1. **`fm_tool_proxy.py`** (stdlib Python, port 1977 → 1976) [TESTED-MOCK]. It parses leaked calls (`{"tool_call":[..]}`, `<tool_call>`, bare `{name,arguments}`, `[default_api:fn{..}]`) into OpenAI `tool_calls`. It also:
   - strips template markers and drops tool names that weren't offered
   - always sends `stream:false` upstream and rebuilds SSE for Goose
   - maps fm's 500 context error to 400 `context_length_exceeded`, so Goose compacts instead of retrying
   - adds `meta.n_ctx` to `/v1/models`
   - fills in missing tool descriptions
   - drops forced `tool_choice`

   **Limit:** it only helps when the model actually writes the call into `content`. When the model invents a result, nothing can be recovered. On real hardware expect intermittent tool use, not a reliable loop [inferred from SRC rates].
2. **Turn tools off:** use `GOOSE_TOOLSHIM=true` (no `tools` array) or a chat-only setup. Requests stay small, but the agent can't do anything.
3. **`fm_sdk_server.py`** (port 1978) [UNTESTED]. This is an OpenAI-compatible server on top of `apple_fm_sdk`'s native `Tool` API, which works according to spacecorps.
   - Each OpenAI tool becomes a dynamic `fm.Tool`. `call()` waits on a future until Goose's next request brings the `role:"tool"` result.
   - Install with `pip install apple-fm-sdk fastapi uvicorn`.
   - Known gaps: prior history is flattened into the prompt, the JSON-schema mapping is shallow, and there is no tool_choice. A Swift alternative is spacecorps' OpenAppleModelsServer.
4. Also consider gregbarbosa/fm-proxy (Node). It does the same envelope fixes and turns *forced* tool_choice into a `response_format` schema, but auto mode is still broken there.

## 5. Context ceiling
fm-proxy measured 4096 on the RC (4045 tokens pass, ~4255 overflow). apfel reports 4096 on an M2 Air and 8192 on M3 or later with 12 GB or more. Spacecorps saw 8192. What this means for Goose:
- The default profile (~4.3k) **cannot run at 4096**. At 8192 it barely fits.
- The lean profile (~1.0k) leaves about 2.2k tokens of work before the 80% compaction threshold.
- Keep the working directory free of a large `AGENTS.md` or hints file. fm-proxy saw Pi go from 1,188 to 9,582 tokens because of one.
- Set `GOOSE_CONTEXT_LIMIT`, because Goose otherwise defaults to 128k.

## 6. Run
- Mock (Linux or Mac): `./run_mock_tests.sh` (size table), `./test_goose_smoke.sh` (strict regression test), `./test_goose_session.sh` (multi-turn, small-model replies), `python3 fm_tool_proxy.py --selftest`
- Mac: `sudo fm license` once, then `./start_mac.sh`, then the env from §1. Runtime logs (`proxy.jsonl`, `last_upstream_400.json`, `stack.out`) go to `runs/`, which git ignores.
- Files: `mock_fm_serve.py`, `fm_tool_proxy.py` (`--selftest` runs the parser tests), `fm_sdk_server.py`, `config.example.yaml`, `start_mac.sh`, `run_mock_tests.sh`, `test_goose_smoke.sh`, `test_goose_session.sh`, `logs/` (request logs and bodies from the original mock runs), `bin/goose`.

## Sources
- fm-proxy README + AGENTS.md (live audit, 27.0 RC, fm 2.0.68.1.402): https://github.com/gregbarbosa/fm-proxy · https://raw.githubusercontent.com/gregbarbosa/fm-proxy/main/AGENTS.md · fm CLI help: https://raw.githubusercontent.com/gregbarbosa/fm-proxy/main/docs/fm-reference.md
- spacecorps research + server: https://spacecorps.github.io/open-apple-models/RESEARCH.md · https://spacecorps.github.io/open-apple-models/SERVER.md
- apfel vs fm: https://cdn.jsdelivr.net/gh/arthur-ficial/apfel@main/docs/apfel-vs-fm.md
- fm teardown (search snippet only; README fetch returned 404): https://github.com/llostinthesauce/fm-teardown
- apple_fm_sdk: https://apple.github.io/python-apple-fm-sdk/tools.html · https://github.com/apple/python-apple-fm-sdk
- Goose source: `crates/goose/src/providers/openai_def.rs` (host/base-path/API key resolution), `crates/goose-providers/src/http_status.rs` (context-error matching), `crates/goose/src/providers/toolshim.rs`, `crates/goose/src/agents/platform_extensions/mod.rs`, `crates/goose-context-management/src/lib.rs` (`DEFAULT_COMPACTION_THRESHOLD = 0.8`)
- **No source found**: the exact JSON shape of fm serve's `/v1/models` and `/health` responses (the mock's version is assumed), and fm serve's exact 400 message wording for `stop` or a missing tool description.
