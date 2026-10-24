#!/usr/bin/env bash
# Regression test for the Goose smoke test (setup script / README §1 prompt) against the STRICT mock.
#   1. Goose -> strict mock directly must reproduce fm serve's 400 ("...because it is missing.")
#   2. Goose -> fm_tool_proxy -> strict mock must run `shell` and print hello.
# Usage: ./test_goose_smoke.sh [path/to/goose]   (default: bin/goose, else goose on PATH)
# Uses ports 19761 (mock) / 19771 (proxy) and a throwaway HOME, so it never touches a running stack.
set -u
cd "$(dirname "$0")"; D=$PWD
G=${1:-$D/bin/goose}; [ -x "$G" ] || G=$(command -v goose) || { echo "no goose binary"; exit 2; }
PY=${PY:-python3}; T=$(mktemp -d); trap 'kill $MP $PP 2>/dev/null; rm -rf "$T"' EXIT
FM_MOCK_PORT=19761 FM_MOCK_STRICT=1 FM_MOCK_LOG=$T/mock.jsonl $PY mock_fm_serve.py >$T/mock.out 2>&1 & MP=$!
PROXY_PORT=19771 FM_UPSTREAM=http://127.0.0.1:19761 PROXY_LOG=$T/proxy.jsonl $PY fm_tool_proxy.py >$T/proxy.out 2>&1 & PP=$!
for i in $(seq 1 50); do curl -s -m 1 http://127.0.0.1:19771/proxy/version >/dev/null && curl -s -m 1 http://127.0.0.1:19761/v1/models >/dev/null && break; sleep 0.1; done
mkdir -p "$T/home/.config/goose" "$T/work"; : > "$T/home/.config/goose/config.yaml"
g(){ (cd "$T/work" && HOME=$T/home GOOSE_DISABLE_KEYRING=1 GOOSE_PROVIDER=openai GOOSE_MODEL=system OPENAI_HOST=$1 \
      OPENAI_BASE_PATH=v1/chat/completions GOOSE_CONTEXT_LIMIT=4096 GOOSE_MODE=auto timeout 120 \
      "$G" run --no-session --no-profile --with-builtin developer -t "Run a shell command that prints hello" </dev/null 2>&1); }
FAIL=0
OUT=$(g http://127.0.0.1:19761)
if printf %s "$OUT" | grep -q "because it is missing"; then echo "PASS 1: direct to strict mock reproduces the 400: $(printf %s "$OUT" | grep -m1 -o 'Bad request.*missing')"
  echo "        rejected field: $(grep -o '"rejected": "[^"]*"' $T/mock.jsonl | head -1)"
else echo "FAIL 1: expected the 400 when Goose talks to the strict mock directly"; printf '%s\n' "$OUT" | tail -5; FAIL=1; fi
: > $T/mock.jsonl
OUT=$(g http://127.0.0.1:19771)
if printf %s "$OUT" | grep -q "hello" && ! printf %s "$OUT" | grep -qi "error"; then echo "PASS 2: via proxy, Goose ran shell and printed hello"
  printf '%s\n' "$OUT" | grep -v '^\s*$' | tail -4 | sed 's/^/        /'
else echo "FAIL 2: via proxy"; printf '%s\n' "$OUT" | tail -8; tail -5 $T/proxy.out; FAIL=1; fi
grep -q '"rejected"' $T/mock.jsonl && { echo "FAIL 2b: the mock rejected a proxied request: $(grep -o '"rejected": "[^"]*"' $T/mock.jsonl)"; FAIL=1; }
exit $FAIL
