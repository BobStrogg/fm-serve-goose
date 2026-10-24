#!/usr/bin/env bash
# Reproduce the Linux mock tests. Starts mock fm serve (:1976, regression mode) + proxy (:1977),
# runs Goose scenarios, prints request-size table. Usage: ./run_mock_tests.sh
# (Strict request validation is off here to keep these size measurements comparable;
#  ./test_goose_smoke.sh is the strict regression test.)
set -u
cd "$(dirname "$0")"; D=$PWD; PY=${PY:-$D/.venv/bin/python}; [ -x "$PY" ] || PY=python3
pkill -f mock_fm_serve.py 2>/dev/null; pkill -f fm_tool_proxy.py 2>/dev/null; sleep 0.5
mkdir -p logs runs/empty; rm -rf logs/bodies
FM_MOCK_MODE=regression FM_MOCK_STRICT=0 FM_MOCK_LOG=logs/mock_requests.jsonl $PY mock_fm_serve.py > logs/mock.out 2>&1 &
PROXY_LOG=logs/proxy.jsonl python3 fm_tool_proxy.py > logs/proxy.out 2>&1 &
sleep 1
export HOME=$D/home GOOSE_DISABLE_KEYRING=1 GOOSE_PROVIDER=openai GOOSE_MODEL=system \
       OPENAI_BASE_PATH=v1/chat/completions OPENAI_API_KEY=sk-local
mkdir -p "$HOME/.config/goose"; : > "$HOME/.config/goose/config.yaml"   # A-C: env-only, Goose defaults
G="$D/bin/goose run --no-session"; cd runs/empty
summ(){ python3 - "$D/logs/mock_requests.jsonl" <<'P'
import json,sys
for l in open(sys.argv[1]):
    e=json.loads(l); print(f"   req: est_tokens={e['prompt_tokens_est']:5} system={e['system_tokens']:4} tools={e['tools_tokens']:4} (x{e['n_tools']}) msgs={e['messages_tokens']}")
P
}
run(){ echo; echo "### $1"; : > "$D/logs/mock_requests.jsonl"; shift; timeout 120 "$@" 2>&1 | grep -v '^\s*$' | tail -6; summ; }
run "A  direct :1976, default extensions"            env OPENAI_HOST=http://127.0.0.1:1976 $G -t "Say hi"
run "B  direct :1976, --no-profile --with-builtin developer (regression visible)" env OPENAI_HOST=http://127.0.0.1:1976 $G --no-profile --with-builtin developer -t "Run a shell command that prints hello"
run "C  proxy :1977, --no-profile --with-builtin developer (workaround)" env OPENAI_HOST=http://127.0.0.1:1977 $G --no-profile --with-builtin developer -t "Run a shell command that prints hello"
run "D  proxy :1977, default extensions, GOOSE_CONTEXT_LIMIT=4096" env OPENAI_HOST=http://127.0.0.1:1977 GOOSE_CONTEXT_LIMIT=4096 $G -t "Say hi"
cp "$D/config.example.yaml" "$HOME/.config/goose/config.yaml"
U="env -u OPENAI_HOST -u GOOSE_PROVIDER -u GOOSE_MODEL -u OPENAI_BASE_PATH -u OPENAI_API_KEY"
run "E  config.example.yaml only (lean: developer/shell), no API key" $U $G -t "Run a shell command that prints hello"
run "F  config.example.yaml + GOOSE_TOOLSHIM=true (no tools array sent)" $U GOOSE_TOOLSHIM=true $G -t "Run a shell command that prints hello"
pkill -f mock_fm_serve.py; pkill -f fm_tool_proxy.py
