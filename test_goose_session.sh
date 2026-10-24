#!/usr/bin/env bash
# Multi-turn regression test for what real fm serve + Goose 1.56 did on a Mac (2026-10-23):
# echoed <turn_context> wrappers, no working directory, tool calls written as TEXT with invented <stdout>.
# Mock: FM_MOCK_MODE=small (strict). For each text tool-call format (tagged, fenced, json), real Goose runs
#   turn 1: What files are in this directory?            -> must really run shell in Goose's cwd
#   turn 2: Try "<dir>/3D Models"  (resumed session)      -> must list that folder
# plus a control straight to the mock (no proxy), which reproduces the "please clarify" reply.
# Usage: ./test_goose_session.sh [path/to/goose]   Ports 19763/19773; throwaway HOME.
set -u
cd "$(dirname "$0")"; D=$PWD
G=${1:-$D/bin/goose}; [ -x "$G" ] || G=$(command -v goose) || { echo "no goose binary"; exit 2; }
PY=${PY:-python3}; T=$(mktemp -d); FAIL=0
W="$T/work dir"; mkdir -p "$W/3D Models" "$T/home/.config/goose"; touch "$W/alpha-notes.txt" "$W/beta.csv" "$W/3D Models/benchy.stl"
: > "$T/home/.config/goose/config.yaml"
g(){ (cd "$W" && HOME=$T/home GOOSE_DISABLE_KEYRING=1 GOOSE_PROVIDER=openai GOOSE_MODEL=system OPENAI_HOST=$1 \
      OPENAI_BASE_PATH=v1/chat/completions GOOSE_CONTEXT_LIMIT=4096 GOOSE_MODE=auto timeout 120 \
      "$G" run --no-profile --with-builtin developer "${@:2}" </dev/null 2>&1); }
start(){ FM_MOCK_STRICT=${STRICT:-1} FM_MOCK_PORT=19763 FM_MOCK_MODE=small FM_MOCK_TOOLFMT=$1 FM_MOCK_LOG=$T/mock.jsonl $PY mock_fm_serve.py >>$T/mock.out 2>&1 & MP=$!
  PROXY_PORT=19773 FM_UPSTREAM=http://127.0.0.1:19763 PROXY_LOG=$T/proxy.jsonl $PY fm_tool_proxy.py >>$T/proxy.out 2>&1 & PP=$!
  for i in $(seq 1 50); do curl -s -m 1 http://127.0.0.1:19773/proxy/version >/dev/null && curl -s -m 1 http://127.0.0.1:19763/v1/models >/dev/null && break; sleep 0.1; done; }
stop(){ kill $MP $PP 2>/dev/null; wait $MP $PP 2>/dev/null; }
trap 'stop; rm -rf "$T"' EXIT
check(){ # $1 label, $2 output, $3 must-contain
  if printf %s "$2" | grep -q "▸ shell" && printf %s "$2" | grep -qF "$3" && ! printf %s "$2" | grep -qE '<turn[-_]context|<tool_call|<stdout>|run\.sh'; then
    echo "PASS $1"; else echo "FAIL $1"; printf '%s\n' "$2" | grep -v '^\s*$' | tail -8 | sed 's/^/      /'; FAIL=1; fi; }
STRICT=0 start rotate   # the old proxy fixed the 400 but passed this shape through
OUT=$(g http://127.0.0.1:19763 --no-session -t "What files are in this directory?")
if printf %s "$OUT" | grep -q "clarify which directory"; then echo "PASS control: straight to the mock, Goose shows the Mac's failure (asks which directory)"
else echo "FAIL control"; printf '%s\n' "$OUT" | tail -4; FAIL=1; fi
stop
for fmt in tagged fenced json; do
  : > $T/mock.jsonl; start $fmt
  OUT=$(g http://127.0.0.1:19773 -n "s-$fmt" -t "What files are in this directory?")
  check "$fmt turn 1 (real shell run in Goose's cwd, listing alpha-notes.txt)" "$OUT" alpha-notes.txt
  OUT=$(g http://127.0.0.1:19773 -n "s-$fmt" -r -t "Try \"$W/3D Models\"")
  check "$fmt turn 2 (resumed; lists the quoted folder: benchy.stl)" "$OUT" benchy.stl
  [ $fmt = tagged ] && printf "%s\n" "$OUT" | grep -v "^\s*$" | tail -6 | sed "s/^/        /"
  grep -q "Working directory: $W " "$T/mock.jsonl" 2>/dev/null || grep -q "\"context line: cwd=peer\"" $T/proxy.jsonl || { echo "FAIL $fmt: cwd not conveyed"; FAIL=1; }
  stop
done
grep -o '"context line: cwd=[a-z-]*"' $T/proxy.jsonl | sort | uniq -c | sed 's/^/      proxy: /'
exit $FAIL
