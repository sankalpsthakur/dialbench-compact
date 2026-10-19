#!/bin/bash
# Shell test of tools/gpu_push_wait.sh with a fake `kaggle` on PATH and a temporary benchmarks root: no network, no GPU, no real lock is touched.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d); mkdir -p "$T/bin" "$T/locks" "$T/gauges/logs/local" "$T/gauges/tools" "$T/gauges/kern"
cp "$HERE/tools/wait_kernel.sh" "$T/gauges/tools/"
: > "$T/RESOURCES.md"; : > "$T/shapes.txt"
fail=0; check() { if [ "$1" = "$2" ]; then echo "ok   $3"; else echo "FAIL $3 (got '$1', want '$2')"; fail=1; fi; }
fake() {   # $1 = remaining quota text, $2 = space-separated shapes the fake accepts (others are refused)
  : > "$T/shapes.txt"; : > "$T/tvals.txt"; : > "$T/dirs.txt"
  cat > "$T/bin/kaggle" <<EOF
#!/bin/bash
case "\$1 \$2" in
  "quota "*|"quota") printf 'resource  used    remaining  total   refreshAt\n--------  ------  ---------  ------  -------------------\nGPU       0.50h   $1  30.00h  2026-10-17T00:00:00\nTPU       1.37h   18.63h     20.00h  2026-10-17T00:00:00\n';;
  "kernels push")
    shape=""; tv=""; pd=""; while [ \$# -gt 0 ]; do [ "\$1" = "--accelerator" ] && shape=\$2; [ "\$1" = "-t" ] && tv=\$2; [ "\$1" = "-p" ] && pd=\$2; shift; done
    echo "\$shape" >> "$T/shapes.txt"; echo "\$tv" >> "$T/tvals.txt"; echo "\$pd" >> "$T/dirs.txt"
    if [ -f "$T/slots_full" ]; then echo "Kernel push error: Maximum batch GPU session count of 2 reached."; exit 0; fi
    case " $2 " in *" \$shape "*) echo "Kernel version 1 successfully pushed";; *) echo "Kernel push error: machine shape \$shape is not available";; esac;;
  "kernels status") echo 'x has status "KernelWorkerStatus.COMPLETE"';;
  "kernels output") echo "no files";;
esac
EOF
  chmod +x "$T/bin/kaggle"
}
export PATH="$T/bin:$PATH" BM="$T"
S="$HERE/tools/gpu_push_wait.sh"
ALL="NvidiaL4X1 NvidiaTeslaT4"
# 1. lock held by someone else: exit 3, nothing pushed, their lock untouched
fake 29.50h "$ALL"; mkdir "$T/locks/kaggle-gpu"; echo "gas something" > "$T/locks/kaggle-gpu/owner"
"$S" kern ref/x "$T/out" 10800 90 "note" > "$T/o1.txt" 2>&1; check $? 3 "lock held -> exit 3"
check "$(wc -l < "$T/shapes.txt" | tr -d ' ')" 0 "lock held -> no push attempted"
check "$(cat "$T/locks/kaggle-gpu/owner")" "gas something" "other owner's lock untouched"
rm -rf "$T/locks/kaggle-gpu"
# 2. quota too low in BILLED hours (2.50 h remaining for a 10,800 s = 3.00 h billed cap): exit 4, lock released, nothing pushed
fake 2.50h "$ALL"
"$S" kern ref/x "$T/out" 10800 90 "note" > "$T/o2.txt" 2>&1; check $? 6 "quota below the billed cap -> exit 6"
check "$([ -d "$T/locks/kaggle-gpu" ] && echo held || echo free)" free "quota below the billed cap -> lock released"
check "$(wc -c < "$T/RESOURCES.md" | tr -d ' ')" 0 "quota below the billed cap -> RESOURCES.md untouched"
check "$(wc -l < "$T/shapes.txt" | tr -d ' ')" 0 "quota below the billed cap -> no push attempted"
# 3. DRY run with enough quota: exit 0, lock released
fake 29.50h "$ALL"
DRY=1 "$S" kern ref/x "$T/out" 10800 90 "note" > "$T/o3.txt" 2>&1; check $? 0 "DRY -> exit 0"
check "$(grep -c 'DRY: gates passed' "$T/o3.txt")" 1 "DRY prints the push line"
check "$([ -d "$T/locks/kaggle-gpu" ] && echo held || echo free)" free "DRY -> lock released"
# 4. every shape refused: exit 4, lock released, both shapes tried in order, RESOURCES.md untouched
fake 29.50h "none"
GPU_SHAPES="$ALL" "$S" kern ref/x "$T/out" 10800 90 "note" > "$T/o4.txt" 2>&1; check $? 4 "all shapes refused -> exit 4"
check "$(tr '\n' ' ' < "$T/shapes.txt")" "NvidiaL4X1 NvidiaTeslaT4 " "all shapes refused -> L4 tried first, then T4"
check "$([ -d "$T/locks/kaggle-gpu" ] && echo held || echo free)" free "all shapes refused -> lock released"
check "$(wc -c < "$T/RESOURCES.md" | tr -d ' ')" 0 "all shapes refused -> RESOURCES.md untouched"
# 5. L4 refused, T4 accepted: falls back, row names the T4 shape
fake 29.50h "NvidiaTeslaT4"
GPU_SHAPES="$ALL" "$S" kern ref/bench-gauges-gpu-sweep1 "$T/out" 10800 90 "note" > "$T/o5.txt" 2>&1; check $? 0 "fallback -> exit 0"
check "$(tr '\n' ' ' < "$T/shapes.txt")" "NvidiaL4X1 NvidiaTeslaT4 " "fallback -> both shapes tried in order"
check "$(grep -c 'shape requested NvidiaTeslaT4' "$T/RESOURCES.md")" 1 "fallback -> RESOURCES row names the T4 shape"
check "$([ -d "$T/locks/kaggle-gpu" ] && echo held || echo free)" free "fallback -> lock released after COMPLETE"
# 6. single-GPU shape accepted first: one push only
: > "$T/RESOURCES.md"
fake 29.50h "$ALL"
GPU_SHAPES="$ALL" "$S" kern ref/bench-gauges-gpu-sweep1 "$T/out" 10800 90 "note" > "$T/o6.txt" 2>&1; check $? 0 "first shape accepted -> exit 0"
check "$(tr '\n' ' ' < "$T/shapes.txt")" "NvidiaL4X1 " "first shape accepted -> exactly one push"
check "$(grep -c 'shape requested NvidiaL4X1' "$T/RESOURCES.md")" 1 "first shape accepted -> one RESOURCES row naming it"
# 7. GPU_SHAPES override
: > "$T/RESOURCES.md"
fake 29.50h "NvidiaTeslaT4"
GPU_SHAPES="NvidiaTeslaT4" "$S" kern ref/bench-gauges-gpu-sweep1 "$T/out" 10800 90 "note" > "$T/o7.txt" 2>&1; check $? 0 "GPU_SHAPES override -> exit 0"
check "$(tr '\n' ' ' < "$T/shapes.txt")" "NvidiaTeslaT4 " "GPU_SHAPES override -> only the T4 shape requested"
# 8. -t: T4 shape (the default) -> cap / 2 + 600 rounded up to 300 s; a single-GPU shape -> the cap; GPU_T_S overrides
: > "$T/RESOURCES.md"
fake 29.50h "NvidiaTeslaT4"
"$S" kern ref/bench-gauges-gpu-sweep1 "$T/out" 10800 90 "note" > "$T/o8.txt" 2>&1; check $? 0 "default shape -> exit 0"
check "$(tr '\n' ' ' < "$T/shapes.txt")" "NvidiaTeslaT4 " "default shape is the T4 shape only"
check "$(tr '\n' ' ' < "$T/tvals.txt")" "6000 " "T4 default -t = cap / 2 + 600 (10,800 -> 6,000)"
: > "$T/RESOURCES.md"
fake 29.50h "NvidiaL4X1"
GPU_SHAPES="NvidiaL4X1" "$S" kern ref/bench-gauges-gpu-sweep1 "$T/out" 10800 90 "note" > "$T/o8b.txt" 2>&1; check $? 0 "single-GPU shape -> exit 0"
check "$(tr '\n' ' ' < "$T/tvals.txt")" "10800 " "single-GPU shape -> -t equals the cap"
: > "$T/RESOURCES.md"
fake 29.50h "NvidiaTeslaT4"
GPU_T_S=5555 "$S" kern ref/bench-gauges-gpu-sweep1 "$T/out" 10800 90 "note" > "$T/o9.txt" 2>&1; check $? 0 "GPU_T_S -> exit 0"
check "$(tr '\n' ' ' < "$T/tvals.txt")" "5555 " "GPU_T_S overrides the push -t value"
check "$(grep -c -- '-t 5555 wall s, billed cap 10800 s' "$T/RESOURCES.md")" 1 "RESOURCES row records both the wall -t and the billed cap"
# 9. queue wrapper: retries while the lock is held (exit 3), pushes as soon as it is free, gives up after QUEUE_MAX_MIN
: > "$T/RESOURCES.md"
fake 29.50h "NvidiaTeslaT4"
mkdir "$T/locks/kaggle-gpu"; echo "small-lm something" > "$T/locks/kaggle-gpu/owner"
QUEUE_POLL_S=1 QUEUE_MAX_MIN=0 "$HERE/tools/gpu_queue_push.sh" kern ref/x "$T/out" 10800 90 "note" > "$T/o10.txt" 2>&1; check $? 3 "queue wrapper gives up with exit 3 after QUEUE_MAX_MIN"
check "$(wc -l < "$T/shapes.txt" | tr -d ' ')" 0 "queue wrapper: nothing pushed while the lock is held"
( sleep 2; rm -rf "$T/locks/kaggle-gpu" ) &
QUEUE_POLL_S=1 QUEUE_MAX_MIN=1 "$HERE/tools/gpu_queue_push.sh" kern ref/bench-gauges-gpu-sweep1 "$T/out" 10800 90 "note" > "$T/o11.txt" 2>&1; check $? 0 "queue wrapper pushes once the lock is free"
wait
check "$(tr '\n' ' ' < "$T/shapes.txt")" "NvidiaTeslaT4 " "queue wrapper: exactly one push after the wait"
check "$([ -d "$T/locks/kaggle-gpu" ] && echo held || echo free)" free "queue wrapper: lock released after COMPLETE"
# 10. the account's concurrent GPU session limit: exit 5, lock released, nothing logged; the queue wrapper retries and pushes once a slot is free
: > "$T/RESOURCES.md"
fake 29.50h "NvidiaTeslaT4"; touch "$T/slots_full"
"$S" kern ref/x "$T/out" 10800 90 "note" > "$T/o12.txt" 2>&1; check $? 5 "session limit -> exit 5"
check "$([ -d "$T/locks/kaggle-gpu" ] && echo held || echo free)" free "session limit -> lock released"
check "$(wc -c < "$T/RESOURCES.md" | tr -d ' ')" 0 "session limit -> RESOURCES.md untouched"
QUEUE_SLOT_POLL_S=1 QUEUE_MAX_MIN=0 "$HERE/tools/gpu_queue_push.sh" kern ref/x "$T/out" 10800 90 "note" > "$T/o13.txt" 2>&1; check $? 5 "queue wrapper: gives up with exit 5 when the limit persists"
( sleep 2; rm -f "$T/slots_full" ) &
QUEUE_SLOT_POLL_S=1 QUEUE_MAX_MIN=1 "$HERE/tools/gpu_queue_push.sh" kern ref/bench-gauges-gpu-sweep1 "$T/out" 10800 90 "note" > "$T/o14.txt" 2>&1; check $? 0 "queue wrapper: pushes once a slot is free"
wait
check "$(grep -c 'Kaggle GPU kernel ref/bench-gauges-gpu-sweep1' "$T/RESOURCES.md")" 1 "queue wrapper: exactly one RESOURCES row after the retry"
# 11. tiered push (amendment 7): the biggest tier whose billed cap the remaining quota covers is pushed; slots full -> wait; nothing fits -> exit 6
TIERS="big:17400:122 mid:13500:94 small:9300:66"
for q in "29.50h big" "4.00h mid" "2.70h small"; do
  set -- $q
  : > "$T/RESOURCES.md"; fake $1 "NvidiaTeslaT4"; rm -f "$T/slots_full"
  "$HERE/tools/gpu_tiered_push.sh" "$T/out" ref/bench-gauges-gpu-final "note" big:17400:122 mid:13500:94 small:9300:66 > "$T/o15.txt" 2>&1; check $? 0 "tiered push with $1 remaining -> exit 0"
  check "$(tail -n 1 "$T/dirs.txt")" "$2" "tiered push with $1 remaining pushes the $2 tier"
  check "$(wc -l < "$T/dirs.txt" | tr -d ' ')" "1" "tiered push with $1 remaining: exactly one push"
done
: > "$T/RESOURCES.md"; fake 1.00h "NvidiaTeslaT4"
"$HERE/tools/gpu_tiered_push.sh" "$T/out" ref/x "note" big:17400:122 mid:13500:94 small:9300:66 > "$T/o16.txt" 2>&1; check $? 6 "tiered push: nothing fits the remaining quota -> exit 6"
check "$(wc -l < "$T/dirs.txt" | tr -d ' ')" 0 "tiered push: nothing fits -> no push attempted"
fake 29.50h "NvidiaTeslaT4"; touch "$T/slots_full"
( sleep 2; rm -f "$T/slots_full" ) &
QUEUE_SLOT_POLL_S=1 QUEUE_MAX_MIN=1 "$HERE/tools/gpu_tiered_push.sh" "$T/out" ref/bench-gauges-gpu-final "note" big:17400:122 mid:13500:94 small:9300:66 > "$T/o17.txt" 2>&1; check $? 0 "tiered push waits for a free session slot, then pushes"
wait
check "$(tail -n 1 "$T/dirs.txt")" "big" "tiered push after the wait pushes the big tier"
rm -rf "$T"
[ $fail = 0 ] && echo "ALL OK" || { echo "FAILED"; exit 1; }
