#!/bin/bash
# Take the program GPU lock, gate on the Kaggle quota IN BILLED HOURS, push ONE private GPU kernel (single-GPU shape first, the documented T4 shape if that push is refused),
# log it in RESOURCES.md (append only), then poll and fetch the small outputs and release the lock via tools/wait_kernel.sh.
# Nothing is pushed unless the lock was taken AND the remaining quota covers the billed cap; a failed gate or push releases the lock.
# usage: tools/gpu_push_wait.sh KERNEL_DIR KERNEL_REF OUT_DIR CAP_BILLED_S EXPECTED_WALL_MINUTES "note for RESOURCES.md" [FILE_REGEX]
# Decision of the coordinator (2026-10-10): T4 shape only (the SDK does not know the L4 shape and the request was silently replaced by 2 x T4), push -t = cap / 2 + 600.
#   CAP_BILLED_S = billed GPU-seconds the kernel may use (the quota is billed per GPU held: a session on 2 T4 bills 2 s per wall second). It is the push -t value (the wall
#   limit for a single GPU); the kernel's wrapper stops its workers at (cap - 600) / GPUs held, so a 2-GPU session ends in half the wall time.
#   Caution: `-t` is a WALL limit. On 2 GPUs (what Kaggle gave for the single-L4 request too) a `-t` equal to the billed cap is a backstop of 2 x the cap; only the
#   wrapper's own hard limit keeps the session at the cap. For the T4 shape the default -t is therefore cap / 2 + 600 rounded up to 300 s (Stage A: 6000), so the
#   backstop is about the billed cap; for a single-GPU shape (L4, P100) it is the cap itself.
# exit codes: 3 = GPU lock held by someone else (nothing done), 4 = push failed (lock released), 6 = the remaining quota is below the billed cap (lock released), 5 = the account's concurrent GPU session limit is reached
#   ("Maximum batch GPU session count of 2 reached": the CLI exits 0 on that error, so the output is read; lock released, retry later), 0 = ended (or DRY run).
# env: BM = benchmarks root (required in this export; the author's default path is redacted); GPU_T_S = the push -t value in wall seconds (default: see above); GPU_SHAPES = machine shapes to
#   request, in order (default "NvidiaTeslaT4"; Kaggle accepts an unknown shape silently and still gives 2 x T4, so read the WRAPPER gpus line of the log after the start);
#   DRY=1 = take the lock, run the gates, print the push line, release.
# Run it in the background (it waits for the kernel). Never create logs/chain/*.next for GPU kernels.
set -u
KDIR=$1; REF=$2; OUT=$3; CAP=$4; EXPM=$5; NOTE=$6; PAT=${7:-'.*\.(json|log|npz)$'}
BM=${BM:?set BM to the benchmarks root that holds locks/ and RESOURCES.md}
SHAPES=${GPU_SHAPES:-"NvidiaTeslaT4"}
LOCK="$BM/locks/kaggle-gpu"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "GPU lock held by: $(cat "$LOCK/owner" 2>/dev/null); nothing pushed"; exit 3
fi
now=$(date -u +%FT%TZ)
echo "gauges $REF $now gate check, billed cap $CAP s" > "$LOCK/owner"
mkdir -p "$BM/gauges/logs/local"
q=$(kaggle quota 2>&1); echo "$q"
echo "$q" > "$BM/gauges/logs/local/quota_before_$(basename "$KDIR").txt"
rem=$(echo "$q" | awk '$1=="GPU"{gsub("h","",$3); print $3}')
need=$(awk -v t="$CAP" 'BEGIN{printf "%.2f", t/3600}')
if [ -z "$rem" ] || ! awk -v r="$rem" -v n="$need" 'BEGIN{exit !(r+0 >= n+0)}'; then
  echo "quota gate: remaining GPU hours '${rem:-unreadable}' do not cover the billed cap of $need h; nothing pushed, lock released"
  rm -rf "$LOCK"; exit 6
fi
if [ "${DRY:-}" = "1" ]; then
  case "$SHAPES" in *L4*|*P100*) DEF_T=$CAP;; *) DEF_T=$(( (CAP / 2 + 600 + 299) / 300 * 300 ));; esac
  echo "DRY: gates passed; would try shapes [$SHAPES]: kaggle kernels push -p $KDIR -t ${GPU_T_S:-$DEF_T} --accelerator <shape>"; rm -rf "$LOCK"; exit 0
fi
case "$SHAPES" in *L4*|*P100*) DEF_T=$CAP;; *) DEF_T=$(( (CAP / 2 + 600 + 299) / 300 * 300 ));; esac
TVAL=${GPU_T_S:-$DEF_T}
used=""; full=0
for shape in $SHAPES; do
  out=$(kaggle kernels push -p "$KDIR" -t "$TVAL" --accelerator "$shape" 2>&1); echo "[$shape] $(echo "$out" | tail -2)"
  if echo "$out" | grep -q "successfully pushed"; then used=$shape; break; fi
  if echo "$out" | grep -q "Maximum batch GPU session count"; then full=1; break; fi
done
if [ -z "$used" ] && [ "$full" = 1 ]; then
  echo "GPU session slots full (the account allows 2 concurrent batch GPU sessions); nothing pushed, lock released, retry later"; rm -rf "$LOCK"; exit 5
fi
if [ -z "$used" ]; then
  echo "push refused for every shape [$SHAPES], lock released"; rm -rf "$LOCK"; exit 4
fi
endt=$(date -u -v+"$EXPM"M +%FT%TZ)
echo "gauges $REF $now running, shape requested $used, expected end about $endt, billed cap $CAP s" > "$LOCK/owner"
echo "| $(date -u '+%Y-%m-%d %H:%M') | gauges | Kaggle GPU kernel $REF ($NOTE; shape requested $used, -t $TVAL wall s, billed cap $CAP s; the quota bills per GPU held: read the WRAPPER gpus line and compare kaggle quota with wall time) | ~$EXPM min wall, billed cap $need GPU-h | running |" >> "$BM/RESOURCES.md"
echo "pushed with shape $used. In the first 10 minutes read the WRAPPER gpus / DEVICE lines of the log and compare kaggle quota with wall time (FROZEN-4 amendment 5)."
exec "$BM/gauges/tools/wait_kernel.sh" "$REF" "$OUT" "$LOCK" "$PAT" $(( TVAL / 60 + 10 )) 300
