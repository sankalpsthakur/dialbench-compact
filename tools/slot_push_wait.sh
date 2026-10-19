#!/bin/bash
# Wait for a free program Kaggle CPU slot (locks/kaggle-cpu-1..4), take it, push a private kernel, log it in RESOURCES.md (append only),
# then poll every 5 minutes, fetch the matching small outputs when it ends and release the slot (via tools/wait_kernel.sh).
# usage: tools/slot_push_wait.sh KERNEL_DIR KERNEL_REF OUT_DIR FILE_REGEX KERNEL_TIMEOUT_S EXPECTED_MINUTES "note for RESOURCES.md" [MAX_SLOT_WAIT_MIN=180]
set -u
KDIR=$1; REF=$2; OUT=$3; PAT=$4; KTO=$5; EXPM=$6; NOTE=$7; MAXW=${8:-180}
BM=${BM:?set BM to the benchmarks root that holds locks/ and RESOURCES.md}
start=$SECONDS; got=""
while true; do
  for n in 1 2 3 4; do
    if mkdir "$BM/locks/kaggle-cpu-$n" 2>/dev/null; then got="$BM/locks/kaggle-cpu-$n"; break; fi
  done
  [ -n "$got" ] && break
  if [ $(( (SECONDS - start) / 60 )) -ge "$MAXW" ]; then echo "$(date -u +%H:%M:%S) no free slot after $MAXW min, nothing pushed"; exit 3; fi
  sleep 5
done
now=$(date -u +%FT%TZ); endt=$(date -u -v+"$EXPM"M +%FT%TZ)
echo "gauges $REF $now running, expected end about $endt" > "$got/owner"
echo "$now took $got"
out=$(kaggle kernels push -p "$KDIR" -t "$KTO" 2>&1); echo "$out" | tail -3
if ! echo "$out" | grep -q "successfully pushed"; then echo "push failed, releasing $got"; rm -rf "$got"; exit 4; fi
echo "| $(date -u '+%Y-%m-%d %H:%M') | gauges | Kaggle CPU kernel $REF (slot $(basename "$got"); $NOTE) | ~$EXPM min CPU | running |" >> "$BM/RESOURCES.md"
exec "$BM/gauges/tools/wait_kernel.sh" "$REF" "$OUT" "$got" "$PAT" $(( EXPM * 3 )) 300
