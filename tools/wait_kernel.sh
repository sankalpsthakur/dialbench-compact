#!/bin/bash
# Poll a Kaggle kernel every 5 minutes; when it ends (any terminal state) fetch the outputs whose names match FILE_REGEX into OUT_DIR,
# release the program lock LOCK_DIR (only if its `owner` file is ours) and exit. Run it with run_in_background.
# usage: tools/wait_kernel.sh KERNEL_REF OUT_DIR LOCK_DIR FILE_REGEX [MAX_MINUTES=240] [POLL_SECONDS=300]
# Chain (optional): if logs/chain/<kernel-name>.next exists (shell assignments NEXT_DIR NEXT_REF NEXT_OUT NEXT_PAT NEXT_KTO NEXT_EXPM NEXT_NOTE and optionally
# NEXT_REQUIRE_FILE) and this kernel ended COMPLETE with that file among its outputs, the next kernel is pushed INTO THE SAME SLOT (no gap in which another
# track could take it), watched the same way, and only then is the slot released.
REF=$1; OUT=$2; LOCK=$3; PAT=$4; MAXM=${5:-240}; POLL=${6:-300}
BM=${BM:?set BM to the benchmarks root that holds locks/ and RESOURCES.md}
CHAIN_DIR=$BM/gauges/logs/chain
ST=""

poll() {   # ref out pattern max_minutes poll_seconds -> sets ST; returns 2 if it gave up
  local ref=$1 out=$2 pat=$3 maxm=$4 pol=$5 start=$SECONDS st
  mkdir -p "$out"
  while true; do
    st=$(kaggle kernels status "$ref" 2>&1 | tail -1)
    echo "$(date -u +%H:%M:%S) $st"
    case "$st" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac
    if [ $(( (SECONDS - start) / 60 )) -ge "$maxm" ]; then echo "wait_kernel: gave up on $ref after $maxm min, lock kept"; ST="GAVEUP"; return 2; fi
    sleep "$pol"
  done
  kaggle kernels output "$ref" -p "$out" --file-pattern "$pat" 2>&1 | tail -20
  echo "$(date -u +%FT%TZ) $st" > "$out/_wait_done.txt"
  ST="$st"
  return 0
}

poll "$REF" "$OUT" "$PAT" "$MAXM" "$POLL" || exit 2
FIRST_ST="$ST"
NEXT="$CHAIN_DIR/${REF##*/}.next"
if [ -f "$NEXT" ] && echo "$FIRST_ST" | grep -q COMPLETE; then
  (
    source "$NEXT"
    if [ -n "${NEXT_REQUIRE_FILE:-}" ] && ! kaggle kernels files "$REF" --page-size 200 -v 2>&1 | grep -q "$NEXT_REQUIRE_FILE"; then
      echo "chain: $NEXT_REQUIRE_FILE is not among the outputs of $REF, not pushing $NEXT_REF"; exit 0
    fi
    echo "chain: pushing $NEXT_REF into $LOCK"
    out=$(kaggle kernels push -p "$NEXT_DIR" -t "$NEXT_KTO" 2>&1); echo "$out" | tail -3
    if echo "$out" | grep -q "successfully pushed"; then
      now=$(date -u +%FT%TZ); endt=$(date -u -v+"$NEXT_EXPM"M +%FT%TZ)
      echo "gauges $NEXT_REF $now running, expected end about $endt" > "$LOCK/owner"
      echo "| $(date -u '+%Y-%m-%d %H:%M') | gauges | Kaggle CPU kernel $NEXT_REF (slot $(basename "$LOCK"), handed over from $REF without releasing it; $NEXT_NOTE) | ~$NEXT_EXPM min CPU | running |" >> "$BM/RESOURCES.md"
      poll "$NEXT_REF" "$NEXT_OUT" "$NEXT_PAT" $(( NEXT_EXPM * 3 )) 300
    else
      echo "chain: push of $NEXT_REF failed, releasing the slot"
    fi
  )
fi
if [ -d "$LOCK" ] && grep -q gauges "$LOCK/owner" 2>/dev/null; then rm -rf "$LOCK" && echo "lock $LOCK released"; else echo "lock $LOCK not ours or absent, left alone"; fi
echo "wait_kernel: done: $FIRST_ST"
