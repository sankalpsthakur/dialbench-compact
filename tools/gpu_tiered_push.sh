#!/bin/bash
# Quota-aware, queue-aware GPU push for ONE kernel ref with several sized variants (FROZEN-4 amendment 7): try the biggest tier first; if the account's remaining GPU quota
# is below its billed cap (gpu_push_wait.sh exit 6) try the next smaller tier; if the lock is held (3) or the account's concurrent GPU session limit is reached (5), wait and
# start again from the biggest tier; any other outcome ends the loop with that exit code. Run it in the background (one waiter per kernel).
# usage: tools/gpu_tiered_push.sh OUT_DIR KERNEL_REF "note" TIER [TIER ...]      TIER = KERNEL_DIR:CAP_BILLED_S:EXPECTED_WALL_MINUTES, biggest first
# env: QUEUE_POLL_S (default 30, lock held), QUEUE_SLOT_POLL_S (default 120, session limit), QUEUE_MAX_MIN (default 720), plus those of gpu_push_wait.sh
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1; REF=$2; NOTE=$3; shift 3
start=$SECONDS
while true; do
  stop=0; rc=0
  for tier in "$@"; do
    IFS=: read -r dir cap expm <<< "$tier"
    "$HERE/gpu_push_wait.sh" "$dir" "$REF" "$OUT" "$cap" "$expm" "$NOTE [tier $dir, billed cap $cap s]"; rc=$?
    case "$rc" in
      6) echo "$(date -u +%H:%M:%S) tier $dir: remaining quota below its billed cap of $cap s, trying the next smaller tier"; rc=6; continue;;
      3|5) stop=1; break;;
      *) exit "$rc";;
    esac
  done
  if [ "$stop" = 0 ]; then echo "remaining quota is below the billed cap of every tier, nothing pushed"; exit 6; fi
  if [ $(( (SECONDS - start) / 60 )) -ge "${QUEUE_MAX_MIN:-720}" ]; then echo "queue wait over after ${QUEUE_MAX_MIN:-720} min, nothing pushed"; exit "$rc"; fi
  echo "$(date -u +%H:%M:%S) attempt ended with code $rc (3 = lock held, 5 = session limit), retrying"
  if [ "$rc" -eq 5 ]; then sleep "${QUEUE_SLOT_POLL_S:-120}"; else sleep "${QUEUE_POLL_S:-30}"; fi
done
