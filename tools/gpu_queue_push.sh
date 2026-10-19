#!/bin/bash
# Retry tools/gpu_push_wait.sh while the shared GPU lock is held by another track (its exit code 3) or while the account's concurrent GPU session limit is reached
# (exit code 5; the lock is released between attempts so that the other tracks can try too); any other outcome ends the loop with that exit code.
# usage: tools/gpu_queue_push.sh <the same arguments as tools/gpu_push_wait.sh>
#   env: QUEUE_POLL_S (default 30, lock held), QUEUE_SLOT_POLL_S (default 120, session limit reached), QUEUE_MAX_MIN (default 600), plus those of gpu_push_wait.sh
# The lock is only ever taken by gpu_push_wait.sh (atomic mkdir after the polling sleep), so this never steals it from another track's runner mid-hand-over.
# Run it in the background (one waiter per kernel); it prints one line per refused attempt.
HERE=$(cd "$(dirname "$0")" && pwd)
start=$SECONDS
while true; do
  "$HERE/gpu_push_wait.sh" "$@"; rc=$?
  { [ "$rc" -ne 3 ] && [ "$rc" -ne 5 ]; } && exit "$rc"
  if [ $(( (SECONDS - start) / 60 )) -ge "${QUEUE_MAX_MIN:-600}" ]; then echo "queue wait over after ${QUEUE_MAX_MIN:-600} min, nothing pushed"; exit "$rc"; fi
  echo "$(date -u +%H:%M:%S) attempt ended with code $rc (3 = lock held, 5 = session limit), retrying"
  if [ "$rc" -eq 5 ]; then sleep "${QUEUE_SLOT_POLL_S:-120}"; else sleep "${QUEUE_POLL_S:-30}"; fi
done
