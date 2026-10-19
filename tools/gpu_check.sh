#!/bin/bash
# One-shot status of a running GPU kernel: kernel status, the filtered tail of its log (hardware line, device lines, pre-flight, epochs, done / skipped / errors) and the GPU quota row.
# usage: tools/gpu_check.sh KERNEL_REF [LINES=24]     (read-only; the log pattern and the unescaping are the ones of the STATUS.md runbook, checked on a finished kernel)
REF=$1; N=${2:-24}
echo "$(date -u +%FT%TZ) $(kaggle kernels status "$REF" 2>&1 | tail -1)"
kaggle kernels logs "$REF" 2>&1 | grep -E "WRAPPER|DEVICE|PREFLIGHT|missing|SKIPPED|CONFIG DONE|HARD LIMIT|Traceback|Error|epoch" | sed 's/\\"/"/g' | grep -v '"config"' | cut -c1-230 | tail -"$N"
kaggle quota 2>&1 | sed -n 3p
