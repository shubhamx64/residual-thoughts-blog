#!/bin/bash
# Auto-stop the pod when all 4 lanes finish (or a 7h safety cap). Tars results to the
# persistent /workspace volume first, then a 15-min grace (so results can be pulled),
# then runpodctl stop. Runs in its own tmux session.
POD=YOUR_POD_ID
START=$(date +%s); MAXSEC=25200
log(){ echo "[$(date -u +%H:%M:%S)] $*" >> /workspace/autostop.log; }
log "watcher started (pod $POD, cap ${MAXSEC}s)"
while true; do
  d=0
  { grep -q QWEN_SEEDS_DONE /workspace/lane1.log 2>/dev/null && grep -q QWEN_SEEDS_DONE /workspace/lane1b.log 2>/dev/null; } && d=$((d+1))
  grep -q SCALING_DONE    /workspace/lane2.log 2>/dev/null && d=$((d+1))
  grep -q "E4SCALE 3b DONE" /workspace/lane3.log 2>/dev/null && d=$((d+1))
  grep -q "E4SCALE 7b DONE" /workspace/lane4.log 2>/dev/null && d=$((d+1))
  el=$(( $(date +%s) - START ))
  if [ "$d" -ge 4 ] || [ "$el" -gt "$MAXSEC" ]; then
    log "trigger: lanes_done=$d elapsed=${el}s"
    cd /workspace/s2 && tar czf /workspace/results.tgz \
      e4-continual/results/log_B_*.jsonl escale/results/align_*.json 2>/dev/null
    log "results tarred -> /workspace/results.tgz ; 15min grace before stop"
    sleep 900
    log "issuing runpodctl stop pod $POD"
    runpodctl stop pod "$POD" >> /workspace/autostop.log 2>&1
    log "stop issued"
    break
  fi
  sleep 120
done
