#!/bin/bash
# Overnight RoboTTT serve-ablation bundle (steelman ladder A0+A1+A2+A3+A4+A5 + box control).
# 4 GPUs, arms serial per GPU. Reference rows: sealed p5 per-task Q0/Q2 numbers + q2ref box control.
# TOP8 tasks = largest Q0−Q2 deltas; TOP4 = first four.
set -uxo pipefail
RUN=/data/ablate_robottt_nagababa.sh
Q2=/data/work/ckpt/q2-59999
TOP8=SteamInMicrowave,TurnOnElectricKettle,WashLettuce,StoreLeftoversInBowl,PreSoakPan,HeatKebabSandwich,NavigateKitchen,CoffeeSetupMug
TOP4=SteamInMicrowave,TurnOnElectricKettle,WashLettuce,StoreLeftoversInBowl
LOG=/data2/evals/overnight_$(date -u +%m%d_%H%M).log

run() { ARM="$1" CKPT="$2" TASKS="$3" GPU="$4" ABLATION="$5" bash "$RUN" >>"$LOG" 2>&1 || echo "ARM $1 FAILED" >>"$LOG"; }

# GPU0: A1 causality probe — freeze W all episode, full TOP8 signal
( run a1_freeze      "$Q2" "$TOP8" 0 "freeze" ) &
# GPU1: A2 reset at trained chain length, then A4 eta scale-down
( run a2_reset7      "$Q2" "$TOP4" 1 "reset:7"
  run a4_eta03       "$Q2" "$TOP4" 1 "eta:0.3" ) &
# GPU2: A3 decay toward W0, two gammas
( run a3_decay08     "$Q2" "$TOP4" 2 "decay:0.8"
  run a3_decay05     "$Q2" "$TOP4" 2 "decay:0.5" ) &
# GPU3: box control (unmodified Q2 — validates transfer for the robottt server), then eta 0.1
# (a5 commitfirst dropped: verified no-op — serve already commits before the next condition)
( run q2ref          "$Q2" "$TOP4" 3 ""
  run a4_eta01       "$Q2" "$TOP4" 3 "eta:0.1" ) &
wait
echo "OVERNIGHT BUNDLE COMPLETE" >>"$LOG"
echo OVERNIGHT_DONE
