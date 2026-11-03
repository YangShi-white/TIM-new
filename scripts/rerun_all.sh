#!/usr/bin/env bash
# Re-run every experiment after the review fixes (double-use row exclusion,
# CV+ inflation, tuned baselines, survivorship-corrected budget, control in
# distortion analysis). Serial on purpose: parallel runs starve each other.
set -e
cd "$(dirname "$0")/.."

run() {   # name, script
  echo "=============================================================="
  echo "=== $1"
  echo "=============================================================="
  python -I -u "$2" > "data/log_$1.txt" 2>&1
  tail -6 "data/log_$1.txt"
}

run loio            experiments/h_transfer/run_loio.py
run distortion      experiments/h_transfer/run_distortion_structure.py
run headtohead      experiments/h_transfer/run_headtohead.py
run std_size        experiments/h_transfer/run_std_size.py
run ablation        experiments/h_transfer/run_ablation.py
run selection       experiments/h_transfer/run_selection_validity.py
run multisource     experiments/h_transfer/run_multisource.py
run conformal       experiments/h_uncertainty/run_conformal.py
run selfassessment  experiments/h_uncertainty/run_selfassessment.py
run adaptive        experiments/h_uncertainty/run_adaptive.py
run broth           experiments/h_broth/run_broth_validation.py
run batch_holdout   experiments/h_broth/run_batch_holdout.py
run significance    experiments/h_transfer/run_significance.py

echo "ALL EXPERIMENTS RE-RUN"
