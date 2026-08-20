#!/usr/bin/env bash
# Full experiment grid. ~40 min on a Colab T4.
set -e
cd "$(dirname "$0")/src"
for seed in 0 1 2; do
  python train.py --model scratch  --seed $seed --epochs 20
  python train.py --model scratch  --seed $seed --epochs 20 --augment
  python train.py --model resnet18 --seed $seed --epochs 10 --augment
done
python report.py
