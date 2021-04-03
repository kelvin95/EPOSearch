#!/usr/bin/env bash

DSET="${1:-all}"
SOLV="${2:-itmtl}"
ARCH="${3:-lenet}"
# number of random seeds
NUM_RUNS="${4:-5}"

for _ in $(seq $NUM_RUNS); do
  poetry run python train.py \
    --dset="$DSET" --arch="$ARCH" --seed="$RANDOM" --solver="$SOLV"
done
