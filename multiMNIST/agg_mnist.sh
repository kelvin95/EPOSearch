#!/usr/bin/env sh
fd "${1:-lenet}" -t d -d 1 . runs -x poetry \
  run python aggregator.py --path {} \
  --subpaths '[".", "tasks_0_1_cosine_angle", "tasks_0_1_grad_magnitude_sim"]' \
  --output csv
