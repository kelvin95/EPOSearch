#!/usr/bin/env sh
for dset in mnist fashion fashion_and_mnist; do
  for id in 0 1; do
    dir="individual_"$dset"_lenet_100-t"$id""
    rsync -avz --progress -h \
      vws51:/h/amanjitsk/Desktop/EPOSearch/multiMNIST/runs/"$dir"/1 \
      indiv/"$dir"
  done
done
