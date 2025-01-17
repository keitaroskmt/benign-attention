#!/bin/bash

for noise_ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
  src_directory="results/bert/agnews/noise_ratio${noise_ratio}/pretrain/"
  target_directory="results/bert/agnews/noise_ratio_${noise_ratio}/pretrain/0/"

  mkdir -p ${target_directory}
  mv ${src_directory}* ${target_directory}
done
