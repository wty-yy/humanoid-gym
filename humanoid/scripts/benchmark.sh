#!/bin/bash

SEED=(42 0 1 2)
 
for seed in "${SEED[@]}"; do
  # python humanoid/scripts/play.py --task=kuavo42_legged_single_obs_ppo --run-name v1 --command benchmark --cycle-time 0.64 --load-onnx models/kuavo42_legged/Kuavo42_legged_single_obs_ppo_v1_model_3001.onnx --seed $seed --headless
  # python humanoid/scripts/play.py --task=kuavo42_legged_single_obs_ppo --run-name v1.1 --command benchmark --cycle-time 0.64 --seed $seed --headless
  # python humanoid/scripts/play.py --task=kuavo42_legged_single_obs_ppo --run-name v1.2 --command benchmark --cycle-time 0.64 --seed $seed --headless
  python humanoid/scripts/play.py --task=kuavo42_legged_fine_ppo --run-name v1 --command benchmark --cycle-time 1.2 --load-onnx models/kuavo42_legged/Kuavo42_legged_fine_ppo_v1_model_3001.onnx --seed $seed --headless
  # python humanoid/scripts/play.py --task=g1_ppo --run-name v1.1 --command benchmark --cycle-time 0.64 --seed $seed
done