export SEED=2
python humanoid/scripts/play.py --task=kuavo42_legged_single_obs_ppo --run-name v1 --command benchmark --cycle-time 0.64 --load-onnx models/kuavo42_legged/Kuavo42_legged_single_obs_ppo_v1_model_3001.onnx --seed $SEED --headless
python humanoid/scripts/play.py --task=kuavo42_legged_single_obs_ppo --run-name v1.1 --command benchmark --cycle-time 0.64 --seed $SEED --headless
python humanoid/scripts/play.py --task=kuavo42_legged_single_obs_ppo --run-name v1.2 --command benchmark --cycle-time 0.64 --seed $SEED --headless
python humanoid/scripts/play.py --task=g1_ppo --run-name v1 --command benchmark --cycle-time 0.64 --seed $SEED --headless