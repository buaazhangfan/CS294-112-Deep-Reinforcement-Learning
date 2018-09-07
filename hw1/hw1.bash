#!/bin/bash
set -eux
if [ ! -d "expert_data" ]; then
  mkdir expert_data
fi

python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --num_rollouts 20
python run_expert.py experts/Reacher-v2.pkl Reacher-v2 --num_rollouts 400
python behavior_cloning.py experts/Hopper-v2.pkl Hopper-v2 --num_rollouts 20
python behavior_cloning.py experts/Reacher-v2.pkl Reacher-v2 --num_rollouts 400
python DAgger.py experts/Hopper-v2.pkl Hopper-v2 --num_rollouts 20
python plot.py Hopper-v2 --num_rollouts 20