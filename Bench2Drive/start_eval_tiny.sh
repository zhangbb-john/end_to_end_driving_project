#!/bin/bash

# 设置环境变量
export CARLA_ROOT="/home/slxy/zca/code/drivetransformer_private/carla"
export CARLA_SERVER="/home/slxy/zca/code/drivetransformer_private/carla/CarlaUE4.sh"
export SCENARIO_RUNNER_ROOT="/home/slxy/zca/code/BenchDrive/scenario_runner"
export PYTHONPATH="/home/slxy/zca/code/drivetransformer_private/carla/PythonAPI:/home/slxy/zca/code/drivetransformer_private/carla/PythonAPI/carla:/home/slxy/zca/code/drivetransformer_private/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:/home/slxy/zca/code/BenchDrive:/home/slxy/zca/code/BenchDrive/leaderboard:/home/slxy/zca/code/BenchDrive/leaderboard/team_code:/home/slxy/zca/code/BenchDrive:/home/slxy/zca/code/BenchDrive/scenario_runner"

# Python路径
PYTHON="/home/slxy/.miniconda3/envs/drivetransformer/bin/python"
PROGRAM="leaderboard/leaderboard/leaderboard_evaluator.py"

cd /home/slxy/zca/code/Bench2Drive

# 执行命令
$PYTHON $PROGRAM \
    --routes=leaderboard/data/drivetransformer_bench2drive_dev4.xml \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint=DriveTransformer_b2d_only_traj/eval_bench2drive_dev_4.json \
    --agent=leaderboard/team_code/drivetransformer_vis_agent.py \
    --agent-config=DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_tiny.py+DriveTransformer/ckpts/iter_93750.pth \
    --debug=0 \
    --record="" \
    --port=30001 \
    --traffic-manager-port=50000 \
    --gpu-rank=0
