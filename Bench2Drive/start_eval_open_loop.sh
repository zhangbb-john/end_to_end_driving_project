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

# 创建输出目录
mkdir -p DriveTransformer_b2d_open_loop

echo "========================================"
echo "开环仿真 - 使用GT航点"
echo "Open-loop Simulation with GT Waypoints"
echo "========================================"
echo ""
echo "按 Ctrl+C 停止仿真"
echo "========================================"

# 执行命令
$PYTHON $PROGRAM \
    --routes=leaderboard/data/drivetransformer_bench2drive_dev10_open_loop.xml \
    --repetitions=1 \
    --track=SENSORS \
    --checkpoint=DriveTransformer_b2d_open_loop/eval_bench2drive_dev_10_open_loop.json \
    --agent=leaderboard/team_code/drivetransformer_vis_agent_open_loop.py \
    --agent-config=DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py+DriveTransformer/ckpts/drivetransformer_large.pth \
    --debug=0 \
    --record="" \
    --port=30002 \
    --traffic-manager-port=50001 \
    --gpu-rank=0

echo ""
echo "========================================"
echo "仿真结束"
echo "========================================"