#!/usr/bin/env bash
# Set up and launch Flash ALOHA ACT training for YAM sponge on a cloud GPU node.
#   bash train_h100.sh setup     # repo + env + data + config check
#   bash train_h100.sh smoke     # 200 steps, to time it and catch errors
#   bash train_h100.sh train     # full 100k-step run in tmux "yamtrain", log in ~/yam/train.log
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd); W=~/yam; R=$W/flash-aloha
case "${1:-}" in
setup)
  mkdir -p $W/data
  [ -d $R ] || git clone -q $B/flash-aloha.bundle $R
  (cd $R && git checkout -q aec35bb)
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=$HOME/.local/bin:$PATH
  (cd $R && uv sync --extra dev)
  tar -xf $B/yam_sponge_flash60.tar -C $W/data
  cp $B/yam_sponge_act.yaml $R/src/config/training/
  sed "s#DATA_DIR#$W/data/yam_sponge_flash60_train#" $B/yam_sponge.yaml > $R/src/config/tasks/yam_sponge.yaml
  cp $B/yam_mask.json $B/yam_mask.py $W/
  (cd $R && .venv/bin/python scripts/config/check_training_config.py --training-config yam_sponge_act >/dev/null && echo config ok)
  nvidia-smi --query-gpu=name,memory.total --format=csv; free -g | sed -n 2p; nproc
  ;;
smoke)
  cd $R && .venv/bin/python scripts/policy/imitate_episodes_train.py --training-config yam_sponge_act --num_steps 200 2>&1 | tail -30
  ;;
train)
  cd $R && tmux new -d -s yamtrain ".venv/bin/python scripts/policy/imitate_episodes_train.py --training-config yam_sponge_act > $W/train.log 2>&1"
  echo started; ;;
*) echo "usage: $0 setup|smoke|train"; exit 1;;
esac
