#!/bin/bash

env_name=$1
python_version=$2

# Check if a name and Python version were provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Missing env_name or python version. Usage: ./google_cloud_config.sh ENV_NAME PYTHON_VERSION"
  exit 1
fi

sudo apt update
sudo apt-get install tmux
sudo apt install -y ffmpeg
sudo apt-get install git-lfs

# check if conda is installed
if ! which conda > /dev/null; then
  # Download the installer script
  curl -o install-conda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

  # Run the installer script
  bash install-conda.sh
fi

conda create --name $env_name python=$python_version
conda activate env_name

python run_speech_recognition_seq2seq_streaming.py \
	--model_name_or_path="openai/whisper-small" \
	--dataset_name="mozilla-foundation/common_voice_11_0" \
	--dataset_config_name="es" \
	--language="spanish" \
	--train_split_name="train+validation" \
	--eval_split_name="test" \
	--model_index_name="Whisper Small Spanish" \
	--max_steps="5000" \
	--output_dir="./" \
	--per_device_train_batch_size="64" \
	--per_device_eval_batch_size="32" \
	--logging_steps="25" \
	--learning_rate="1e-5" \
	--warmup_steps="500" \
	--evaluation_strategy="steps" \
	--eval_steps="1000" \
	--save_strategy="steps" \
	--save_steps="1000" \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="sentence" \
	--freeze_feature_encoder="False" \
	--report_to="tensorboard" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
	--load_best_model_at_end \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--do_normalize_eval \
	--streaming \
	--use_auth_token \
	--push_to_hub

