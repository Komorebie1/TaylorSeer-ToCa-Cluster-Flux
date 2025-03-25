export FLUX_DEV="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"
export AE="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/ae.safetensors"

max_order=1
fresh_threshold=5


CUDA_VISIBLE_DEVICES=0 python src/sample.py --prompt_file ./prompt.txt \
  --width 1024 --height 1024 \
  --model_name flux-dev \
  --add_sampling_metadata \
  --output_dir ./samples/test \
  --num_steps 50 \
  --max_order $max_order \
  --fresh_threshold $fresh_threshold \
  --cluster_num 16 \
  --smooth_rate 0.007 \
