export HF_ENDPOINT=https://hf-mirror.com

path='./samples/Taylor-Cache-8-1-0.01-16'

python image_reward.py \
    --image_folder $path \
    --model_name ImageReward-v1.0 \