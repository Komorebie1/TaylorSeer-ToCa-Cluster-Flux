export HF_ENDPOINT=https://hf-mirror.com

path='./samples/Taylor-5-1-0.007-16'

python image_reward.py \
    --image_folder $path \
    --model_name ImageReward-v1.0 \