export HF_ENDPOINT=https://hf-mirror.com

smooth_rates=(0.0)

for rate in ${smooth_rates[@]};
do 
    echo "evaluate on smooth rate: $rate..."
    python image_reward.py --image_folder cluster-both-smooth-img/6-1/Taylor-Cluster/0.0-16 --model_name ImageReward-v1.0
done