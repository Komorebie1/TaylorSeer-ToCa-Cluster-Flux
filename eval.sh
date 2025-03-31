export HF_ENDPOINT=https://hf-mirror.com

smooth_rates=(0.005)

for rate in ${smooth_rates[@]};
do 
    echo "evaluate on smooth rate: $rate..."
    python image_reward.py --image_folder cluster-both-smooth-img/4-2/Taylor-Cluster/128/$rate --model_name ImageReward-v1.0
done