export HF_ENDPOINT=https://hf-mirror.com

# path='./samples/Taylor-7-4-0.004-16'
path='./samples/Taylor-Cluster-8-1-0.006-16'

smooth_rates=(0.0 0.001 0.002 0.003 0.004 0.005)

for rate in ${smooth_rates[@]};
do 
    echo "evaluate on smooth rate: $rate..."
    python image_reward.py --image_folder ./cluster-both-smooth-img/7-1/Taylor-Cluster/$rate-16 --model_name ImageReward-v1.0
done


# python image_reward.py \
#     --image_folder $path \
#     --model_name ImageReward-v1.0 \