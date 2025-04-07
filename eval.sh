export HF_ENDPOINT=https://hf-mirror.com

# fresh_threshold=7
fresh_thresholds=(4 5 6 7 8)
max_order=1

# python image_reward.py --image_folder April_1/$fresh_threshold-$max_order/Taylor --model_name ImageReward-v1.0
smooth_rates=(0.0 0.001 0.002 0.003 0.004 0.005)
for fresh_threshold in ${fresh_thresholds[@]};
do
    for rate in ${smooth_rates[@]};
    do 
        echo "evaluate on smooth rate: $rate..."
        python image_reward.py --image_folder April_4/$fresh_threshold-$max_order/Taylor-Cluster/256/$rate --model_name ImageReward-v1.0
        # python image_reward.py --image_folder origin_flux/4-1/Taylor --model_name ImageReward-v1.0
    done
done