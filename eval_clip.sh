export HF_ENDPOINT=https://hf-mirror.com

fresh_thresholds=(6)
max_orders=(1)

smooth_rates=(0.003)

for fresh_threshold in ${fresh_thresholds[@]};
do
    for order in ${max_orders[@]};
    do 
        for rate in ${smooth_rates[@]};
        do 
            echo "evaluate on fresh_threhols: $fresh_threshold, max_order: $order smooth rate: $rate..."
            python clip_score.py --image_folder single_and_double/$fresh_threshold-$order/Taylor-Cluster/256/$rate
        done
    done
done