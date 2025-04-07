export HF_ENDPOINT=https://hf-mirror.com

fresh_thresholds=(4 5 6 7 8)
max_orders=(1 2)

smooth_rates=(0.0)

for fresh_threshold in ${fresh_thresholds[@]};
do
    for order in ${max_orders[@]};
    do 
        for rate in ${smooth_rates[@]};
        do 
            echo "evaluate on fresh_threhols: $fresh_threshold, max_order: $order smooth rate: $rate..."
            python clip_score.py --image_folder April_1/$fresh_threshold-$order/Taylor
        done
    done
done