export FLUX_DEV="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors"
export AE="/root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev/ae.safetensors"

max_orders=(1 2)
fresh_thresholds=(5 6 7)
smooth_rates=(0.0 0.001 0.002 0.003 0.004 0.005)

for max_order in ${max_orders[@]}; do
    for fresh_threshold in ${fresh_thresholds[@]}; do

        for rate in ${smooth_rates[@]}; do
            echo "running with smooth rate: $rate..."

            CUDA_VISIBLE_DEVICES=0 python src/sample.py --prompt_file ./prompt.txt \
            --width 1024 --height 1024 \
            --model_name flux-dev \
            --add_sampling_metadata \
            --output_dir ./single_and_double \
            --num_steps 50 \
            --mode Taylor-Cluster \
            --test_FLOPs \
            --max_order $max_order \
            --fresh_threshold $fresh_threshold \
            --cluster_num 256 \
            --smooth_rate $rate
        done
    done
done
