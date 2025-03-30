export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download google/t5-v1_1-xxl --local-dir /root/autodl-tmp/pretrained_models/google/t5-v1_1-xxl
huggingface-cli download openai/clip-vit-large-patch14 --local-dir /root/autodl-tmp/pretrained_models/openai/clip-vit-large-patch14
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir /root/autodl-tmp/pretrained_models/black-forest-labs/FLUX.1-dev