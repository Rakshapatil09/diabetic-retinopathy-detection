#!/bin/bash

# Fine-tune Stable Diffusion (LoRA) per class using DreamBooth LoRA script
# Requirements:
#   pip install diffusers[torch] transformers accelerate torch torchvision peft
#   curl -O https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora.py

set -euo pipefail

CLASSES=("Proliferate_DR" "Severe" "Moderate" "Mild" "No_DR")
BASE_DIR="/Users/pradeepkd/Desktop/projectphase1/archive/colored_images"
OUTPUT_BASE="/Users/pradeepkd/Desktop/projectphase1/lora_models"

mkdir -p "$OUTPUT_BASE"

for CLASS in "${CLASSES[@]}"; do
  INSTANCE_DIR="$BASE_DIR/$CLASS"
  if [ ! -d "$INSTANCE_DIR" ]; then
    echo "[WARN] Skipping $CLASS — directory not found: $INSTANCE_DIR" >&2
    continue
  fi

  echo "[INFO] Training LoRA for class: $CLASS"
  python3 /Users/pradeepkd/Desktop/projectphase1/train_dreambooth_lora.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --instance_data_dir="$INSTANCE_DIR" \
    --output_dir="$OUTPUT_BASE/$CLASS" \
    --instance_prompt="a photo of a $CLASS retina" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-4 \
    --max_train_steps=400 \
    --seed=1337
done

echo "[DONE] LoRA training completed for available classes."


