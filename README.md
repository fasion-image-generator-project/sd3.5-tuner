# sd3.5-tuner

## 추론 명령어 

```
python run_inference.py \
  --prompt "A stylish black bomber jacket with minimalistic patches on the sleeves" \
  --adapter_path ./outputs/models/pytorch_lora_weights.safetensors \
  --model_id stabilityai/stable-diffusion-3.5-medium \
  --lora_scale 1.0 \
  --steps 20 \
  --seed 42 \
  --width 1024 \
  --height 1024 \
  --guidance_scale 3.0 \
  --output_dir ./results \
  --output_name bomber_jacket.png
```
