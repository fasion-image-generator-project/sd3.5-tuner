---
license: other
base_model: "stabilityai/stable-diffusion-3.5-medium"
tags:
  - sd3
  - sd3-diffusers
  - text-to-image
  - diffusers
  - simpletuner
  - not-for-all-audiences
  - lora
  - template:sd-lora
  - lycoris
inference: true
widget:
- text: 'unconditional (blank prompt)'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_0_0.png
- text: 'unconditional (blank prompt)'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_1_1.png
- text: 'A minimalist and modern fashion design featuring a clean-cut oversized wool coat'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_2_0.png
- text: 'A minimalist and modern fashion design featuring a clean-cut oversized wool coat'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_3_1.png
---

# simpletuner-lora

This is a LyCORIS adapter derived from [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium).


The main validation prompt used during training was:
```
A minimalist and modern fashion design featuring a clean-cut oversized wool coat
```


## Validation settings
- CFG: `3.0`
- CFG Rescale: `0.0`
- Steps: `20`
- Sampler: `FlowMatchEulerDiscreteScheduler`
- Seed: `42`
- Resolutions: `1024x1024,1280x768`
- Skip-layer guidance: 

Note: The validation settings are not necessarily the same as the [training settings](#training-settings).

You can find some example images in the following gallery:


<Gallery />

The text encoder **was not** trained.
You may reuse the base model text encoder for inference.


## Training settings

- Training epochs: 1
- Training steps: 25000
- Learning rate: 0.0001
  - Learning rate schedule: polynomial
  - Warmup steps: 100
- Max grad norm: 2.0
- Effective batch size: 1
  - Micro-batch size: 1
  - Gradient accumulation steps: 1
  - Number of GPUs: 1
- Gradient checkpointing: True
- Prediction type: flow-matching (extra parameters=['shift=3'])
- Optimizer: adamw_bf16
- Trainable parameter precision: Pure BF16
- Caption dropout probability: 10.0%


### LyCORIS Config:
```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "full_matrix": true,
    "linear_alpha": 1,
    "factor": 16,
    "linear_dim": 32,
    "apply_preset": {
        "target_module": [
            "Attention",
            "FeedForward"
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 16
            },
            "FeedForward": {
                "factor": 8
            }
        }
    }
}
```

## Datasets

### image-dataset
- Repeats: 0
- Total number of images: 5667
- Total number of aspect buckets: 1
- Resolution: 1.048576 megapixels
- Cropped: False
- Crop style: None
- Crop aspect: None
- Used for regularisation data: No


## Inference


```python
import torch
from diffusers import DiffusionPipeline
from lycoris import create_lycoris_from_weights


def download_adapter(repo_id: str):
    import os
    from huggingface_hub import hf_hub_download
    adapter_filename = "pytorch_lora_weights.safetensors"
    cache_dir = os.environ.get('HF_PATH', os.path.expanduser('~/.cache/huggingface/hub/models'))
    cleaned_adapter_path = repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    path_to_adapter = os.path.join(cache_dir, cleaned_adapter_path)
    path_to_adapter_file = os.path.join(path_to_adapter, adapter_filename)
    os.makedirs(path_to_adapter, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id, filename=adapter_filename, local_dir=path_to_adapter
    )

    return path_to_adapter_file
    
model_id = 'stabilityai/stable-diffusion-3.5-medium'
adapter_repo_id = 'eunolee/simpletuner-lora'
adapter_filename = 'pytorch_lora_weights.safetensors'
adapter_file_path = download_adapter(repo_id=adapter_repo_id)
pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16) # loading directly in f16
lora_scale = 1.0
wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
wrapper.merge_to()

prompt = "A minimalist and modern fashion design featuring a clean-cut oversized wool coat"
negative_prompt = 'blurry, cropped, ugly'

## Optional: quantise the model to save on vram.
## Note: The model was quantised during training, and so it is recommended to do the same during inference time.
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)
    
pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # the pipeline is already in its target precision level
image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=3.0,
).images[0]
image.save("output.png", format="PNG")
```



