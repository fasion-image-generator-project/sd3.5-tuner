import os
import time
import torch
import argparse
from diffusers import DiffusionPipeline
from lycoris import create_lycoris_from_weights
from huggingface_hub import hf_hub_download
from optimum.quanto import quantize, freeze, qint8


def resolve_adapter_path(adapter_path_or_repo: str, adapter_filename: str = "pytorch_lora_weights.safetensors"):
    """
    로컬 경로 (절대 또는 상대 경로)로 시작하면 로컬 파일로 취급하고,
    그렇지 않으면 Hugging Face repo id로 간주하여 파일을 다운로드합니다.
    """
    if adapter_path_or_repo.startswith('.') or adapter_path_or_repo.startswith('/'):
        abs_path = os.path.abspath(adapter_path_or_repo)
        if os.path.isfile(abs_path):
            return abs_path
        elif os.path.isdir(abs_path):
            candidate = os.path.join(abs_path, adapter_filename)
            if os.path.isfile(candidate):
                return candidate
            else:
                raise FileNotFoundError(f"'{adapter_filename}' not found in directory '{abs_path}'.")
        else:
            raise FileNotFoundError(f"Local adapter path '{abs_path}' does not exist.")
    else:
        print(f"[INFO] Downloading LoRA adapter from Hugging Face: {adapter_path_or_repo}")
        return hf_hub_download(repo_id=adapter_path_or_repo, filename=adapter_filename)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    
    total_start = time.time()

    # 1. Base model 로딩
    model_start = time.time()
    pipeline = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    model_time = time.time() - model_start
    print(f"[TIME] Base model load time: {model_time:.2f} seconds")

    # 2. LoRA adapter 로드 및 merge
    merge_start = time.time()
    adapter_path = resolve_adapter_path(args.adapter_path)
    wrapper, _ = create_lycoris_from_weights(args.lora_scale, adapter_path, pipeline.transformer)
    wrapper.merge_to()
    merge_time = time.time() - merge_start
    print(f"[TIME] LoRA merge time: {merge_time:.2f} seconds")

    # # 3. Quantize (VRAM 절약)     # VRAM 차이 거의 없음. 추론시간 잡아먹으므로 제거
    # quantize_start = time.time()
    # quantize(pipeline.transformer, weights=qint8)
    # freeze(pipeline.transformer)
    # quantize_time = time.time() - quantize_start
    # print(f"[TIME] Quantization time: {quantize_time:.2f} seconds")

    # 4. 디바이스로 이동
    pipeline.to(device)

    # 5. 추론 실행
    inference_start = time.time()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        generator=generator,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidance_scale,
    ).images[0]
    inference_time = time.time() - inference_start
    print(f"[TIME] Inference time: {inference_time:.2f} seconds")

    # 6. 결과 저장
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    image.save(output_path, format="PNG")
    total_elapsed = time.time() - total_start

    print(f"[DONE] Image saved to: {output_path}")
    print(f"[TIME] Total execution time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stable Diffusion inference with LoRA adapter")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="blurry, cropped, ugly", help="Negative prompt")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium", help="Base model ID")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter file (local path) or Hugging Face repo id")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="Scale for LoRA adapter merge")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save output image")
    parser.add_argument("--output_name", type=str, default="output.png", help="Output filename")

    args = parser.parse_args()
    main(args)
