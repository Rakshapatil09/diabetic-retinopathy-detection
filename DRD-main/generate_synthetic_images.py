from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os


CLASSES = ["Proliferate_DR", "Severe", "Moderate", "Mild", "No_DR"]
LORA_BASE = "/Users/pradeepkd/Desktop/projectphase1/lora_models"
OUTPUT_BASE = "/Users/pradeepkd/Desktop/projectphase1/synthetic_images"


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    ensure_directory(OUTPUT_BASE)

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Use Apple Silicon GPU (Metal) if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe.to(device)

    for class_name in CLASSES:
        class_lora_dir = os.path.join(LORA_BASE, class_name)
        if not os.path.isdir(class_lora_dir):
            print(f"[WARN] Skipping {class_name} — LoRA directory not found: {class_lora_dir}")
            continue

        pipe.load_lora_weights(class_lora_dir)
        class_output_dir = os.path.join(OUTPUT_BASE, class_name)
        ensure_directory(class_output_dir)

        prompt = f"a photo of a {class_name} retina"
        # Generate 200 images per class by default; adjust as needed
        num_images = 200
        for i in range(num_images):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            out_path = os.path.join(class_output_dir, f"{class_name}_{i + 1:04d}.png")
            image.save(out_path)
        print(f"[DONE] Generated {num_images} images for {class_name}")


if __name__ == "__main__":
    main()


