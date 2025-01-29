import sys
import os
import torch
import json
from pathlib import Path
from diffusers import StableDiffusionPipeline, DDIMScheduler

def load_pipeline(model_path):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, 
        safety_checker=None, 
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    
    return pipe

def train_model(concept_name, model_name, data_dir):
    print(f"Starting DreamBooth training for concept '{concept_name}' with model '{model_name}'...")

    output_dir = Path(f"trained_models/{concept_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    concepts_list = [
        {
            "instance_prompt": f"photo of {concept_name} person",
            "class_prompt": "photo of a person",
            "instance_data_dir": str(data_dir),
            "class_data_dir": "/path/to/class/images"
        }
    ]
    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)

    os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_huggingface_token_here"

    command = f"""
    python train_dreambooth.py \
        --pretrained_model_name_or_path="{model_name}" \
        --instance_data_dir="{data_dir}" \
        --output_dir="{output_dir}" \
        --instance_prompt="photo of {concept_name}" \
        --class_prompt="photo of a person" \
        --resolution=512 \
        --train_batch_size=1 \
        --use_8bit_adam \
        --learning_rate=5e-6 \
        --max_train_steps=200 \
        --mixed_precision="fp16" \
        --gradient_accumulation_steps=1
    """
    os.system(command.replace("\n", " "))

    print(f"Training completed. Model saved at {output_dir}")

def generate_images(prompt, model_path, output_dir):
    print(f"Generating images for prompt: '{prompt}' using model at {model_path}")

    pipeline = load_pipeline(model_path)

    images = pipeline(prompt, num_inference_steps=30, guidance_scale=7.5, num_images_per_prompt=4).images

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_images = []
    
    for i, img in enumerate(images):
        image_path = f"{output_dir}/image_{i}.png"
        img.save(image_path)
        saved_images.append(image_path)

    return saved_images

if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "generate":
        prompt = sys.argv[2]
        model_path = sys.argv[3]
        output_dir = sys.argv[4]
        generate_images(prompt, model_path, output_dir)
    else:
        concept_name = sys.argv[1]
        model_name = sys.argv[2]
        data_dir = sys.argv[3]
        train_model(concept_name, model_name, data_dir)
