import sys
import os
import json
from pathlib import Path
from diffusers import StableDiffusionPipeline

def train_model(concept_name, model_name, data_dir):
    print(f"Starting DreamBooth training for concept '{concept_name}' with model '{model_name}'...")

    # Set up training directory
    output_dir = Path(f"trained_models/{concept_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate concepts_list.json (required by DreamBooth training)
    concepts_list = [
        {
            "instance_prompt": f"photo of {concept_name} person",
            "class_prompt": "photo of a person",
            "instance_data_dir": str(data_dir),
            "class_data_dir": "/path/to/class/images"  # General class images
        }
    ]
    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)

    # Run DreamBooth training
    command = f"""
    HUGGINGFACE_HUB_TOKEN='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' python train_dreambooth.py \\
        --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \\
        --instance_data_dir="{data_dir}" \\
        --output_dir="{output_dir}" \\
        --instance_prompt="photo of {concept_name} person" \\
        --class_prompt="photo of a person" \\
        --resolution=512 \\
        --train_batch_size=1 \\
        --learning_rate=5e-6 \\
        --max_train_steps=1000 \\
        --mixed_precision="fp16"
        --device="cuda"
    """
    os.system(command)

    print(f"Training completed. Model saved at {output_dir}")

def generate_images(prompt, model_path, output_dir):
    print(f"Generating images for prompt: '{prompt}' using model at {model_path}")

    # Load the fine-tuned model
    pipeline = StableDiffusionPipeline.from_pretrained(model_path)
    pipeline.to("cuda")

    # Generate images
    images = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=4).images

    # Save generated images
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.save(f"{output_dir}/image_{i}.png")

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
