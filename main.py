import os
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import subprocess
from pyngrok import ngrok

app = Flask(__name__)

# Create public URL for the app
public_url = ngrok.connect(5000)
print(f"🚀 Public URL: {public_url}")

# Upload & Output Directories
UPLOAD_FOLDER = "uploads"
GENERATED_IMAGES_FOLDER = "static/generated_images"
TRAINED_MODELS_FOLDER = "trained_models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_IMAGES_FOLDER, exist_ok=True)
os.makedirs(TRAINED_MODELS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Serve Generated Images
@app.route('/static/generated_images/<path:filename>')
def serve_generated_images(filename):
    return send_from_directory(GENERATED_IMAGES_FOLDER, filename)

@app.route("/")
def index():
    return render_template("index.html")

# Train Model Endpoint
@app.route("/train", methods=["POST"])
def train():
    concept_name = request.form["concept_name"]
    model = request.form["model"]
    files = request.files.getlist("images")

    # Save uploaded files
    concept_dir = os.path.join(UPLOAD_FOLDER, concept_name)
    os.makedirs(concept_dir, exist_ok=True)
    
    for file in files:
        filename = secure_filename(file.filename)
        file.save(os.path.join(concept_dir, filename))

    # Run training in the background
    subprocess.Popen(["python", "train.py", concept_name, model, concept_dir])

    return jsonify({"message": "Training started!", "concept": concept_name, "model": model})

# Generate Images Endpoint
@app.route("/generate", methods=["POST"])
def generate():
    concept_prompt = request.form["concept_prompt"]
    concept_name_gen = request.form["concept_name_gen"]

    model_path = f"{TRAINED_MODELS_FOLDER}/{concept_name_gen}"

    if not os.path.exists(model_path):
        return jsonify({"error": f"No model found for concept '{concept_name_gen}'"}), 404

    output_dir = f"{GENERATED_IMAGES_FOLDER}/{concept_name_gen}"
    os.makedirs(output_dir, exist_ok=True)

    subprocess.run(["python", "train.py", "generate", concept_prompt, model_path, output_dir])

    images = [f"/static/generated_images/{concept_name_gen}/{img}" for img in os.listdir(output_dir) if img.endswith(".png")]
    
    if not images:
        return jsonify({"error": "No images were generated."}), 500

    return jsonify({"images": images})

# Get Available Trained Concepts
@app.route("/get_concepts", methods=["GET"])
def get_concepts():
    concepts = [name for name in os.listdir(TRAINED_MODELS_FOLDER) if os.path.isdir(os.path.join(TRAINED_MODELS_FOLDER, name))]
    return jsonify({"concepts": concepts})

if __name__ == "__main__":
    app.run(port=5000)
