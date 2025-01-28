import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

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

    # Trigger training (asynchronously, so the server remains responsive)
    subprocess.Popen(["python", "train.py", concept_name, model, concept_dir])

    return jsonify({"message": "Training started!", "concept": concept_name, "model": model})

@app.route("/generate", methods=["POST"])
def generate():
    concept_prompt = request.form["concept_prompt"]
    concept_name_gen = request.form["concept_name_gen"]

    # Define the path to the trained model
    model_path = f"trained_models/{concept_name_gen}"

    if not os.path.exists(model_path):
        return jsonify({"error": f"No model found for concept '{concept_name_gen}'"}), 404

    # Run the generation script
    generated_images_dir = f"generated_images/{concept_name_gen}"
    os.makedirs(generated_images_dir, exist_ok=True)

    subprocess.run(["python", "generate.py", concept_prompt, model_path, generated_images_dir])

    # Return the generated images (as URLs)
    images = [f"/{generated_images_dir}/{img}" for img in os.listdir(generated_images_dir) if img.endswith(".png")]
    return jsonify({"images": images})

if __name__ == "__main__":
    app.run(debug=True)
