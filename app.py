from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

MODELS = {
    "pretrained_no_ft": "models/pretrained_no_finetune.h5",
    "finetune_25": "models/pretrained_finetune_25.h5",
    "finetune_50": "models/pretrained_finetune_50.h5",
    "finetune_100": "models/pretrained_finetune_100.h5"
}

CLASS_NAMES = {0: "Bean", 1: "Bitter_Gourd", 2: "Bottle_Gourd", 3: "Brinjal", 4: "Broccoli", 5: "Cabbage", 6: "Capsicum", 7: "Carrot", 8: "Cauliflower",
          9: "Cucumber", 10: "Papaya", 11: "Potato", 12: "Pumpkin", 13: "Radish", 14: "Tomato"}  

def load_selected_model(model_key):
    model_path = MODELS.get(model_key)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_key}' tidak ditemukan.")
    return load_model(model_path)

def prepare_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_model = None

    if request.method == 'POST':
        file = request.files['image']
        model_key = request.form['model']
        selected_model = model_key

        if file:
            filepath = os.path.join('static/uploads', file.filename)
            file.save(filepath)

            try:
                print(f"[INFO] Model yang dipilih: {model_key}")  # <<--- CETAK MODEL YANG DIPILIH
                model = load_selected_model(model_key)
                img_array = prepare_image(filepath)
                preds = model.predict(img_array)
                prediction = CLASS_NAMES[np.argmax(preds)]
                print(f"[INFO] Prediksi hasil: {prediction}")  # <<--- CETAK HASIL PREDIKSI
            except Exception as e:
                prediction = f"Error: {str(e)}"
                print(f"[ERROR] {str(e)}")

            return render_template('index.html', prediction=prediction, image_path=filepath, selected_model=selected_model)

    return render_template('index.html', prediction=prediction, selected_model=selected_model)


if __name__ == '__main__':
    app.run(debug=True)
