from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

MODELS = {
    "baseline": "models/baseline.h5",
    "dense_64_epochs_5": "models/dense_64_epochs_5.h5",
    "dense_128_epoch_5": "models/dense_128_epochs_5.h5",
    "dense_64_epochs_10": "models/dense_64_epochs_10.h5",
    "dense_128_epochs_10": "models/dense_128_epochs_10.h5"
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
    predictions = {}  # Menyimpan prediksi semua model

    if request.method == 'POST':
        file = request.files['image']

        if file:
            upload_folder = 'static/uploads'
            os.makedirs(upload_folder, exist_ok=True)

            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)

            try:
                img_array = prepare_image(filepath)

                for model_key, model_path in MODELS.items():
                    if not os.path.exists(model_path):
                        predictions[model_key] = f"Model '{model_key}' tidak ditemukan."
                        continue

                    model = load_model(model_path)
                    preds = model.predict(img_array)
                    confidence = np.max(preds)
                    predicted_class = CLASS_NAMES[np.argmax(preds)]

                    predictions[model_key] = {
                        "class": predicted_class,
                        "confidence": f"{confidence:.2%}"
                    }

            except Exception as e:
                return render_template('index.html', error=f"Error: {str(e)}")

            return render_template('index.html', predictions=predictions, image_path=filepath)

    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
