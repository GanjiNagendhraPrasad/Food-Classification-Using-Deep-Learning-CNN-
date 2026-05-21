import os
import io
import json

import redis
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

# =====================================================
# APP SETUP
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

CORS(app)

# =====================================================
# REDIS
# =====================================================

r = redis.Redis(
    host="127.0.0.1",
    port=6379,
    decode_responses=True
)

# =====================================================
# LABELS
# =====================================================

LABELS = [
    "butter_naan",
    "pav_bhaji",
    "Sandwich",
    "chicken_curry",
    "Hot Dog",
    "cheesecake",
    "sushi",
    "chai",
    "burger",
    "ice_cream",
    "kadai_paneer",
    "Baked Potato",
    "chapati",
    "masala_dosa",
    "dal_makhani",
    "Donut",
    "jalebi",
    "fried_rice",
    "chole_bhature",
    "kulfi",
    "kaathi_rolls",
    "dhokla",
    "Fries",
    "omelette",
    "pakode",
    "momos",
    "paani_puri",
    "samosa",
    "Taco",
    "idli",
    "Taquito",
    "Crispy Chicken",
    "pizza",
    "apple_pie"
]

NUM_CLASSES = len(LABELS)

# =====================================================
# MODEL METRICS
# =====================================================

MODEL_METRICS = {
    "custom_cnn": {
        "accuracy": 49.4,
        "precision": 51.2,
        "f1": 48.7
    },

    "vgg16": {
        "accuracy": 67.3,
        "precision": 68.9,
        "f1": 67.1
    },

    "resnet50": {
        "accuracy": 72.1,
        "precision": 73.5,
        "f1": 72.4
    }
}

# =====================================================
# MODEL CACHE
# =====================================================

_models = {}

# =====================================================
# CUSTOM CNN
# =====================================================

def build_custom_cnn(input_shape=(256, 256, 3)):

    inputs = tf.keras.Input(shape=input_shape)

    x = inputs

    for filters in [32, 64, 128, 256, 512]:

        x = tf.keras.layers.Conv2D(
            filters,
            (3, 3),
            activation="relu",
            padding="same"
        )(x)

        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)

    for units in [1024, 512, 256, 128, 64]:

        x = tf.keras.layers.Dense(
            units,
            activation="relu"
        )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    return tf.keras.Model(inputs, outputs)

# =====================================================
# VGG16
# =====================================================

def build_vgg16():

    base = tf.keras.applications.VGG16(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = tf.keras.layers.Flatten()(base.output)

    x = tf.keras.layers.Dense(
        512,
        activation="relu"
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    return tf.keras.Model(base.input, outputs)

# =====================================================
# RESNET50
# =====================================================

def build_resnet50():

    base = tf.keras.applications.ResNet50(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)

    x = tf.keras.layers.Dense(
        512,
        activation="relu"
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    return tf.keras.Model(base.input, outputs)

# =====================================================
# MODEL FILES
# =====================================================

WEIGHT_FILES = {
    "custom_cnn": "food_classification_weights.weights.h5",
    "vgg16": "vgg16_food_classification_weights.weights.h5",
    "resnet50": "resnet50_food_classification_weights.weights (3).h5"
}

BUILDERS = {
    "custom_cnn": build_custom_cnn,
    "vgg16": build_vgg16,
    "resnet50": build_resnet50
}

# =====================================================
# LOAD MODEL
# =====================================================

def load_model(model_name):

    if model_name in _models:
        return _models[model_name]

    model = BUILDERS[model_name]()

    path = os.path.join(
        BASE_DIR,
        WEIGHT_FILES[model_name]
    )

    try:

        model.load_weights(path)

    except Exception:

        model.load_weights(
            path,
            skip_mismatch=True,
            by_name=True
        )

    _models[model_name] = model

    return model

# =====================================================
# PREPROCESS
# =====================================================

INPUT_SIZES = {
    "custom_cnn": (256, 256),
    "vgg16": (224, 224),
    "resnet50": (224, 224)
}

def preprocess(image_bytes, model_name):

    size = INPUT_SIZES[model_name]

    img = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    img = img.resize(size)

    img_array = np.array(
        img,
        dtype=np.float32
    )

    img_batch = np.expand_dims(
        img_array,
        axis=0
    )

    if model_name == "vgg16":

        from tensorflow.keras.applications.vgg16 import preprocess_input

        return preprocess_input(img_batch)

    elif model_name == "resnet50":

        from tensorflow.keras.applications.resnet50 import preprocess_input

        return preprocess_input(img_batch)

    return img_batch / 255.0

# =====================================================
# REDIS NUTRITION
# =====================================================

def get_nutrition(food_name):

    try:

        raw = r.get("food_details")

        if not raw:
            return None

        food_details = json.loads(raw)

        normalized_name = (
            food_name
            .strip()
            .lower()
            .replace(" ", "_")
        )

        normalized_food_details = {
            key.strip().lower().replace(" ", "_"): value
            for key, value in food_details.items()
        }

        nutrition = normalized_food_details.get(normalized_name)

        if nutrition:

            return {
                "calories": nutrition.get("calories", 0),
                "protein": nutrition.get("protein", 0),
                "carbs": nutrition.get("carbs", 0),
                "fats": nutrition.get("fats", 0),
                "fiber": nutrition.get("fiber", 0)
            }

    except Exception as e:

        print(f"[REDIS ERROR] {e}")

    return None

# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def home():

    return render_template("index.html")

# =====================================================
# PREDICT
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return jsonify({
                "error": "No image uploaded"
            }), 400

        model_name = (
            request.form.get(
                "model",
                "custom_cnn"
            )
            .strip()
            .lower()
            .replace(" ", "_")
        )

        image_bytes = request.files["image"].read()

        model = load_model(model_name)

        processed_image = preprocess(
            image_bytes,
            model_name
        )

        preds = model.predict(
            processed_image,
            verbose=0
        )[0]

        top_idx = int(np.argmax(preds))

        predicted_label = LABELS[top_idx]

        confidence = round(
            float(preds[top_idx]) * 100,
            2
        )

        # TOP 5

        top5_indices = np.argsort(preds)[::-1][:5]

        top5 = []

        for i in top5_indices:

            top5.append({
                "label": LABELS[i],
                "confidence": round(float(preds[i]) * 100, 2)
            })

        return jsonify({

            "predicted_label": predicted_label,

            "confidence": confidence,

            "model_used": model_name,

            "top5": top5,

            "nutrition": get_nutrition(predicted_label),

            "metrics": {
                "accuracy": MODEL_METRICS[model_name]["accuracy"],
                "precision": MODEL_METRICS[model_name]["precision"],
                "f1": MODEL_METRICS[model_name]["f1"]
            }

        })

    except Exception as e:

        print(f"[ERROR] {e}")

        return jsonify({
            "error": str(e)
        }), 500

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )