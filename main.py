import io
import json
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Plant Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "plant_model_v3.h5"
CLASS_NAMES_PATH = "class_names.json"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)


def segment_leaf(image_pil):
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 20, 20], dtype=np.uint8)
    upper_green = np.array([100, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image_pil

    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < 500:
        return image_pil

    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_rgb = image_np[y:y+h, x:x+w]

    if cropped_rgb.size == 0:
        return image_pil

    return Image.fromarray(cropped_rgb)


def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = segment_leaf(image)
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        print("preprocess_image error:", repr(e))
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


def get_class_name(index: int) -> str:
    if isinstance(CLASS_NAMES, list):
        if 0 <= index < len(CLASS_NAMES):
            return CLASS_NAMES[index]
    elif isinstance(CLASS_NAMES, dict):
        return CLASS_NAMES.get(str(index), f"Unknown class {index}")

    return f"Unknown class {index}"

ADVICE_MAP = {
    "Apple___Apple_scab": {
        "status": "Diseased",
        "advice": "Apple scab detected. Remove infected leaves and fallen debris, improve airflow around the tree, avoid overhead watering, and use a suitable fungicide if needed."
    },
    "Apple___Black_rot": {
        "status": "Diseased",
        "advice": "Apple black rot detected. Remove infected fruits, leaves, and branches, prune dead wood, sanitize tools, and apply an appropriate fungicide if the infection spreads."
    },
    "Apple___Cedar_apple_rust": {
        "status": "Diseased",
        "advice": "Cedar apple rust detected. Remove nearby alternate hosts if possible, prune affected parts, improve airflow, and apply a recommended fungicide during active infection periods."
    },
    "Apple___healthy": {
        "status": "Healthy",
        "advice": "The apple leaf appears healthy. Continue regular watering, good airflow, and routine monitoring."
    },
    "Background_without_leaves": {
        "status": "No Leaf Detected",
        "advice": "No clear leaf was detected in the image. Please upload a clear photo showing a single plant leaf."
    },
    "Blueberry___healthy": {
        "status": "Healthy",
        "advice": "The blueberry leaf appears healthy. Continue normal care and regular monitoring."
    },
    "Cherry___Powdery_mildew": {
        "status": "Diseased",
        "advice": "Cherry powdery mildew detected. Remove affected leaves, reduce humidity around the plant, improve airflow, and apply a suitable fungicide if necessary."
    },
    "Cherry___healthy": {
        "status": "Healthy",
        "advice": "The cherry leaf appears healthy. Continue normal care and regular monitoring."
    },
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": {
        "status": "Diseased",
        "advice": "Corn gray leaf spot detected. Remove heavily infected leaves if practical, avoid prolonged leaf wetness, rotate crops, and apply a fungicide if the disease becomes severe."
    },
    "Corn___Common_rust": {
        "status": "Diseased",
        "advice": "Corn common rust detected. Monitor disease spread, improve field airflow if possible, and use resistant varieties or fungicide treatment when needed."
    },
    "Corn___Northern_Leaf_Blight": {
        "status": "Diseased",
        "advice": "Corn northern leaf blight detected. Remove infected plant residue after harvest, rotate crops, and consider fungicide treatment if symptoms spread significantly."
    },
    "Corn___healthy": {
        "status": "Healthy",
        "advice": "The corn leaf appears healthy. Continue normal care and regular monitoring."
    },
    "Grape___Black_rot": {
        "status": "Diseased",
        "advice": "Grape black rot detected. Remove infected leaves and fruit, prune for better airflow, avoid overhead irrigation, and apply a recommended fungicide if necessary."
    },
    "Grape___Esca_(Black_Measles)": {
        "status": "Diseased",
        "advice": "Grape Esca detected. Prune affected wood when possible, sanitize tools, reduce plant stress, and monitor the vine closely for worsening symptoms."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "status": "Diseased",
        "advice": "Grape leaf blight detected. Remove infected leaves, improve airflow, avoid overhead watering, and use appropriate fungicide treatment if needed."
    },
    "Grape___healthy": {
        "status": "Healthy",
        "advice": "The grape leaf appears healthy. Continue normal care and regular monitoring."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "status": "Diseased",
        "advice": "Citrus greening detected. This is a serious disease. Isolate the affected plant if possible, control insect vectors, remove severely affected parts, and consult local agricultural guidance."
    },
    "Peach___Bacterial_spot": {
        "status": "Diseased",
        "advice": "Peach bacterial spot detected. Remove infected leaves when possible, avoid overhead watering, improve airflow, and apply appropriate bacterial disease treatment if recommended."
    },
    "Peach___healthy": {
        "status": "Healthy",
        "advice": "The peach leaf appears healthy. Continue normal care and regular monitoring."
    },
    "Pepper,_bell___Bacterial_spot": {
        "status": "Diseased",
        "advice": "Bell pepper bacterial spot detected. Remove infected leaves, avoid splashing water on foliage, improve airflow, and apply suitable treatment if needed."
    },
    "Pepper,_bell___healthy": {
        "status": "Healthy",
        "advice": "The bell pepper leaf appears healthy. Continue normal care and regular monitoring."
    },
    "Potato___Early_blight": {
        "status": "Diseased",
        "advice": "Potato early blight detected. Remove infected leaves, avoid overhead watering, rotate crops, and apply fungicide if symptoms continue to spread."
    },
    "Potato___Late_blight": {
        "status": "Diseased",
        "advice": "Potato late blight detected. Remove infected foliage immediately, avoid excess moisture, improve airflow, and use an appropriate fungicide as soon as possible."
    },
    "Potato___healthy": {
        "status": "Healthy",
        "advice": "The potato leaf appears healthy. Continue normal care and regular monitoring."
    },
    "Raspberry___healthy": {
        "status": "Healthy",
        "advice": "The raspberry leaf appears healthy. Continue normal care and regular monitoring."
    },
    "Soybean___healthy": {
        "status": "Healthy",
        "advice": "The soybean leaf appears healthy. Continue normal care and regular monitoring."
    },
    "Squash___Powdery_mildew": {
        "status": "Diseased",
        "advice": "Squash powdery mildew detected. Remove affected leaves, reduce humidity, improve spacing and airflow, and apply a suitable fungicide if needed."
    },
    "Strawberry___Leaf_scorch": {
        "status": "Diseased",
        "advice": "Strawberry leaf scorch detected. Remove infected leaves, avoid overhead watering, improve airflow, and monitor the plant closely for further spread."
    },
    "Strawberry___healthy": {
        "status": "Healthy",
        "advice": "The strawberry leaf appears healthy. Continue normal care and regular monitoring."
    },
    "Tomato___Bacterial_spot": {
        "status": "Diseased",
        "advice": "Tomato bacterial spot detected. Remove infected leaves, avoid splashing water on foliage, improve airflow, and apply appropriate treatment if necessary."
    },
    "Tomato___Early_blight": {
        "status": "Diseased",
        "advice": "Tomato early blight detected. Remove infected lower leaves, mulch the soil, avoid overhead watering, and use a recommended fungicide if needed."
    },
    "Tomato___Late_blight": {
        "status": "Diseased",
        "advice": "Tomato late blight detected. Remove infected leaves quickly, avoid excess moisture, improve airflow, and apply fungicide as soon as possible."
    },
    "Tomato___Leaf_Mold": {
        "status": "Diseased",
        "advice": "Tomato leaf mold detected. Reduce humidity, improve airflow, avoid wetting leaves, and remove affected foliage."
    },
    "Tomato___Septoria_leaf_spot": {
        "status": "Diseased",
        "advice": "Tomato Septoria leaf spot detected. Remove infected leaves, avoid overhead watering, improve airflow, and apply fungicide if the infection spreads."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "status": "Diseased",
        "advice": "Tomato spider mite damage detected. Remove badly affected leaves, wash leaves gently if appropriate, reduce plant stress, and use a suitable miticide or pest control method if needed."
    },
    "Tomato___Target_Spot": {
        "status": "Diseased",
        "advice": "Tomato target spot detected. Remove infected leaves, improve airflow, reduce leaf wetness, and apply fungicide if symptoms worsen."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "status": "Diseased",
        "advice": "Tomato yellow leaf curl virus detected. Remove severely affected leaves or plants, control whiteflies, isolate infected plants if possible, and monitor nearby plants closely."
    },
    "Tomato___Tomato_mosaic_virus": {
        "status": "Diseased",
        "advice": "Tomato mosaic virus detected. Remove infected plant material, disinfect tools and hands after handling, avoid touching healthy plants after infected ones, and monitor spread carefully."
    },
    "Tomato___healthy": {
        "status": "Healthy",
        "advice": "The tomato leaf appears healthy. Continue normal care and regular monitoring."
    }
}

def get_plant_advice(class_name: str):
    return ADVICE_MAP.get(
        class_name,
        {
            "status": "Unknown",
            "advice": "Sorry this leaf was not found in our dataset"
        }
    )


@app.get("/")
def root():
    return {"message": "Plant Disease Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("filename:", file.filename)
    print("content_type:", file.content_type)

    image_bytes = await file.read()
    print("bytes length:", len(image_bytes))

    processed_image = preprocess_image(image_bytes)

    predictions = model.predict(processed_image, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    predicted_class = get_class_name(predicted_index)

    advice_info = get_plant_advice(predicted_class)

    return {
        "class_name": predicted_class,
        "status": advice_info["status"],
        "confidence": round(confidence, 4),
        "advice": advice_info["advice"]
    }
