import os
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
import firebase_admin
from firebase_admin import credentials, firestore

app = FastAPI(title="Smart Pesticide Web & AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

latest_frame_bytes = None

# --- INITIALIZE FIREBASE ADMIN SDK VIA RENDER ENV VARIABLE ---
db = None
try:
    firebase_json_str = os.environ.get("FIREBASE_CREDENTIALS_JSON")
    if firebase_json_str:
        cred_dict = json.loads(firebase_json_str)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("[SUCCESS] Connected to Firebase Firestore database!")
    else:
        print("[WARNING] FIREBASE_CREDENTIALS_JSON environment variable not found.")
except Exception as e:
    print(f"[ERROR] Firebase initialization failed: {e}")

# --- LOAD TRAINED AI MODEL ---
MODEL_PATH = "tomato_disease_model.h5"
try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("[SUCCESS] Loaded tomato_disease_model.h5 into memory!")
except Exception as e:
    print(f"[ERROR] Could not load model file: {e}")
    MODEL = None

CLASS_NAMES = ["early_blight", "healthy", "late_blight"]


def decode_rgb565(raw_bytes: bytes) -> np.ndarray:
    raw_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    byte1 = raw_array[0::2].astype(np.uint16)
    byte2 = raw_array[1::2].astype(np.uint16)
    rgb565 = (byte1 << 8) | byte2

    r = np.round(((rgb565 >> 11) & 0x1F) * 255 / 31).astype(np.uint8)
    g = np.round(((rgb565 >> 5) & 0x3F) * 255 / 63).astype(np.uint8)
    b = np.round((rgb565 & 0x1F) * 255 / 31).astype(np.uint8)

    return np.stack([r, g, b], axis=-1).reshape((120, 160, 3))


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>UI file (index.html) not found.</h1>"


@app.get("/api/status")
def get_status():
    return {"status": "ACTIVE", "hardware": "esp32-leaf.local", "pump": "IDLE"}


@app.get("/api/latest_frame")
def get_latest_frame():
    global latest_frame_bytes
    if latest_frame_bytes is not None:
        return Response(content=latest_frame_bytes, media_type="application/octet-stream")
    return JSONResponse(status_code=404, content={"error": "No image frame captured yet"})


# --- FETCH SPRAY LOGS FROM FIREBASE ---
@app.get("/api/history")
@app.get("/api/get_logs")
def get_logs():
    if not db:
        return []
    try:
        docs = db.collection("spray_logs").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(50).stream()
        logs = []
        for doc in docs:
            data = doc.to_dict()
            if "timestamp" in data and data["timestamp"]:
                data["created_at"] = data["timestamp"].isoformat() if hasattr(data["timestamp"], "isoformat") else str(data["timestamp"])
            logs.append(data)
        return logs
    except Exception as e:
        print(f"[FIREBASE READ ERROR] {e}")
        return []


# --- INGESTION & PREDICTION ENDPOINT ---
@app.post("/api/predict_and_store")
async def predict_and_store(request: Request):
    global latest_frame_bytes
    if MODEL is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded on server"})

    try:
        raw_bytes = await request.body()
        if len(raw_bytes) != 38400:
            return JSONResponse(status_code=400, content={"error": "Invalid frame payload size"})

        latest_frame_bytes = raw_bytes

        # Decode RGB565 & run AI Inference
        rgb_matrix = decode_rgb565(raw_bytes)
        normalized_img = np.expand_dims(rgb_matrix / 255.0, axis=0)

        predictions = MODEL.predict(normalized_img)
        pred_idx = int(np.argmax(predictions[0]))
        disease = CLASS_NAMES[pred_idx]
        confidence = float(np.max(predictions[0]))

        should_spray = disease in ["early_blight", "late_blight"] and confidence >= 0.70
        action_text = "5V SPRAY ACTIVATED" if should_spray else "STANDBY (HEALTHY)"
        disease_title = disease.upper().replace("_", " ")

        # Write log entry to Firebase Firestore collection 'spray_logs'
        if db:
            try:
                db.collection("spray_logs").add({
                    "disease_condition": disease_title,
                    "confidence_score": f"{round(confidence * 100, 1)}%",
                    "action_executed": action_text,
                    "target_hardware": "IRLZ44N MOSFET (GPIO 14)",
                    "timestamp": firestore.SERVER_TIMESTAMP
                })
                print("[SUCCESS] Log saved to Firebase!")
            except Exception as fb_err:
                print(f"[FIREBASE WRITE ERROR] {fb_err}")

        return {
            "disease": disease_title,
            "confidence": round(confidence * 100, 1),
            "trigger_spray": should_spray,
            "command": "PUMP_ON" if should_spray else "PUMP_OFF"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
