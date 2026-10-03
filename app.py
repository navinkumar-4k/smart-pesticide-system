import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from supabase import create_client, Client

app = FastAPI(title="Smart Pesticide Web & AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize Supabase Client (Paste your FULL key here)
SUPABASE_URL = "https://jyrzvrzlisgomugznkfl.supabase.co"
SUPABASE_KEY = "sb_secret_bUVsgS9-qULSg7W8APQACg_3WUfWTU7"

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"[ERROR] Supabase init failed: {e}")
    supabase = None

# 2. Load Trained AI Model
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
    return {"status": "ready", "bytes": 38400}


@app.get("/api/history")
@app.get("/api/get_logs")
def get_logs():
    if not supabase:
        return []
    try:
        response = supabase.table("spray_logs").select("*").order("id", desc=True).limit(50).execute()
        return response.data if response.data is not None else []
    except Exception as e:
        print(f"[DB READ ERROR] {e}")
        return []


@app.post("/api/override/pump")
async def override_pump(request: Request):
    return {"status": "command_received", "pump": "triggered"}


@app.post("/api/predict_and_store")
async def predict_and_store(request: Request):
    if MODEL is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded on server"})

    try:
        raw_bytes = await request.body()
        if len(raw_bytes) != 38400:
            return JSONResponse(status_code=400, content={"error": "Invalid frame payload size"})

        # AI Inference
        rgb_matrix = decode_rgb565(raw_bytes)
        normalized_img = np.expand_dims(rgb_matrix / 255.0, axis=0)

        predictions = MODEL.predict(normalized_img)
        pred_idx = int(np.argmax(predictions[0]))
        disease = CLASS_NAMES[pred_idx]
        confidence = float(np.max(predictions[0]))

        should_spray = disease in ["early_blight", "late_blight"] and confidence >= 0.70
        action_text = "5V SPRAY ACTIVATED" if should_spray else "STANDBY (HEALTHY)"
        disease_title = disease.upper().replace("_", " ")

        # Log to Supabase gracefully without crashing if DB call fails
        if supabase:
            try:
                supabase.table("spray_logs").insert({
                    "disease_condition": disease_title,
                    "confidence_score": f"{round(confidence * 100, 1)}%",
                    "action_executed": action_text,
                    "target_hardware": "IRLZ44N MOSFET (GPIO 14)"
                }).execute()
            except Exception as db_err:
                print(f"[DB WRITE ERROR] {db_err}")

        return {
            "disease": disease_title,
            "confidence": round(confidence * 100, 1),
            "trigger_spray": should_spray,
            "command": "PUMP_ON" if should_spray else "PUMP_OFF"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
