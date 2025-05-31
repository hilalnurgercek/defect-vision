from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import shutil
import os
from datetime import datetime

# --- Setup ---
app = FastAPI()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development (you can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path for log file
log_file = "inspection_log.csv"

# Ensure the log file has headers
if not os.path.exists(log_file):
    df = pd.DataFrame(columns=["timestamp", "filename", "prediction"])
    df.to_csv(log_file, index=False)

# --- Dummy model prediction (replace with real model later) ---
def predict_defect(image_path: str) -> str:
    # Here you will load your model and make prediction
    # For now, this is a dummy prediction
    return "No defects detected ✅"

# --- Prediction Endpoint ---
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Predict using the dummy or real model
    prediction = predict_defect(temp_file_path)

    # Log the prediction
    new_log = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "filename": file.filename,
        "prediction": prediction
    }])
    new_log.to_csv(log_file, mode='a', header=False, index=False)

    # Remove temp file after prediction
    os.remove(temp_file_path)

    return {"prediction": prediction}

# --- Logs Endpoint ---
@app.get("/logs/")
async def get_logs():
    df = pd.read_csv(log_file)
    return df.to_dict(orient="records")

# --- Server Runner ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
