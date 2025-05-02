from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import traceback
import pandas as pd

import preparation
import xgboost as xgb
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔄 Loading booster...")
booster = xgb.Booster()
booster.load_model("color_type_model.json")
print("✅ Booster loaded!")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    print("🔔 /analyze endpoint HIT")
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        features_dict = preparation.extract_features(file_location)
        if not features_dict:
            return JSONResponse(status_code=400, content={"error": "Face not detected"})

        # Список колонок у правильному порядку
        columns = [
            "skin_H", "skin_S", "skin_V",
            "hair_H", "hair_S", "hair_V",
            "eyes_H", "eyes_S", "eyes_V",
            "contrast_score", "saturation_contrast"
        ]

        features_array = [features_dict[k] for k in columns]
        print("👀 Features:", features_array)

        df = pd.DataFrame([features_array], columns=columns)
        dinput = xgb.DMatrix(df, feature_names=columns) 
        pred_probs = booster.predict(dinput)
        predicted_class = int(np.argmax(pred_probs))

        print("✅ Prediction:", predicted_class)
        return JSONResponse(content={"result": predicted_class})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})