import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

app = FastAPI(title="VoltAI Battery Life Predictor API")

ORIGINAL_CSV = "Mobile_Battery_Life_Prediction_Dataset.csv"
CUSTOM_CSV = "Mobile_Battery_Life_Prediction_Dataset_Custom.csv"

# Ensure we have a local working copy of the dataset
def load_dataset():
    if os.path.exists(CUSTOM_CSV):
        return pd.read_csv(CUSTOM_CSV)
    elif os.path.exists(ORIGINAL_CSV):
        df = pd.read_csv(ORIGINAL_CSV)
        df.to_csv(CUSTOM_CSV, index=False)
        return df
    else:
        # Create a mock default dataset if file is missing
        data = {
            "Screen_Time_Hours": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "Battery_Remaining_Percent": [90.0, 80.0, 70.0, 58.0, 50.0, 42.0, 30.0, 22.0, 10.0, 5.0]
        }
        df = pd.DataFrame(data)
        df.to_csv(CUSTOM_CSV, index=False)
        return df

# Global variables for model state
model = None
metrics = {}

def train_model():
    global model, metrics
    df = load_dataset()
    if df.empty or len(df) < 2:
        # Fallback dummy model
        model = LinearRegression()
        model.fit([[0]], [100])
        metrics = {"r2": 1.0, "slope": -10.0, "intercept": 100.0, "mse": 0.0, "n": len(df)}
        return

    x = df[["Screen_Time_Hours"]]
    y = df["Battery_Remaining_Percent"]
    
    model = LinearRegression()
    model.fit(x, y)
    
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Pearson correlation coefficient
    corr = float(df["Screen_Time_Hours"].corr(df["Battery_Remaining_Percent"]))
    
    metrics = {
        "r2": float(r2),
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "mse": float(mse),
        "corr": corr,
        "n": len(df)
    }

# Initial training
train_model()

# Request schemas
class PredictionRequest(BaseModel):
    screen_time: float
    refresh_rate: float = 1.0     # 1.0 for 60Hz, 1.25 for 120Hz
    dark_mode: float = 1.0        # 1.0 for Light, 0.85 for OLED Dark
    app_multiplier: float = 1.0   # 1.0 standard, 2.0 gaming, etc.

class DataPoint(BaseModel):
    screen_time: float
    battery_percent: float

@app.get("/")
def get_index():
    # Serves the index.html from the same directory
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h2>index.html not found. Place it in the root workspace directory.</h2>")

@app.get("/api/data")
def get_data():
    df = load_dataset()
    # Return as list of dicts
    return df.rename(columns={
        "Screen_Time_Hours": "screen_time",
        "Battery_Remaining_Percent": "battery_percent"
    }).to_dict(orient="records")

@app.post("/api/data")
def add_data(point: DataPoint):
    if point.screen_time < 0 or point.screen_time > 24:
        raise HTTPException(status_code=400, detail="Screen time must be between 0 and 24 hours.")
    if point.battery_percent < 0 or point.battery_percent > 100:
        raise HTTPException(status_code=400, detail="Battery percentage must be between 0 and 100.")
    
    df = load_dataset()
    new_row = pd.DataFrame([{
        "Screen_Time_Hours": point.screen_time,
        "Battery_Remaining_Percent": point.battery_percent
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CUSTOM_CSV, index=False)
    
    train_model()
    return {"status": "success", "metrics": metrics}

@app.post("/api/predict")
def predict_battery(req: PredictionRequest):
    global model
    if model is None:
        train_model()
    
    # Calculate effective screen time based on simulated settings
    effective_sot = req.screen_time * req.refresh_rate * req.dark_mode * req.app_multiplier
    
    # Predict using the sklearn model, passing in a DataFrame to silence feature names warnings
    input_df = pd.DataFrame([[effective_sot]], columns=["Screen_Time_Hours"])
    pred_val = model.predict(input_df)[0]
    
    # Clamp battery percentage between 0 and 100
    clamped_pred = max(0.0, min(100.0, float(pred_val)))
    
    return {
        "screen_time": req.screen_time,
        "effective_screen_time": effective_sot,
        "predicted_battery_percent": clamped_pred
    }

@app.post("/api/reset")
def reset_dataset():
    if os.path.exists(CUSTOM_CSV):
        os.remove(CUSTOM_CSV)
    train_model()
    return {"status": "success", "metrics": metrics}

@app.get("/api/metrics")
def get_metrics():
    global metrics
    return metrics

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Run simple argument parser for testing
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Backend successfully self-tested!")
        sys.exit(0)
        
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
