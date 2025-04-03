from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import traceback
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Try loading the model (check if it exists)
try:
    model = joblib.load("./models/tb_xray_model.joblib")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None  # Set to None if model loading fails

# Define expected input schema
class TBInput(BaseModel):
    Patient_ID: str
    Age: int
    Gender: str
    Chest_Pain: str
    Cough_Severity: int
    Breathlessness: int
    Fatigue: int
    Weight_Loss: float
    Fever: str
    Night_Sweats: str
    Sputum_Production: str
    Blood_in_Sputum: str
    Smoking_History: str
    Previous_TB_History: str

# Categorical Encoding Mapping
gender_map = {"Male": 0, "Female": 1}
fever_map = {"Mild": 0, "Moderate": 1, "High": 2}
sputum_map = {"Low": 0, "Medium": 1, "High": 2}
yes_no_map = {"No": 0, "Yes": 1}
smoking_map = {"Never": 0, "Former": 1, "Current": 2}

@app.post("/predict")
def predict(data: TBInput):
    try:
        if model is None:
            raise ValueError("Model not loaded!")

        print("✅ Received Data:", data.dict())  # Debugging

        # Convert input data to model-friendly format
        features = np.array([
            data.Age,
            gender_map.get(data.Gender, -1),
            yes_no_map.get(data.Chest_Pain, -1),
            data.Cough_Severity,
            data.Breathlessness,
            data.Fatigue,
            data.Weight_Loss,
            fever_map.get(data.Fever, -1),
            yes_no_map.get(data.Night_Sweats, -1),
            sputum_map.get(data.Sputum_Production, -1),
            yes_no_map.get(data.Blood_in_Sputum, -1),
            smoking_map.get(data.Smoking_History, -1),
            yes_no_map.get(data.Previous_TB_History, -1)
        ]).reshape(1, -1)

        # Check for invalid values
        if -1 in features:
            print("❌ ERROR: Invalid categorical value detected!")
            raise ValueError("Invalid input detected!")

        # Make prediction
        prediction = model.predict(features)[0]

        # Convert to human-readable format
        result = "Tuberculosis" if prediction == 1 else "Normal"
        
        return {"prediction": result}
    
    except Exception as e:
        print("❌ Error in prediction:", str(e))
        print(traceback.format_exc())  # Show full error traceback
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
