from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from typing import Literal
import pandas as pd
import joblib

app= FastAPI()
model_regression= joblib.load("random_forest_regression.pkl")
model_classification= joblib.load("random_forest_classification.pkl")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class weather_info(BaseModel):
    Humidity: int= Field(...,ge=0,le=100,description="Enter humidity between 0 to 100")
    Precipitation: float= Field(...,ge=0,le=120,description="Enter Precipitation between 0 to 120")
    Atmospheric_Pressure: float= Field(...,ge=980,le=1040,description="Enter Atmospheric Pressure between 980 to 1040")
    Wind_Speed: float= Field(...,ge=0,le=90,description="Enter Wind Speed between 0 to 90")
    UV_index: int= Field(...,ge=0,le=15,description="Enter uv index between 0 to 100")
    Visibility: float= Field(...,ge=0,le=20,description="Enter visibility betwwen 0 to 20")
    Cloud_Cover: Literal['partly cloudy', 'clear', 'overcast', 'cloudy']
    Season: Literal['Winter', 'Spring', 'Summer', 'Autumn']
    Location: Literal['inland', 'mountain', 'coastal']
    
class response(BaseModel):
    temperature: float
    weather_type: Literal['Rainy', 'Cloudy', 'Sunny', 'Snowy']
    
@app.get("/")
def home():
    return {'message': "weather forcastor API"}

@app.post("/predict",response_model= response)
def predict(features: weather_info):
    try:
        row = pd.DataFrame([{
            'Humidity': features.Humidity,
            'Precipitation (%)': features.Precipitation,
            'Atmospheric Pressure': features.Atmospheric_Pressure,
            'Wind Speed': features.Wind_Speed,
            'UV Index': features.UV_Index,
            'Visibility (km)': features.Visibility,
            'Cloud Cover': features.Cloud_Cover,
            'Season': features.Season,
            'Location': features.Location
        }])
        
        prediction1= model_regression.predict(row)[0]
        prediction2= model_classification.predict(row)[0]
        return{
            'temperature': round(float(prediction1),2),
            'weather_type': str(prediction2)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code= 500,
            detail= f"Something went worng from our site {str(e)}"
        )