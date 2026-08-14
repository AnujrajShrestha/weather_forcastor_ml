Weather Forecastor

A machine learning–powered weather forecasting project that predicts temperature and weather type from atmospheric and environmental features.

The project contains two parts:

Machine Learning pipeline — developed in project7.ipynb

FastAPI inference backend — implemented in main.py

Overview

This project solves two related ML tasks using Random Forest models:

Temperature Prediction — regression problem that predicts a continuous temperature value.

Weather Type Prediction — classification problem that predicts one of four weather categories:

Rainy

Cloudy

Sunny

Snowy

The trained pipelines are exported with Joblib and loaded directly by the FastAPI application.

Dataset

The notebook uses weather_classification_data.csv.

The dataset contains 13,200 records and 11 columns:

Feature

Description

Temperature

Target for regression

Humidity

Relative humidity

Wind Speed

Wind speed

Precipitation (%)

Precipitation percentage

Cloud Cover

Cloud-cover category

Atmospheric Pressure

Atmospheric pressure

UV Index

UV index

Season

Season of the year

Visibility (km)

Visibility in kilometers

Location

Location type

Weather Type

Target for classification

The categorical values used by the API are:

Cloud Cover: partly cloudy, clear, overcast, cloudy

Season: Winter, Spring, Summer, Autumn

Location: inland, mountain, coastal

The classification target contains Rainy, Cloudy, Sunny, and Snowy.

Machine Learning Workflow

The notebook follows a complete preprocessing and model-training workflow:

Load the dataset with pandas.

Inspect the dataset and data types.

Remove missing values.

Perform descriptive statistics and exploratory analysis.

Separate regression and classification targets.

Split the data into training and testing sets using a 70/30 split with random_state=42.

Build preprocessing pipelines.

Train Random Forest regression and classification models.

Tune hyperparameters using RandomizedSearchCV.

Evaluate the optimized models.

Save the complete fitted pipelines with Joblib.

The dataset was checked for missing values; the notebook reports no remaining null values after cleaning.

Preprocessing Pipeline

The project uses a ColumnTransformer with separate transformations for different feature groups.

Left-skewed numerical features

The following features use:

PowerTransformer(method="yeo-johnson") → StandardScaler

Humidity

Precipitation (%)

Atmospheric Pressure

Right-skewed numerical features

The following features use:

log1p → StandardScaler

Wind Speed

UV Index

Visibility (km)

Categorical features

The following features use:

SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore")

Cloud Cover

Season

Location

Because preprocessing is included inside the scikit-learn pipelines, the same transformations used during training are automatically applied during inference.

Models

1. Temperature Regression

Model:

RandomForestRegressor

Hyperparameter tuning was performed with RandomizedSearchCV using:

15 random configurations

5-fold cross-validation

R² scoring

random_state=42

n_jobs=-1

The selected parameters were:

Parameter

Value

n_estimators

300

max_depth

10

min_samples_split

10

min_samples_leaf

1

Test-set performance:

R²: 0.6032

MAE: 7.8928

2. Weather Type Classification

Model:

RandomForestClassifier

Hyperparameter tuning used:

15 random configurations

5-fold cross-validation

accuracy scoring

random_state=42

n_jobs=-1

The selected parameters were:

Parameter

Value

n_estimators

200

max_depth

15

min_samples_split

5

min_samples_leaf

2

Test-set performance:

Accuracy: 90.30%

Classification report:

Weather Type

Precision

Recall

F1-score

Cloudy

0.88

0.89

0.88

Rainy

0.90

0.89

0.89

Snowy

0.90

0.93

0.92

Sunny

0.94

0.90

0.92

Overall macro and weighted averages are approximately 0.90.

Saved Models

The notebook saves the best complete pipelines as:

random_forest_regression.pkl
random_forest_classification.pkl

These files contain the preprocessing and trained model, allowing the API to accept the original feature representation without manually reproducing the notebook's transformations.

FastAPI Backend

The API is implemented with FastAPI.

The backend loads both trained pipelines at startup:

model_regression = joblib.load("random_forest_regression.pkl")
model_classification = joblib.load("random_forest_classification.pkl")

The API validates incoming data with Pydantic.

Input validation

Field

Accepted range / values

Humidity

0–100

Precipitation

0–120

Atmospheric_Pressure

980–1040

Wind_Speed

0–90

UV_index

0–15

Visibility

0–20

Cloud_Cover

partly cloudy, clear, overcast, cloudy

Season

Winter, Spring, Summer, Autumn

Location

inland, mountain, coastal

API Endpoints

GET /

Returns a basic API status message.

POST /predict

Accepts weather information and returns both predictions.

Example request:

{
  "Humidity": 73,
  "Precipitation": 82,
  "Atmospheric_Pressure": 1010.82,
  "Wind_Speed": 9.5,
  "UV_index": 2,
  "Visibility": 3.5,
  "Cloud_Cover": "partly cloudy",
  "Season": "Winter",
  "Location": "inland"
}

Example response:

{
  "temperature": 14.23,
  "weather_type": "Rainy"
}

The exact prediction depends on the loaded trained models and input values.

Project Structure

weather-forecastor/
│
├── project7.ipynb
├── weather_classification_data.csv
├── main.py
├── random_forest_regression.pkl
├── random_forest_classification.pkl
└── README.md

Installation

1. Clone the repository

git clone <your-repository-url>
cd weather-forecastor

2. Create a virtual environment

Windows:

python -m venv .venv
.venv\Scripts\activate

macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install fastapi uvicorn pandas numpy scikit-learn joblib matplotlib seaborn

Run the API

Start the FastAPI server with:

uvicorn main:app --reload

The API will normally be available at:

http://127.0.0.1:8000

FastAPI's interactive API documentation is available at:

http://127.0.0.1:8000/docs

You can use /docs to test the /predict endpoint directly from the browser.

API Architecture

User Input
    │
    ▼
FastAPI /predict
    │
    ▼
Pydantic Validation
    │
    ▼
Pandas DataFrame
    │
    ├───────────────┐
    ▼               ▼
Regression       Classification
Pipeline         Pipeline
    │               │
    ▼               ▼
Temperature      Weather Type
Prediction       Prediction
    │               │
    └───────┬───────┘
            ▼
       JSON Response

Technologies Used

Machine Learning

Python

NumPy

pandas

scikit-learn

Random Forest

PowerTransformer

StandardScaler

OneHotEncoder

SimpleImputer

ColumnTransformer

Pipeline

RandomizedSearchCV

Joblib

Backend

FastAPI

Pydantic

Uvicorn

pandas

Joblib

Data Analysis & Visualization

Matplotlib

Seaborn

pandas

Key Features

Temperature prediction using Random Forest regression

Weather-type classification using Random Forest classification

End-to-end scikit-learn preprocessing pipelines

Numerical transformation and scaling

Categorical encoding

Hyperparameter tuning with randomized search

Cross-validation

Pydantic request validation

FastAPI REST endpoint

Serialized production-ready model pipelines

Interactive Swagger API documentation

Notes

The regression and classification models are separate trained pipelines, even though they share the same feature preprocessing strategy. The API runs both models for every /predict request and returns the predicted temperature and weather type together.

The notebook also contains exploratory data analysis and model evaluation steps before exporting the final pipelines.

Author

Anuj Shrestha

Machine Learning models, preprocessing pipelines, model training, hyperparameter tuning, evaluation, model serialization, and FastAPI backend implementation are authored as part of this project.