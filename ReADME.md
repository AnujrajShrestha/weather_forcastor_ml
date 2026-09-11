# 🌦️ Weather Forecastor ML

A machine learning–powered weather forecasting system that predicts **temperature** and **weather type** from atmospheric and environmental conditions.

The project combines an end-to-end **scikit-learn ML pipeline** with a **FastAPI backend** for serving predictions through a REST API.

## 🚀 Overview

This project solves two machine learning problems:

* 🌡️ **Temperature Prediction** — Regression using `RandomForestRegressor`
* 🌤️ **Weather Type Prediction** — Classification using `RandomForestClassifier`

The trained models are saved as complete preprocessing + model pipelines using Joblib and loaded directly by the FastAPI application.

### Weather Classes

The classification model predicts one of:

* ☀️ Sunny
* 🌧️ Rainy
* ☁️ Cloudy
* ❄️ Snowy

## ✨ Features

* End-to-end machine learning pipelines
* Random Forest regression
* Random Forest classification
* Feature preprocessing with `ColumnTransformer`
* Numerical feature transformation and scaling
* Categorical feature encoding
* Hyperparameter tuning with `RandomizedSearchCV`
* 5-fold cross-validation
* Model evaluation
* Joblib model serialization
* Pydantic input validation
* FastAPI REST API
* Interactive Swagger documentation
* Exploratory data analysis with Pandas, Matplotlib, and Seaborn

## 🧠 Machine Learning Pipeline

The project follows this workflow:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Train / Test Split
   ↓
Preprocessing Pipelines
   ↓
RandomizedSearchCV
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Save Trained Pipelines
   ↓
FastAPI Deployment
```

## 📊 Dataset

The project uses `weather_classification_data.csv`.

The dataset contains **13,200 records and 11 columns**.

| Feature              | Description               |
| -------------------- | ------------------------- |
| Temperature          | Target for regression     |
| Humidity             | Relative humidity         |
| Wind Speed           | Wind speed                |
| Precipitation (%)    | Precipitation percentage  |
| Cloud Cover          | Cloud-cover category      |
| Atmospheric Pressure | Atmospheric pressure      |
| UV Index             | UV index                  |
| Season               | Season of the year        |
| Visibility (km)      | Visibility distance       |
| Location             | Location type             |
| Weather Type         | Target for classification |

### Categorical Values

**Cloud Cover**

* partly cloudy
* clear
* overcast
* cloudy

**Season**

* Winter
* Spring
* Summer
* Autumn

**Location**

* inland
* mountain
* coastal

**Weather Type**

* Rainy
* Cloudy
* Sunny
* Snowy

## 🔧 Preprocessing

The project uses `ColumnTransformer` to apply different transformations to different feature groups.

### Left-Skewed Numerical Features

The following features use:

```text
PowerTransformer(method="yeo-johnson")
        ↓
StandardScaler
```

Features:

* Humidity
* Precipitation (%)
* Atmospheric Pressure

### Right-Skewed Numerical Features

The following features use:

```text
log1p
  ↓
StandardScaler
```

Features:

* Wind Speed
* UV Index
* Visibility (km)

### Categorical Features

Categorical features use:

```text
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

Features:

* Cloud Cover
* Season
* Location

Because preprocessing is included inside the trained pipelines, the same transformations are automatically applied during inference.

## 🤖 Models

### 1. Temperature Regression

Model:

```text
RandomForestRegressor
```

Hyperparameter tuning:

* `RandomizedSearchCV`
* 15 random configurations
* 5-fold cross-validation
* R² scoring
* `random_state=42`
* `n_jobs=-1`

Selected parameters:

| Parameter           | Value |
| ------------------- | ----: |
| `n_estimators`      |   300 |
| `max_depth`         |    10 |
| `min_samples_split` |    10 |
| `min_samples_leaf`  |     1 |

### Performance

| Metric |  Score |
| ------ | -----: |
| R²     | 0.6032 |
| MAE    | 7.8928 |

---

### 2. Weather Type Classification

Model:

```text
RandomForestClassifier
```

Hyperparameter tuning:

* `RandomizedSearchCV`
* 15 random configurations
* 5-fold cross-validation
* Accuracy scoring
* `random_state=42`
* `n_jobs=-1`

Selected parameters:

| Parameter           | Value |
| ------------------- | ----: |
| `n_estimators`      |   200 |
| `max_depth`         |    15 |
| `min_samples_split` |     5 |
| `min_samples_leaf`  |     2 |

### Performance

**Test Accuracy: 90.30%**

| Weather Type | Precision | Recall | F1-Score |
| ------------ | --------: | -----: | -------: |
| Cloudy       |      0.88 |   0.89 |     0.88 |
| Rainy        |      0.90 |   0.89 |     0.89 |
| Snowy        |      0.90 |   0.93 |     0.92 |
| Sunny        |      0.94 |   0.90 |     0.92 |

Overall macro and weighted averages are approximately **0.90**.

## 💾 Saved Models

The trained pipelines are exported using Joblib:

```text
random_forest_regression.pkl
random_forest_classification.pkl
```

Each file contains both the preprocessing steps and trained model, allowing the API to receive the original feature representation directly.

## ⚡ FastAPI Backend

The trained pipelines are served through FastAPI.

### API Endpoints

#### `GET /`

Returns a basic API status message.

#### `POST /predict`

Accepts weather information and returns:

* predicted temperature
* predicted weather type

### Example Request

```json
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
```

### Example Response

```json
{
  "temperature": 14.23,
  "weather_type": "Rainy"
}
```

The actual prediction depends on the trained models and input values.

## 🏗️ Project Structure

```text
weather_forcastor_ml/
│
├── project7.ipynb
├── weather_classification_data.csv
│
├── main.py
│
├── random_forest_regression.pkl
├── random_forest_classification.pkl
│
├── requirements.txt
├── runtime.txt
├── .gitignore
└── README.md
```

## 🛠️ Technologies Used

### Machine Learning

* Python
* NumPy
* Pandas
* Scikit-learn
* Random Forest
* ColumnTransformer
* Pipeline
* PowerTransformer
* StandardScaler
* OneHotEncoder
* SimpleImputer
* RandomizedSearchCV
* Joblib

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Data Analysis & Visualization

* Pandas
* Matplotlib
* Seaborn

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AnujrajShrestha/weather_forcastor_ml.git

cd weather_forcastor_ml
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the API

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 📚 Swagger Documentation

FastAPI provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

You can use the `/docs` interface to send requests directly to the `/predict` endpoint.

## 🔄 API Architecture

```text
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
          ┌─────────┴─────────┐
          ▼                   ▼
 Regression Pipeline   Classification Pipeline
          │                   │
          ▼                   ▼
 Temperature          Weather Type
 Prediction            Prediction
          │                   │
          └─────────┬─────────┘
                    ▼
               JSON Response
```

## 🎯 What This Project Demonstrates

This project demonstrates practical machine learning engineering concepts including:

* Data preprocessing
* Exploratory data analysis
* Feature transformation
* Feature scaling
* Categorical encoding
* Regression
* Classification
* Ensemble learning
* Hyperparameter optimization
* Cross-validation
* Model evaluation
* Model serialization
* REST API development
* ML model deployment architecture

## ⚠️ Note

This project is intended primarily for **learning and demonstrating machine learning workflows**. The temperature model has an R² of approximately 0.60 on the test set, while the weather classification model achieves approximately 90.30% accuracy. These models should not be treated as professional meteorological forecasting systems.

## 👨‍💻 Author

**Anuj Shrestha**

Machine learning models, preprocessing pipelines, model training, hyperparameter tuning, evaluation, model serialization, and FastAPI backend implementation are part of this project.

## ⭐ Support

If you find this project useful for learning or experimentation, consider giving the repository a ⭐ on GitHub.

---

Built with Python, Scikit-learn, and FastAPI.

```