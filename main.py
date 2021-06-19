import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
import glob
import os
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = FastAPI()


class DiabetesInfo(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

class DiabetesInfoFull(DiabetesInfo):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    quantitative: float


@app.get('/models')
async def get_models():
    files_name = glob.glob("backend/models/*.joblib")
    return [f.split('\\')[1] for f in files_name]


@app.post('/training')
async def training(model_name: str, kwargs: dict, data_train: List[DiabetesInfoFull]):
    headers = ['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6']
    data_train = pd.DataFrame([d.dict() for d in data_train])
    
    X = data_train[headers]
    y = data_train['quantitative']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {'LinearRegression': LinearRegression, 'ElasticNet': ElasticNet, 'RandomForestRegressor': RandomForestRegressor}
    
    model_class = models[model_name]
    trained_model = model_class(**kwargs)
    trained_model.fit(X_train, y_train)
    
    model_name = 'backend/models/' + "diabetes_{}.joblib".format(str(trained_model))
    joblib.dump(trained_model, model_name)
    
    y_pred = trained_model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


@app.post('/predict')
async def predict_diabetes_progress(
        model_name: str,
        age: float, sex: float, bmi: float, bp: float, s1: float, 
        s2: float, s3: float, s4: float, s5: float, s6: float
    ):
    # age, sex, body_mass_index, average_blood_pressure, total_serum_cholesterol, low_density_lipoproteins,
    # high_density_lipoproteins, total_cholesterol, possibly_log_of_serum_triglycerides_level, blood_sugar_level

    model = joblib.load('backend/models/{}'.format(model_name))

    model_input_data = np.array([age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]).reshape(1, -1)
    progression = model.predict(model_input_data)
  
  
    return progression[0]


@app.post('/predict_obj')
async def predict_diabetes_progress_1(model_name: str, data: List[DiabetesInfo]):
    # age, sex, body_mass_index, average_blood_pressure, total_serum_cholesterol, low_density_lipoproteins,
    # high_density_lipoproteins, total_cholesterol, possibly_log_of_serum_triglycerides_level, blood_sugar_level

    model = joblib.load('backend/models/{}'.format(model_name))
    model_input_data = pd.DataFrame([d.dict() for d in data])
    
    progression = model.predict(model_input_data)
    return list(progression)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
To cover:
- possible return types: not nympy array, etc
"""