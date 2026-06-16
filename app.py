from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("student_pass_predictor.pkl")

class Student(BaseModel):
    Study_Hours_per_Week: float
    Attendance_Rate: float
    Past_Exam_Scores: float

@app.get("/")
def home():
    return {"message": "Student Pass Predictor API"}

@app.post("/predict")
def predict(
    study_hours: float,
    attendance: float,
    previous_score: float
):
    prediction = model.predict(
        [[study_hours, attendance, previous_score]]
    )[0]

    probability = model.predict_proba(
        [[study_hours, attendance, previous_score]]
    )[0][1]

    return {
        "prediction": "Pass" if prediction == 1 else "Fail",
        "pass_probability": round(probability * 100, 2)
    }