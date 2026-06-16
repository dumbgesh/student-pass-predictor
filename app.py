from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("student_pass_predictor.pkl")

# HTML templates
templates = Jinja2Templates(directory="templates")


class StudentInput(BaseModel):
    study_hours: float
    attendance: float
    previous_score: float


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
def predict(data: StudentInput):

    prediction = model.predict([
        [
            data.study_hours,
            data.attendance,
            data.previous_score
        ]
    ])[0]

    probability = model.predict_proba([
        [
            data.study_hours,
            data.attendance,
            data.previous_score
        ]
    ])[0][1]

    return {
        "prediction": "Pass" if prediction == 1 else "Fail",
        "pass_probability": float(probability)
    }