from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()

items = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
async def startup_event():
    items["pipeline"] = joblib.load('model.joblib')


@app.get("/predict")
def predict(text):
    #compute score prediction with our model

    # pipeline = get_model_from_gcp()
    pipeline = items["pipeline"]

    # create data matrix
    X = pd.DataFrame({"reviews": [text]})

    # make prediction
    result = pipeline.predict(X)

    # convert response from numpy to python type
    pred = float(result)
    
    return dict(prediction=pred)

# if __name__ == "__main__":
#     text = "The hotel was actually very bad. I hated it!"
#     predict(text)