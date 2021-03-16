FROM python:3.8.6-buster

COPY api /api
COPY reviewlution /reviewlution
COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

CMD uvicorn api:app --host=0.0.0.0  --port=${PORT:-5000} # load web server