FROM python:3.12-slim

WORKDIR /usr/src/app

COPY app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV ARTIFACT_URI=models:/random_forest_model_dev/latest

CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]
