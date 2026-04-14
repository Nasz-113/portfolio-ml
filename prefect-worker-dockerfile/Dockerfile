FROM python:3.12-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_orchestration.py ./

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

CMD [ "" ]
