FROM python:3.12-slim

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY flows/ .

ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV PREFECT_API_URL=http://prefect-server:4200/api

CMD [ "python3", "deploy.py" ]
