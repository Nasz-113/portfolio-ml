# mlopslearn
Learning full process of MLOps using various open source tools.


## TASK:
1. Create pipeline for combining preprocessing with prediction model and load label encoding to the predict_service.
2. Training_service to link with pipeline.
3. Transfer data to pgsql db.
4. Add different machine learning models to open the possibility to select which model to run at the endpoint.
5. Generate Dockerfile for both backend and pipeline, and generate docker-compose.ymal for deploying mlflow, postgresql and prefect server while connecting them with backend and pipeline


### To run this whole backend, you need to run these:

1. MlFlow:              mlflow server (in pipeline/flows)
2. Prefect Server:      prefect server start (in pipeline/flows)
3. Prefect Deployment:  python3 deploy.py (in pipeline/flows)
4. FastAPI:             uvicorn --host 0.0.0.0 --port 8000 main:app