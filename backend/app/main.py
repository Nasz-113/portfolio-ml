from fastapi import FastAPI
from api.routes import predict, health, training

app = FastAPI()

app.include_router(health.router)
app.include_router(predict.router)
# app.include_router(training.router)