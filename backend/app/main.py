from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import predict, health, training

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
# app.include_router(training.router)