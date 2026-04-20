from fastapi import APIRouter
from services.predict_service import predict_position
from schemas.predict_input import PredictInput
from schemas.predict_output import PredictOutput

router = APIRouter()
@router.post("/predict")
def predict(input_data: PredictInput) -> PredictOutput:
    return predict_position(input_data)