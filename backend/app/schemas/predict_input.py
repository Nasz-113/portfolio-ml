from pydantic import BaseModel

class PredictInput(BaseModel):
    sets_per_match: float
    receives_per_match: float
    blocks_per_match: float
    digs_per_match: float
    attacks_per_match: float