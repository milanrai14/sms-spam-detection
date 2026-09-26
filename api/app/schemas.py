from pydantic import BaseModel


class MessageRequest(BaseModel):
    message: str


class PredictionResponse(BaseModel):
    prediction: int
    result: str