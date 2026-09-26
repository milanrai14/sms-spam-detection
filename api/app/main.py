from fastapi import FastAPI
from schemas import MessageRequest, PredictionResponse
from spam_detection import predict_spam

app = FastAPI(
    title="Spam Detection API", 
    description="Spam detection using Machine learning and NLP"
)

@app.get("/")
def home(): 
    return{
        "message": "Building API for the SMS detection"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest): 
    return predict_spam(request.message)