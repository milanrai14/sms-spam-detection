from fastapi import FastAPI
from schemas import MessageRequest, PredictionResponse
from spam_detection import predict_spam
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Spam Detection API", 
    description="Spam detection using Machine learning and NLP"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home(): 
    return{
        "message": "Building API for the SMS detection"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest): 
    return predict_spam(request.message)