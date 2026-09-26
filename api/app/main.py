from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home(): 
    return{
        "message": "Building API for the SMS detection"
    }