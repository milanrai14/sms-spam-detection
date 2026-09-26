from model_loader import model, vectorizer

def predict_spam(message: str): 
    #convert text into TF-IDF feature
    X = vectorizer.transform([message])

    #make prediction
    prediction = model.predict(X)[0]

    result = 'Spam' if prediction == 1 else "No Spam"

    return{
        "Prediction": int(prediction), 
        "result": result
    }