import joblib

# Load model
model = joblib.load("models/voting_model.pkl")

# Load vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")