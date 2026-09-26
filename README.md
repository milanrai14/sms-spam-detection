# 📩 SMS Spam Detection

An end-to-end **SMS Spam Detector** that uses **Natural Language Processing (NLP)** and **Machine Learning** to classify SMS messages as **Spam** or **Not Spam**.

The project has three main parts:

- 🧠 **Model Training** — Jupyter Notebook (data, EDA, NLP preprocessing, model training & evaluation)
- ⚡ **API** — FastAPI (serves the trained model as a REST API)
- 💻 **Frontend** — React (simple UI to enter a message and see the prediction)

---

## 🖼️ Preview

| Spam Prediction | Not Spam Prediction |
|---|---|
| ![Spam example](./screenshots/spam.png) | ![Not spam example](./screenshots/not_spam.png) |

**Try it with:**
```
Congratulations! You have won $1,000. Click here now to claim your prize!
```
→ Predicted as **Spam** ✅

```
Hi my name is Milan Rai
```
→ Predicted as **No Spam** ✅

---

## 📁 Project Structure

```
sms-spam-detection/
│
├── notebook/
│   ├── data/
│   │   └── spam.csv                     # Raw SMS Spam Collection dataset
│   └── spam_detection.ipynb             # Full model training pipeline (top to bottom)
│
├── api/
│   └── app/
│       ├── models/
│       │   ├── spam_classifier.pkl      # Trained ML model (exported from notebook)
│       │   └── vectorizer.pkl           # Fitted TF-IDF vectorizer
│       ├── main.py                      # FastAPI entry point
│       ├── model_loader.py              # Loads model + vectorizer into memory
│       ├── schemas.py                   # Pydantic request/response schemas
│       └── spam_prediction.py           # Prediction logic / route handler
│   ├── requirements.txt
│   └── README.md
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── SmsForm.jsx
│   │   │   └── PredictionResult.jsx
│   │   └── api/
│   │       └── spamApi.js
│   ├── package.json
│   ├── vite.config.js
│   └── .env
│
└── README.md
```

---

## 🧠 1. Model Training (Jupyter Notebook)

Everything related to training lives in `notebook/`, with the raw dataset kept separately inside `notebook/data/`.

```
notebook/
├── data/
│   └── spam.csv
└── spam_detection.ipynb
```

The notebook (`spam_detection.ipynb`) walks through the **entire ML pipeline from top to bottom**:

1. **Import Libraries** — pandas, numpy, scikit-learn, nltk, matplotlib, seaborn
2. **Load Dataset** — reads `data/spam.csv` (SMS Spam Collection Dataset — messages labeled `spam` / `ham`)
3. **Exploratory Data Analysis (EDA)**
   - Class distribution (spam vs. ham)
   - Message length distribution
   - Word clouds for spam vs. ham messages
4. **Text Preprocessing (NLP)**
   - Lowercasing
   - Removing punctuation, numbers & special characters
   - Tokenization
   - Stopword removal
   - Stemming / Lemmatization
5. **Feature Extraction**
   - TF-IDF Vectorization (`TfidfVectorizer`)
6. **Train-Test Split**
7. **Model Training** — comparing multiple algorithms:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest
8. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
9. **Model Selection** — pick the best-performing model
10. **Export Artifacts**
    - Save trained model → `../api/app/models/spam_classifier.pkl`
    - Save vectorizer → `../api/app/models/vectorizer.pkl`
    - Using `pickle` or `joblib`

### Run the notebook

```bash
cd notebook
pip install -r requirements.txt
jupyter notebook spam_detection.ipynb
```

> 💡 After training, copy or export the `.pkl` files directly into `api/app/models/` so the API can load them.

---

## ⚡ 2. API — FastAPI

The `api/` folder is the **root of the backend service**. Inside it, `app/` holds all the actual application code, keeping things modular and easy to scale.

```
api/
└── app/
    ├── models/
    │   ├── spam_classifier.pkl
    │   └── vectorizer.pkl
    ├── main.py
    ├── model_loader.py
    ├── schemas.py
    └── spam_prediction.py
```

| File | Responsibility |
|---|---|
| `main.py` | Creates the FastAPI app, sets up CORS, registers routes |
| `model_loader.py` | Loads `spam_classifier.pkl` and `vectorizer.pkl` once at startup |
| `schemas.py` | Pydantic models for request/response validation |
| `spam_prediction.py` | Core prediction logic — transforms text and returns the label |
| `models/` | Stores the trained model + vectorizer artifacts |

### Setup

```bash
cd api
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:
👉 `http://127.0.0.1:8000`

Interactive API docs (Swagger UI):
👉 `http://127.0.0.1:8000/docs`

### Example Endpoint

**POST** `/predict`

**Request body:**
```json
{
  "message": "Congratulations! You have won $1,000. Click here now to claim your prize!"
}
```

**Response:**
```json
{
  "prediction": "spam",
  "message": "This message appears to be a spam message."
}
```

### Sample code layout

**`schemas.py`**
```python
from pydantic import BaseModel

class SMSRequest(BaseModel):
    message: str

class SMSResponse(BaseModel):
    prediction: str
    message: str
```

**`model_loader.py`**
```python
import pickle
import os

BASE_DIR = os.path.dirname(__file__)

def load_model():
    with open(os.path.join(BASE_DIR, "models", "spam_classifier.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "models", "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer
```

**`spam_prediction.py`**
```python
from app.model_loader import load_model

model, vectorizer = load_model()

def predict_spam(message: str) -> dict:
    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]
    label = "spam" if prediction == 1 else "not spam"
    note = (
        "This message appears to be a spam message."
        if label == "spam"
        else "This message appears to be a legitimate message."
    )
    return {"prediction": label, "message": note}
```

**`main.py`**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import SMSRequest, SMSResponse
from app.spam_prediction import predict_spam

app = FastAPI(title="SMS Spam Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=SMSResponse)
def predict(request: SMSRequest):
    return predict_spam(request.message)
```

---

## 💻 3. Frontend — React

The `frontend/` folder is a separate root-level app that talks to the FastAPI service.

```
frontend/
├── src/
│   ├── App.jsx
│   ├── components/
│   │   ├── SmsForm.jsx
│   │   └── PredictionResult.jsx
│   └── api/
│       └── spamApi.js
├── package.json
├── vite.config.js
└── .env
```

### Setup

```bash
cd frontend
npm install
```

### Configure API URL

Create a `.env` file in `frontend/`:

```
VITE_API_URL=http://127.0.0.1:8000
```

### Run the frontend

```bash
npm run dev
```

The app will be available at:
👉 `http://localhost:5173`

### Features

- Textarea to enter an SMS message
- **Check Message** button — calls the FastAPI `/predict` endpoint
- **Clear** button — resets input and result
- Result card, color-coded:
  - 🔴 **Spam** — e.g. `"Congratulations! You have won $1,000. Click here now to claim your prize!"`
  - 🟢 **No Spam** — e.g. `"Hi my name is Milan Rai"`

### Sample API call (`spamApi.js`)

```javascript
const API_URL = import.meta.env.VITE_API_URL;

export async function checkSpam(message) {
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    throw new Error("Failed to get prediction");
  }

  return response.json();
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model Training | Python, Jupyter Notebook, scikit-learn, NLTK, pandas |
| NLP | Tokenization, Stopword Removal, TF-IDF Vectorization |
| API | FastAPI, Uvicorn, Pydantic, Pickle |
| Frontend | React, Vite, CSS / Tailwind |
| Deployment (optional) | Docker, Render / Railway (API), Vercel / Netlify (frontend) |

---

## 📦 Dataset

This project uses the **SMS Spam Collection Dataset**, a public dataset of 5,500+ SMS messages labeled `spam` or `ham` (legitimate), stored at `notebook/data/spam.csv`.

- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

---

## 🚀 Getting Started (Full Setup)

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/sms-spam-detection.git
cd sms-spam-detection

# 2. Train the model (or use the pre-trained artifacts already in api/app/models)
cd notebook
pip install -r requirements.txt
jupyter notebook spam_detection.ipynb

# 3. Start the API
cd ../api
pip install -r requirements.txt
uvicorn app.main:app --reload

# 4. Start the frontend
cd ../frontend
npm install
npm run dev
```

---

## 🧪 Test Messages

| Message | Expected Prediction |
|---|---|
| `Congratulations! You have won $1,000. Click here now to claim your prize!` | 🔴 Spam |
| `URGENT! Your account has been suspended. Verify now: bit.ly/xyz123` | 🔴 Spam |
| `Hi my name is Milan Rai` | 🟢 No Spam |
| `Are we still meeting at 5pm today?` | 🟢 No Spam |


## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and share it.

---

## 🙌 Acknowledgements

- [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- scikit-learn, NLTK, FastAPI, and React communities
