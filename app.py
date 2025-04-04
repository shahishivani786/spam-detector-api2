from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

# Load model and vectorizer
model = joblib.load('spam_detection_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize app
app = FastAPI()

# Optional: Allow all CORS (for frontend calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body format
class EmailRequest(BaseModel):
    email_text: str

# API endpoint
@app.post("/predict/")
def predict(request: EmailRequest):
    email_vector = vectorizer.transform([request.email_text])
    prediction = model.predict(email_vector)
    result = "Spam" if prediction == 1 else "Not Spam"
    return {"prediction": result}

# ✅ This part is needed for Render or local run
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)