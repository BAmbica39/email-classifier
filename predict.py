import numpy as np
from preprocessing import preprocess_email

def classify_email(model, vectorizer, raw_email):
    processed = preprocess_email(raw_email)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]
    confidence = np.max(model.predict_proba(features))
    return prediction, confidence
