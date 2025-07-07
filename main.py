import pandas as pd
import pickle
from preprocessing import preprocess_email
from features import extract_features
from train import train_model
from predict import classify_email

def main():
    # Step 1: Load and preprocess data
    df = pd.read_csv("emails.csv")
    df['processed'] = df['text'].apply(preprocess_email)

    # Step 2: Extract features and labels
    X, vectorizer = extract_features(df['processed'])
    y = df['label']

    # Step 3: Train model
    model = train_model(X, y)

    # Step 4: Save model and vectorizer
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Model and vectorizer saved as 'model.pkl' and 'vectorizer.pkl'")

    # Step 5: Test prediction
    test_email = "Win a free iPhone now by clicking this link."
    label, confidence = classify_email(model, vectorizer, test_email)
    print(f"Predicted label: {label} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()
