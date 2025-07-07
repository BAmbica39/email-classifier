import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

def preprocess_email(text):
    text = re.sub(r'\W', ' ', text)            # Remove special characters
    text = text.lower()                        # Convert to lowercase
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single letters
    text = re.sub(r'\s+', ' ', text).strip()   # Remove extra spaces

    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in tokens]

    return ' '.join(stemmed)
