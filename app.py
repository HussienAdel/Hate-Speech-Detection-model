import numpy as np
from flask import Flask, request, jsonify
import os
import pickle
import logging
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

def preprocess_text(text):
    logging.debug("Preprocessing text: %s", text)
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Convert all the words to lowercase
    tokens = [word.lower() for word in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub('user', ' ', text)
    text = re.sub('rt', ' ', text)

    # Stem the words
    stemmer = PorterStemmer()
    stemmed_text = stemmer.stem(text)
    
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lemmatized_text = lemmatizer.lemmatize(stemmed_text)

    preprocessed_text = lemmatized_text
    logging.debug("Preprocessed text: %s", preprocessed_text)
    return [preprocessed_text]

# Initialize the flask App
app = Flask(__name__)

# Load the trained model and vectorizer
try:
    model_path = 'myModel.pkl'
    vectorizer_path = 'vactorizer.pkl'

    if not os.path.isfile(model_path) or not os.path.isfile(vectorizer_path):
        raise FileNotFoundError(f"Model or vectorizer file not found: {model_path}, {vectorizer_path}")

    model = pickle.load(open(model_path, 'rb'))
    vactorizer = pickle.load(open(vectorizer_path, 'rb'))
    logging.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error("Error loading model or vectorizer: %s", e)
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.debug("Received request: %s", request.data)
        # Get data from the request
        data = request.get_json()
        if not data or 'text' not in data:
            raise ValueError("No text data provided in the request")

        # Preprocess the text
        preprocessed_text = preprocess_text(data['text'])

        # Vectorize the preprocessed text
        vectorized_text = vactorizer.transform(preprocessed_text).toarray()

        # Make a prediction
        prediction = model.predict_proba(vectorized_text)

        out = 1 if prediction[0][1] > 0.25 else 0

        # Return the predictions as JSON
        logging.debug("Prediction: %d", out)
        return jsonify({'predictions': out})
    except Exception as e:
        logging.error("Error in prediction: %s", e)
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
