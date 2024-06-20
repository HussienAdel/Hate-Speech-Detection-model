import numpy as np
from flask import Flask, request, jsonify, render_template
import os
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Convert all the words to lowercase
    tokens = [word.lower() for word in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Stem the words
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)
    lst = []
    lst.append(preprocessed_text)
    return lst


# D:\My_ML_Projects\Hate_Speech_Deployment\myModel.pkl

model = pickle.load(open('myModel.pkl', 'rb')) # Load the trained model
vactorizer = pickle.load(open('vactorizer.pkl', 'rb'))

app = Flask(__name__) # Initialize the flask App

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json() # data = {"text" : data}
     
    # Get data from the request as a string
    #data = request.data.decode('utf-8')

    # Preprocess the text
    preprocessed_text = preprocess_text(data['text'])

    # Vectorize the preprocessed text
    vectorized_text = vactorizer.transform(preprocessed_text).toarray()

    # Make a prediction
    prediction = model.predict_proba(vectorized_text)

    out = None

    if prediction[0][1] > 0.25 :
        out = 1 # if the text is hate speech.
    else :
        out = 0 # if the text is natural.   

    # Return the predictions as JSON
    return jsonify({'predictions': out})

if __name__ == "__main__":
    app.run(debug=True)
