import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import streamlit as st

# Download necessary NLTK resources (only needed the first time)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load pre-trained sentiment analysis model
with open(r'ML_Dep0001_Sentiment_Analysis.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open(r'ML_Dep0001_tfidf.pkl', 'rb') as TFV_file:
    tfv = pickle.load(TFV_file)


# Initialize the Lemmatizer
lemma = WordNetLemmatizer()

st.title("Sentiment Analyser")
st.write("Please enter your text below & click on the predict button")

# Text input box
text = st.text_input("Enter text", "write your text here")
predictions = None

# Sentiment prediction button
if st.button("Predict"):
	dff = pd.DataFrame({'text': [text]})
	lemma = WordNetLemmatizer()
	dff['cleaned'] = dff['text'].str.replace('[^a-zA-Z]', ' ', regex=True).str.lower()
	dff['cleaned'] = dff['cleaned'].str.split()
	dff['lemma'] = dff['cleaned'].apply(lambda words: ' '.join([lemma.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]))
	dff.drop('cleaned', axis=1, inplace=True)
	dff_tfidf = tfv.transform(dff['lemma']).toarray()
	predictions = loaded_model.predict(dff_tfidf)    

# Only display the prediction if it's defined
if predictions is not None:
    st.success(f"Predicted Sentiment: {predictions[0]}")
