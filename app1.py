
import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Classifier")
input_sms = st.text_area("Enter message")

if st.button('Predict'):
    
  transformed_sms = transform_text(input_sms)

  vector_input = tfidf.transform([transformed_sms])

result = model.predict(vector_input)[0]
if result == 1:
        st.header("Spam")
else:
        st.header("Not Spam") 
           