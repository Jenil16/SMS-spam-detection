import streamlit as st
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


def transform_text(text):
    # Lowercasing
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    y = []
    # removing special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    # removing stopwords and punctutation
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title('Email/SMS Spam Classifier')

input_sms = st.text_area("Enter the message")


if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

