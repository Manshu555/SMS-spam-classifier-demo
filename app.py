import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.set_page_config(page_title="Spam Classifier - Manshu Jaiswal", page_icon="📩", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📩 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>By Manshu Jaiswal</h4>", unsafe_allow_html=True)


input_sms = st.text_area("Enter the message below 👇")

if st.button('Predict'):
    with st.spinner('Analyzing...🔍'):
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.error("🚫 This is **Spam**!")
        else:
            st.success("✅ This is **Not Spam**.")


st.markdown("""
    <hr style="border: 1px solid #f4f4f4;">
    <div style='text-align: center'>
        Made with ❤️ by <b>Manshu Jaiswal</b>
    </div>
""", unsafe_allow_html=True)
