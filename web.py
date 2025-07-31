import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # CORRECTED: Changed 'PorterStemter' to 'PorterStemmer'
import string

# --- NLTK Downloads (Optimized) ---
# Check if NLTK resources are already downloaded to avoid repeated downloads
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.word_tokenize("test")
except LookupError:
    nltk.download('punkt')

# Initialize stemmer and stopwords
ps = PorterStemmer() # CORRECTED: Changed 'PorterStemter' to 'PorterStemmer'
stop_words = set(stopwords.words('english'))

# --- Text Preprocessing Function ---
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
        # Check if the word is not a stop word and not an empty string
        if i not in stop_words and i != '':
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# --- Load Vectorizer and Model (with enhanced error handling and caching) ---
@st.cache_resource # Cache resource to avoid reloading model/vectorizer on every rerun
def load_resources():
    try:
        tfidf_model = pickle.load(open('vectorizer.pkl', 'rb'))
        spam_model = pickle.load(open('model.pkl', 'rb'))
        return tfidf_model, spam_model
    except FileNotFoundError:
        st.error("❌ Model or vectorizer files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ An error occurred while loading resources: {e}")
        st.stop()

tfidf, model = load_resources()


# --- Streamlit UI Enhancements ---
st.set_page_config(
    page_title="Intelligent Spam Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextArea label {
        font-size: 1.2em;
        font-weight: bold;
    }
    h1 {
        color: #FF6347; /* Tomato color for main title */
        text-align: center;
    }
    .result-spam {
        color: #DC143C; /* Crimson */
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        background-color: #FFF0F5; /* LavenderBlush */
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #DC143C;
    }
    .result-not-spam {
        color: #228B22; /* ForestGreen */
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        background-color: #F0FFF0; /* Honeydew */
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #228B22;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📧 Intelligent Spam Detector")
st.markdown("---") # Visual separator

st.write("Welcome! This tool helps you identify whether an email or SMS message is spam or not.")
st.write("Simply paste your message into the text box below and click 'Analyze'.")

input_sms = st.text_area("✍️ Enter your message here:", height=200, placeholder="Type or paste your message (e.g., 'Free money now!!!')")

if st.button('🔍 Analyze Message'):
    with st.spinner('Analyzing your message...'):
        if not input_sms.strip(): # Check for empty or whitespace-only input
            st.warning("⚠️ Oop! Please enter a message to analyze its spam potential.")
        else:
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)

            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])

            # 3. Predict
            result = model.predict(vector_input)[0]

            # 4. Display (Enhanced Styling)
            st.markdown("---")
            if result == 1:
                st.markdown('<div class="result-spam">🚨 SPAM ALERT!</div>', unsafe_allow_html=True)
                st.warning("This message is highly likely to be spam. Exercise caution!")
            else:
                st.markdown('<div class="result-not-spam">✅ NOT SPAM</div>', unsafe_allow_html=True)
                st.success("This message appears to be legitimate.")

st.markdown("---")
st.markdown("Coded with ❤️ for a safer inbox. | Created by Prince Mishra")
st.markdown("---")