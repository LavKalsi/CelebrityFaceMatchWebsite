from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from mtcnn import MTCNN
import io
import requests

st.markdown("""
    <style>
    .stApp {
        font-family: 'Arial', sans-serif;
        padding-top: 0px;
        padding-left: 0px;
        padding-right: 0px;
        margin: 0px;
    }
    /* From Uiverse.io by mi-series */ 
    .st-emotion-cache-1vt4y43 {
    width: 150px;
    padding: 0;
    border: none;
    transform: rotate(5deg);
    transform-origin: center;
    font-family: "Gochi Hand", cursive;
    text-decoration: none;
    font-size: 15px;
    cursor: pointer;
    padding-bottom: 3px;
    border-radius: 5px;
    box-shadow: 0 2px 0 #494a4b;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    background-color: #5cdb95;
    }

    .st-emotion-cache-1vt4y43 span {
    background: #f1f5f8;
    display: block;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    border: 2px solid #494a4b;
    }

    .st-emotion-cache-1vt4y43:active {
    transform: translateY(5px);
    padding-bottom: 0px;
    outline: 0;
    }

    .block-container {
        padding-top: 4rem;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
        margin:0px;
        }
    body {
        margin: 0;
        padding: 0;
    }
    .stTitle {
        color: #2C3E50;
        text-align: center;
        font-weight: bold;
        font-size: 2.5em;
    }
    .stHeader {
        color: #3498DB;
        font-weight: bold;
    }
    .stImage {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

# Load feature list and filenames
feature_list = pickle.load(open('embedding.plk', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Define GitHub URL base
GITHUB_URL_BASE = "https://raw.githubusercontent.com/LavKalsi/CelebrityFaceMatchWebsite/main/TrainingData/"

def extract_features(image, model, detector):
    try:
        img = np.array(image)
        results = detector.detect_faces(img)

        if not results:
            st.error("No face detected in the image.")
            return None

        x, y, width, height = results[0]['box']
        face = img[y:y + height, x:x + width]

        # extract its features
        image = Image.fromarray(face)
        image = image.resize((224, 224))

        face_array = np.asarray(image).astype('float32')
        expanded_img = np.expand_dims(face_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        result = model.predict(preprocessed_img).flatten()
        return result
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def recommend(feature_list, features):
    try:
        similarity = [cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0][0] for f in feature_list]
        index_pos = sorted(enumerate(similarity), reverse=True, key=lambda x: x[1])[0][0]
        return index_pos
    except Exception as e:
        st.error(f"Error calculating similarity: {e}")
        return None

def download_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            st.error(f"Failed to download image from {url}")
            return None
    except Exception as e:
        st.error(f"Error downloading image: {e}")
        return None

st.title('Celebrity Face Match Website')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # Load the image
    image = Image.open(uploaded_image)
    display_image = image

    # Extract the features
    features = extract_features(image, model, detector)
    if features is not None:
        # Recommend
        index_pos = recommend(feature_list, features)
        if index_pos is not None:
            predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
            
            # Construct the URL for the matched image
            image_url = GITHUB_URL_BASE + filenames[index_pos].replace('\\', '/').split('/')[-1]
            matched_image = download_image(image_url)

            # Display
            col1, col2 = st.columns(2)

            with col1:
                st.header('Your Image')
                st.image(display_image, use_column_width=True)
            with col2:
                st.header("You Look Like " + predicted_actor)
                if matched_image:
                    st.image(matched_image, width=300)
