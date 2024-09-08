from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from mtcnn import MTCNN
import requests
from io import BytesIO
import os

# CSS styling for Streamlit app
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
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
    img {
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize face detector and model
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

# Load feature list and filenames
feature_list = pickle.load(open('embedding.plk', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Initialize lists to store uploaded images and predicted celebrity images
uploaded_images = []
predicted_celebrity_images = []

def extract_features(image, model, detector):
    try:
        img = np.array(image)
        results = detector.detect_faces(img)

        if not results:
            st.error("No face detected in the image.")
            return None

        x, y, width, height = results[0]['box']
        face = img[y:y + height, x:x + width]

        # Extract its features
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

def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Error downloading image from URL: {e}")
        return None

st.title('Celebrity Face Match Website')

uploaded_image = st.file_uploader('Choose an image')

if uploaded_image is not None:
    # Load the image
    image = Image.open(uploaded_image)
    display_image = image
    
    # Append uploaded image to list
    uploaded_images.append(display_image)
    
    # Extract the features
    features = extract_features(image, model, detector)
    if features is not None:
        # Recommend
        index_pos = recommend(feature_list, features)
        if index_pos is not None:
            # Prepare URL
            image_url = f"https://raw.githubusercontent.com/LavKalsi/CelebrityFaceMatchWebsite/main/TrainingData/{'/'.join(filenames[index_pos].split(os.sep)[-2:])}"

            # Load image from URL
            result_image = load_image_from_url(image_url)
            if result_image:
                # Append predicted celebrity image to list
                predicted_celebrity_images.append(result_image)

                predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
                # Display
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('<div class="stHeader">Your Uploaded Image</div>', unsafe_allow_html=True)
                    st.image(display_image, width=300, use_column_width=True, caption="Uploaded Image")
                with col2:
                    st.markdown(f'<div class="stHeader">Seems like {predicted_actor}</div>', unsafe_allow_html=True)
                    st.image(result_image, width=300, caption="Predicted Celebrity")

# Optionally, you can display the stored images later on
if uploaded_images:
    st.markdown('<div class="stHeader">History of Uploaded Images</div>', unsafe_allow_html=True)
    for i, img in enumerate(uploaded_images):
        st.image(img, width=100, caption=f"Uploaded Image {i+1}")

if predicted_celebrity_images:
    st.markdown('<div class="stHeader">History of Predicted Celebrity Images</div>', unsafe_allow_html=True)
    for i, img in enumerate(predicted_celebrity_images):
        st.image(img, width=100, caption=f"Predicted Celebrity {i+1}")
