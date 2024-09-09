from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import os


detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

# Load feature list and filenames
feature_list = pickle.load(open('embedding.plk', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

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
            
            # Construct local file path
            matched_image_path = filenames[index_pos]
            if os.path.isfile(matched_image_path):
                matched_image = Image.open(matched_image_path)
            else:
                st.error(f"Image file not found: {matched_image_path}")
                matched_image = None

            # Display
            col1, col2 = st.columns(2)

            with col1:
                st.header('Your Image')
                st.image(display_image, use_column_width=True)
            with col2:
                st.header("You Look Like " + predicted_actor)
                if matched_image:
                    st.image(matched_image, width=300)
