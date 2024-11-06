from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from PIL import Image
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn import MTCNN
import os
import tensorflow.compat.v2 as tf
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # This allows cross-origin requests

# Initialize face detector and model
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
            return None, "No face detected in the image."
        
        x, y, width, height = results[0]['box']
        face = img[y:y + height, x:x + width]

        image = Image.fromarray(face)
        image = image.resize((224, 224))

        face_array = np.asarray(image).astype('float32')
        expanded_img = np.expand_dims(face_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        result = model.predict(preprocessed_img).flatten()
        return result, None
    except Exception as e:
        return None, str(e)

def recommend(feature_list, features):
    try:
        similarity = [cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0][0] for f in feature_list]
        index_pos = sorted(enumerate(similarity), reverse=True, key=lambda x: x[1])[0][0]
        return index_pos
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML page

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    image = Image.open(file.stream)
    features, error = extract_features(image, model, detector)
    if features is None:
        return jsonify({'error': error}), 400

    index_pos = recommend(feature_list, features)
    if index_pos is None:
        return jsonify({'error': 'Recommendation failed'}), 400

    predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
    image_url = f"https://raw.githubusercontent.com/LavKalsi/CelebrityFaceMatchWebsite/main/TrainingData/{'/'.join(filenames[index_pos].split(os.sep)[-2:])}"

    return jsonify({
        'message': f'Seems like {predicted_actor}',
        'image_url': image_url
    })

if __name__ == '__main__':
    app.run(debug=True)
