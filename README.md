# Celebrity Face Match Website

A web-based application that matches your uploaded image to a Bollywood celebrity using VGGFace, MTCNN, and cosine similarity.

## Features

- Upload an image and get a Bollywood celebrity match.
- Detects faces and extracts facial features using VGGFace.
- Compares facial features using cosine similarity.
- Provides recommendations based on facial similarity.

## Screenshots

| Feature            | Screenshot      |
|--------------------|-----------------|
| Preview Of Website        | <img src="https://github.com/LavKalsi/CelebrityFaceMatchWebsite/blob/main/Screenshot/image.png" width="1400" height="720"/>|

> *(Replace `#` with actual image links after uploading screenshots)*

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/celebrity-face-match-website.git
    cd celebrity-face-match-website
    ```

2. Install the required Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application using Streamlit:

    ```bash
    streamlit run app.py
    ```

## Screenshots (To Be Updated)

1. **Home Screen**: Displays the image uploader and instruction text for users.

2. **Face Upload Screen**: Allows users to select and upload an image from their system.

3. **Matching Result**: Shows the uploaded image alongside the matched Bollywood celebrity.

## Usage

1. Open the web app in your browser by running the Streamlit app.
2. Upload your image through the file uploader.
3. The app will automatically detect the face, extract facial features, and compare them to the stored Bollywood celebrity faces.
4. The celebrity match will be displayed along with your uploaded image.

## Built With

- [Streamlit](https://streamlit.io/) - Web framework for building UI.
- [VGGFace](https://github.com/rcmalli/keras-vggface) - Pre-trained model for facial feature extraction.
- [MTCNN](https://github.com/ipazc/mtcnn) - Multi-task Cascaded Convolutional Networks for face detection.
- [scikit-learn](https://scikit-learn.org/stable/) - Used for cosine similarity.

## Contributing

Feel free to contribute to this project by submitting a pull request. Please make sure to follow the standard GitHub flow when contributing.

## Acknowledgments

- Thanks to the VGGFace team for their awesome pre-trained model.
- Streamlit for providing an easy-to-use web app framework.

## Author

Lav Kalsi
