from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
from PIL import ImageOps
import pickle

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
PREPROCESSED_FOLDER = 'static/preprocessed'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)

# Load the model
MODEL_PATH = os.path.join('models', 'model_50_epochs.keras')
model = load_model(MODEL_PATH)

# Load the label encoder
LABEL_ENCODER_PATH = os.path.join('models', 'label_encoder.pkl')
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocessing function
def preprocess_image(image_path, target_size=(256, 256)):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("L").resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Mock detection function
def mock_detection(image_path):
    # This function mimics image processing and detection.
    detected_image = image_path  # For now, return the same image.
    detected_class = "Mock Class"  # Replace with actual detection results.
    return detected_image, detected_class

# Routes
@app.route('/')
def index():
    # Dynamically get the classes from the label encoder
    classes = label_encoder.classes_.tolist()
    return render_template('index.html', classes=classes)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file and allowed_file(file.filename):
        # Save the original uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image
        preprocessed_image = preprocess_image(file_path)
        preprocessed_image_path = os.path.join(PREPROCESSED_FOLDER, filename)
        preprocessed_array = (preprocessed_image.squeeze() * 255).astype('uint8')  # Scale back for saving
        Image.fromarray(preprocessed_array).save(preprocessed_image_path)

        # Predict using the model
        prediction = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(prediction)  # Get the class index
        detected_class = label_encoder.inverse_transform([predicted_class_index])[0]  # Decode class label

        # Redirect to results page
        return render_template(
            'results.html',
            uploaded_image=url_for('static', filename=f'uploads/{filename}'),
            detected_image=url_for('static', filename=f'preprocessed/{filename}'),
            detected_class=detected_class
        )
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
