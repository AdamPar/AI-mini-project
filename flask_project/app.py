from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image, ImageOps
import pickle
from tensorflow.keras.models import load_model

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

# Object detection function
def detect_objects(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(gray, (256, 256))
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image_resized, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            hull = cv2.convexHull(contour)
            x, y, w, h = cv2.boundingRect(hull)
            bounding_boxes.append((x, y, w, h))

    # Merge overlapping bounding boxes
    merged_boxes = merge_close_boxes(bounding_boxes, merge_threshold=20)

    # Pass class labels to draw_bounding_boxes
    output_image = draw_bounding_boxes(image_resized, merged_boxes)

    return merged_boxes, output_image

def merge_close_boxes(bounding_boxes, merge_threshold=20):
    merged = []
    for box in bounding_boxes:
        x, y, w, h = box
        new_box = True
        for i, (mx, my, mw, mh) in enumerate(merged):
            if (x < mx + mw + merge_threshold and x + w > mx - merge_threshold and
                y < my + mh + merge_threshold and y + h > my - merge_threshold):
                merged[i] = (min(x, mx), min(y, my), max(x + w, mx + mw) - min(x, mx), max(y + h, my + mh) - min(y, my))
                new_box = False
                break
        if new_box:
            merged.append((x, y, w, h))
    return merged

def draw_bounding_boxes(image, bounding_boxes):
    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

@app.route('/')
def index():
    classes = label_encoder.classes_.tolist()
    return render_template('index.html', classes=classes)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the uploaded image
        preprocessed_image = preprocess_image(file_path)
        preprocessed_image_path = os.path.join(PREPROCESSED_FOLDER, 'preprocessed_' + filename)
        preprocessed_array = (preprocessed_image.squeeze() * 255).astype('uint8')
        Image.fromarray(preprocessed_array).save(preprocessed_image_path)

        # Detect objects and draw bounding boxes
        bounding_boxes, image_with_boxes = detect_objects(file_path)
        processed_image_path = os.path.join(PREPROCESSED_FOLDER, 'detected_' + filename)
        cv2.imwrite(processed_image_path, image_with_boxes)

        # Predict using the model
        prediction = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(prediction)
        detected_class = label_encoder.inverse_transform([predicted_class_index])[0]

        # Return template with paths to display all images
        return render_template(
            'results.html',
            uploaded_image=url_for('static', filename=f'uploads/{filename}'),
            preprocessed_image=url_for('static', filename=f'preprocessed/preprocessed_{filename}'),  # Preprocessed image path
            image_with_boxes=url_for('static', filename=f'preprocessed/detected_{filename}'),  # Image with bounding boxes
            detected_class=detected_class
        )
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
