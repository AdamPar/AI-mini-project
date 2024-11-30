from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Mock detection function
def mock_detection(image_path):
    # This function mimics image processing and detection.
    detected_image = image_path  # For now, return the same image.
    detected_class = "Mock Class"  # Replace with actual detection results.
    return detected_image, detected_class

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Call the mock detection function
        detected_image, detected_class = mock_detection(file_path)

        # Redirect to results page
        return render_template('results.html', 
                               uploaded_image=url_for('static', filename=f'uploads/{filename}'),
                               detected_image=url_for('static', filename=f'uploads/{filename}'),
                               detected_class=detected_class)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
