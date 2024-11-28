import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('trained_models/object_detection_model.keras')

# Preprocess the input image
def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocesses the input image for the model.
    - Loads the image in grayscale
    - Resizes it to the target size
    - Normalizes pixel values to [0, 1]
    - Adds a batch dimension
    """
    image = load_img(image_path, color_mode='grayscale', target_size=target_size)
    image_array = img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Predict the bounding box and class
def predict_bounding_box(model, image_array):
    """
    Predicts the class label and bounding box from the model.
    - Returns the class label (0 or 1) and bounding box coordinates.
    """
    predictions = model.predict(image_array)
    class_prediction = predictions[0]  # First output is the classification
    bbox_prediction = predictions[1]  # Second output is the bounding box

    # Convert class probability to binary label
    class_label = 1 if class_prediction[0] > 0.5 else 0
    return class_label, bbox_prediction[0]

# Visualize the image with the bounding box
def plot_bounding_box(image_path, bbox, class_label, class_names=['empty', 'detected']):
    """
    Visualizes the bounding box on the input image.
    - bbox: [x_min, y_min, x_max, y_max] in normalized coordinates (0 to 1).
    """
    image = load_img(image_path)  # Load the original image (not resized)
    plt.imshow(image, cmap='gray')
    plt.title(f"Class: {class_names[class_label]}")

    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = bbox

    # Scale coordinates to the original image size
    original_width, original_height = image.size
    x_min = int(x_min * original_width)
    x_max = int(x_max * original_width)
    y_min = int(y_min * original_height)
    y_max = int(y_max * original_height)

    # Draw the bounding box
    plt.gca().add_patch(
        plt.Rectangle(
            (x_min, y_min),  # Top-left corner
            x_max - x_min,   # Width
            y_max - y_min,   # Height
            edgecolor='red',
            facecolor='none',
            linewidth=2
        )
    )
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Path to the input image
    image_path = 'path/to/test_image.jpg'

    # Preprocess the image
    image_array = preprocess_image(image_path)

    # Make predictions
    class_label, bbox = predict_bounding_box(model, image_array)

    # Visualize the bounding box
    plot_bounding_box(image_path, bbox, class_label)
