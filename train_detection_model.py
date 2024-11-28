import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load data from image directories with limiting data per class
def load_images_and_labels_from_directory(image_dir, max_detected=1000, max_empty=200):
    images = []
    labels = []

    # Iterate through each subdirectory (e.g., "empty", "detected")
    for label_dir in ['detected', 'empty']:
        full_path = os.path.join(image_dir, label_dir)
        if not os.path.isdir(full_path):
            print(f"Directory {full_path} does not exist.")
            continue

        # Set the label based on the directory name
        label = 1 if label_dir == 'detected' else 0

        # List all image files in the subdirectory
        image_files = [f for f in os.listdir(full_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Limit the number of images to the specified max values for each class
        if label == 1:
            image_files = image_files[:max_detected]
        else:
            image_files = image_files[:max_empty]

        for image_file in image_files:
            try:
                # Load the image and preprocess it
                image_path = os.path.join(full_path, image_file)
                image = load_img(image_path, color_mode='grayscale', target_size=(256, 256))
                image = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]

                # Append to the lists
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {image_file}: {e}")
                continue

    return np.array(images), np.array(labels)

# Custom CNN model for object detection
def build_model(input_shape=(256, 256, 1)):
    input_layer = layers.Input(shape=input_shape)
    
    # Feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)

    # Classification head (binary output)
    class_output = layers.Dense(1, activation='sigmoid', name='class_output')(x)
    
    # Bounding box head (4 continuous outputs)
    bbox_output = layers.Dense(4, activation='linear', name='bbox_output')(x)

    # Combine into a single model
    model = models.Model(inputs=input_layer, outputs=[class_output, bbox_output])
    return model


# Compile and train model with class weights
def train_model(model, images, labels, epochs=10, batch_size=32):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=0.2,
        random_state=42  # For reproducibility
    )

    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weights_dict)

    # Compile the model
    model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss={
        'class_output': 'binary_crossentropy',  # Classification loss
        'bbox_output': 'mse'  # Bounding box regression loss
    },
    metrics={
        'class_output': 'accuracy',
        'bbox_output': 'mse'  # Can use other metrics for bounding box
    }
)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights_dict
    )
    
    # Save the trained model
    os.makedirs('trained_models', exist_ok=True)
    model_path = os.path.join('trained_models', 'object_detection_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return history, X_val, y_val

# Evaluate model and print confusion matrix and classification report
def evaluate_model(model, X_val, y_val):
    # Get predictions
    y_pred_class, y_pred_bbox = model.predict(X_val)

    # Convert class probabilities to binary labels (0 or 1)
    y_pred_classes = (y_pred_class > 0.5).astype(int)

    # Print confusion matrix and classification report for classification output
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_classes))

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_classes, target_names=['empty', 'detected']))

    # For bounding box predictions, you can inspect y_pred_bbox
    print("\nSample Bounding Box Predictions:")
    print(y_pred_bbox[:5])  # Print the first 5 bounding box predictions

# Plot some example images
def plot_images(images, labels, class_names=['empty', 'detected'], num_rows=1, num_cols=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_rows * num_cols):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(class_names[labels[i]])
        plt.axis('off')
    plt.show()

# Example usage
if __name__ == "__main__":
    image_dir = "images/dataset_detection"

    # Load images and labels from the directory
    images, labels = load_images_and_labels_from_directory(image_dir)

    # Print out the shape of the arrays
    print(f"Number of images: {len(images)}")
    print(f"Number of labels: {len(labels)}")
    print(f"Shape of images array: {images.shape}")
    print(f"Shape of labels array: {labels.shape}")

    # Plot a few sample images to check data loading
    plot_images(images[:10], labels[:10])

    # Build and train the model
    model = build_model()
    history, X_val, y_val = train_model(model, images, labels, epochs=10, batch_size=32)

    # Evaluate the model
    evaluate_model(model, X_val, y_val)
