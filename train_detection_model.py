import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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
                image = img_to_array(image)

                # Append to the lists
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {image_file}: {e}")
                continue

    return np.array(images), np.array(labels)

# Custom CNN model for object detection
def build_model(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    # Feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)

    # Output: Binary classification (detected or not detected)
    class_output = layers.Dense(1, activation='sigmoid', name='class_output')(x)

    model = models.Model(inputs=inputs, outputs=class_output)
    return model

# Compile and train model
def train_model(model, images, labels, epochs=10, batch_size=32):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels,
        test_size=0.2,
        random_state=42  # For reproducibility
    )

    # Normalize the images
    X_train, X_val = X_train / 255.0, X_val / 255.0

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Save the trained model
    os.makedirs('trained_models', exist_ok=True)
    model_path = os.path.join('trained_models', 'object_detection_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")

    return history, X_val, y_val

# Evaluate model and print confusion matrix and classification report
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_pred_classes = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels (0 or 1)

    # Print the confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred_classes))
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_classes, target_names=['empty', 'detected']))

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

    # Build and train the model
    model = build_model()
    history, X_val, y_val = train_model(model, images, labels, epochs=10, batch_size=32)

    # Evaluate the model
    evaluate_model(model, X_val, y_val)
