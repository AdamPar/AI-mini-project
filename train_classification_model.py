import os
import cv2
import numpy as np
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Set dataset path
DATASET_DIR = 'images/dataset'
IMG_SIZE = 256

# Step 1: Load and preprocess data
def load_images():
    images = []
    labels = []
    
    # Traverse each class folder
    for class_name in os.listdir(DATASET_DIR):
        class_dir = os.path.join(DATASET_DIR, class_name)
        
        if os.path.isdir(class_dir):  # Ensure it's a directory
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg'):  # Load only .jpg files
                    filepath = os.path.join(class_dir, filename)
                    
                    # Load image as grayscale
                    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    
                    # Resize the image to IMG_SIZE x IMG_SIZE
                    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                    
                    images.append(image)
                    labels.append(class_name)  # Use folder name as label

    return np.array(images), np.array(labels)

images, labels = load_images()

# Normalize pixel values
images = images / 255.0  # Scale to [0, 1]

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# One-hot encode labels
labels_one_hot = to_categorical(labels_encoded)

# Step 2: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_one_hot, test_size=0.1, random_state=42)

# Reshape images to add channel dimension
X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Step 3: Define CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()


# Step 4: Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# Step 5: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Step 6: Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots a confusion matrix using seaborn heatmap.

    Args:
        y_true (array): True class indices.
        y_pred (array): Predicted class indices.
        class_names (list): List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


plot_confusion_matrix(y_true_classes, y_pred_classes, label_encoder.classes_)
print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))

def plot_history(history):
    # Accuracy plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_history(history)


# Save the entire model
# model.save('trained_models/model_1.h5')
model.save('trained_models/model_1.keras')


# Save the label encoder
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

# model.save('trained_models/model_1', save_format='tf')

