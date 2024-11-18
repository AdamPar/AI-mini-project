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
import matplotlib.pyplot as plt


# Set dataset path
DATASET_DIR = 'images/dataset'
IMG_SIZE = 128  # Already the correct size

# Step 1: Load and preprocess data
def load_images():
    images = []
    labels = []
    
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith('.jpg'):
            # Extract label from filename
            label = filename.split('_')[0]
            
            # Load image using OpenCV
            filepath = os.path.join(DATASET_DIR, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            
            images.append(image)
            labels.append(label)

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
X_train, X_test, y_train, y_test = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)

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
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# def create_model():
#     model = Sequential([
#         # First Convolutional Block
#         Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         # Second Convolutional Block
#         Conv2D(64, (3, 3), activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         # Third Convolutional Block
#         Conv2D(128, (3, 3), activation='relu'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
        
#         # Flatten and Fully Connected Layers
#         Flatten(),
#         Dense(256, activation='relu'),
#         Dropout(0.5),
#         Dense(len(label_encoder.classes_), activation='softmax')
#     ])
    
#     # Compile the model with a lower learning rate
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

model = create_model()


# Step 4: Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32
)

# Step 5: Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

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
model.save('path_to_save_model/my_cnn_model.h5')


# Save the label encoder
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

# model.save('path_to_save_model/my_cnn_model', save_format='tf')
