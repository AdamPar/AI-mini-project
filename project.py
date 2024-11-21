import cv2
import numpy as np

# from tensorflow.keras.models import load_model
# import pickle

# model = load_model('trained_models/model_1.h5')
# with open('label_encoder.pkl', 'rb') as file:
#     label_encoder = pickle.load(file)

def detect_objects(image_path):
    # Load and resize the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding for better segmentation
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Adjust kernel size as needed
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # bounding_boxes = []
    # for contour in contours:
    #     # Filter out small contours based on area
    #     if cv2.contourArea(contour) > 500:  # Adjust the threshold as needed
    #         x, y, w, h = cv2.boundingRect(contour)
    #         bounding_boxes.append((x, y, w, h))

    bounding_boxes = []
    for contour in contours:
        # Filter out small contours based on area
        if cv2.contourArea(contour) > 500:  # Adjust the threshold as needed
            # Get the convex hull to better capture intricate shapes like forks
            hull = cv2.convexHull(contour)
            x, y, w, h = cv2.boundingRect(hull)  # Get bounding box of the convex hull
            bounding_boxes.append((x, y, w, h))
    
    merged_boxes = merge_close_boxes(bounding_boxes, merge_threshold=20)

    # return image, bounding_boxes
    return image, merged_boxes


def merge_close_boxes(bounding_boxes, merge_threshold=20):
    """
    Merge bounding boxes that are close to each other.
    """
    merged = []
    for box in bounding_boxes:
        x, y, w, h = box
        new_box = True
        for i, (mx, my, mw, mh) in enumerate(merged):
            # Check if boxes overlap or are close
            if (x < mx + mw + merge_threshold and x + w > mx - merge_threshold and
                y < my + mh + merge_threshold and y + h > my - merge_threshold):
                # Merge boxes
                merged[i] = (min(x, mx), min(y, my),
                             max(x+w, mx+mw) - min(x, mx),
                             max(y+h, my+mh) - min(y, my))
                new_box = False
                break
        if new_box:
            merged.append((x, y, w, h))
    return merged


def crop_and_classify(image, bounding_boxes, model):
    IMG_SIZE = 256  # Ensure it matches the model input size
    predictions = []

    for box in bounding_boxes:
        x, y, w, h = box
        # Crop the object
        cropped = image[y:y+h, x:x+w]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))  # Resize to match model input
        normalized = resized / 255.0                     # Normalize
        reshaped = np.expand_dims(normalized, axis=(0, -1))  # Add batch and channel dims
        
        # Predict using the model
        # pred = model.predict(reshaped)
        # predictions.append((box, np.argmax(pred, axis=1)))
    
    # return predictions


def visualize_results(image, predictions, label_encoder):
    for box, pred_class in predictions:
        x, y, w, h = box
        # label = label_encoder.inverse_transform([pred_class])[0]
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Show the image
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_bounding_boxes(image, bounding_boxes):
    for box in bounding_boxes:
        x, y, w, h = box
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Show the image
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = "images/original_classes/spoon.jpg"

# Detect objects and visualize results
image, bounding_boxes = detect_objects(image_path)
visualize_bounding_boxes(image, bounding_boxes)