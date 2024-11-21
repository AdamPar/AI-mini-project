import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

model = load_model('trained_models/model_1.h5')
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

def detect_objects(image_path):
    # Load and resize the image (to 256x256 grayscale)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(gray, (256, 256))  # Resize the grayscale image to 256x256

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Use adaptive thresholding for better segmentation
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Adjust kernel size as needed
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for contour in contours:
        # Filter out small contours based on area
        if cv2.contourArea(contour) > 500:  # Adjust the threshold as needed
            # Get the convex hull to better capture intricate shapes like forks
            hull = cv2.convexHull(contour)
            x, y, w, h = cv2.boundingRect(hull)  # Get bounding box of the convex hull
            bounding_boxes.append((x, y, w, h))
    
    merged_boxes = merge_close_boxes(bounding_boxes, merge_threshold=20)

    # Return the grayscale image (already resized) and the bounding boxes
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
        resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))  # Resize to match model input size
        normalized = resized / 255.0  # Normalize the image (convert to range 0-1)
        reshaped = np.expand_dims(normalized, axis=(0, -1))  # Add batch and channel dims (e.g., (1, 256, 256, 1))

        # Predict using the model
        pred = model.predict(reshaped)
        predictions.append((box, np.argmax(pred, axis=1)))
    
    return predictions



def visualize_results(image, predictions, label_encoder):
    for box, pred_class in predictions:
        x, y, w, h = box
        label = label_encoder.inverse_transform([pred_class])[0]
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Set text position inside the bounding box
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_w, text_h = text_size
        text_x = x + (w - text_w) // 2  # Center the text horizontally within the box
        text_y = y + (h + text_h) // 2  # Center the text vertically within the box

        # Put label inside the bounding box
        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the image
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_and_predict_image(image_path, model, label_encoder):
    # Step 1: Detect objects and get bounding boxes
    image, bounding_boxes = detect_objects(image_path)
    
    # Step 2: Classify objects using the pre-trained model
    predictions = crop_and_classify(image, bounding_boxes, model)
    
    # Step 3: Visualize the detection and classification results
    visualize_results(image, predictions, label_encoder)


def visualize_bounding_boxes(image, bounding_boxes):
    for box in bounding_boxes:
        x, y, w, h = box
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Show the image
    # cv2.imshow("Detected Objects", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def detectionTest():

    # List to store processed images
    processed_images = []

    # Loop through the image list and process each image
    for image_name in image_names:
        image_path = f"images/original_classes/{image_name}.jpg"
        
        # Detect objects and get bounding boxes
        image, bounding_boxes = detect_objects(image_path)
        
        # Visualize bounding boxes on the image
        visualize_bounding_boxes(image, bounding_boxes)
        
        # Add processed image to the list
        processed_images.append(image)

    stacked_image = np.hstack(processed_images)  # Stack images vertically

    # Display the stacked image
    cv2.imshow("Detected Objects for All Images", stacked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# global image names
image_names = [
    "butterknife", "choppingboard", "fork", "grater", "knife",
    "ladle", "plate", "roller", "spatula", "spoon"]


if __name__ == "__main__":

    image_dir = "images/original_classes/"

    # Loop through each image name and process the corresponding image
    for image_name in image_names:
        image_path = f"{image_dir}{image_name}.jpg"  # Construct the full image path
        
        # Process the image, run prediction, and visualize results
        process_and_predict_image(image_path, model, label_encoder)