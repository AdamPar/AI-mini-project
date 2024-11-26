import os
import cv2
import numpy as np

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

    return merged_boxes


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


def draw_bounding_boxes(image, bounding_boxes):
    """
    Draw bounding boxes on the image.
    """
    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image


def process_and_save_images(input_dir, output_dir):
    """
    Detect objects in images and save processed images with bounding boxes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    classes = ["butterknife", "choppingboard", "fork", "grater", "knife",
               "ladle", "plate", "roller", "spatula", "spoon"]
    
    for class_name in classes:
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        
        for file_name in os.listdir(input_class_dir):
            input_image_path = os.path.join(input_class_dir, file_name)
            output_image_path = os.path.join(output_class_dir, file_name)
            
            # Read the image
            image = cv2.imread(input_image_path)
            if image is None:
                print(f"Error reading image: {input_image_path}")
                continue
            
            # Detect objects and get bounding boxes
            bounding_boxes = detect_objects(input_image_path)
            
            # Draw bounding boxes on the image
            processed_image = draw_bounding_boxes(image, bounding_boxes)
            
            # Save the processed image
            cv2.imwrite(output_image_path, processed_image)
            print(f"Processed and saved: {output_image_path}")


if __name__ == "__main__":
    input_directory = "images/dataset"
    output_directory = "images/bb_dataset"
    
    process_and_save_images(input_directory, output_directory)
