import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_objects(image_path):
    # Load and resize the image (to 256x256 grayscale)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image_resized  = cv2.resize(gray, (256, 256))  # Resize the grayscale image to 256x256

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image_resized, (5, 5), 0)
    
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
        if cv2.contourArea(contour) > 1000:  # Adjust the threshold as needed
            # Get the convex hull to better capture intricate shapes like forks
            hull = cv2.convexHull(contour)
            x, y, w, h = cv2.boundingRect(hull)  # Get bounding box of the convex hull
            bounding_boxes.append((x, y, w, h))
    
    # Display image with initial bounding boxes before merging
    image_with_initial_boxes = draw_bounding_boxes(cv2.imread(image_path).copy(), bounding_boxes)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_initial_boxes, cv2.COLOR_BGR2RGB))
    plt.title('Initial Bounding Boxes')
    plt.axis('off')
    plt.show()
    
    # Merge close bounding boxes
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

def testDetection(image_path):
    # Detect objects and get bounding boxes
    bounding_boxes = detect_objects(image_path)
    
    # Load the original image for drawing
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error reading image: {image_path}")
        return
    
    # Draw the bounding boxes on the image
    image_with_boxes = draw_bounding_boxes(original_image, bounding_boxes)
    
    # Display the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def main():
    testDetection('images/dataset_detection_original/detected/butterknife_124.jpg')

if __name__ == "__main__":
    main()
