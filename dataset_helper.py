from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import shutil



def create_upside_down_images(directory, limit=25):
    """Creates upside-down copies of the first `limit` images in a directory."""
    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Limit to the first `limit` images
    image_files = image_files[:limit]

    # Process each image
    for image_file in image_files:
        # Open the image
        image_path = os.path.join(directory, image_file)
        with Image.open(image_path) as img:
            # Create an upside-down version of the image
            upside_down_img = img.rotate(180)

            # Save the upside-down image with a new name
            new_image_path = os.path.join(directory, f'upsidedown_{image_file}')
            upside_down_img.save(new_image_path)

            print(f'Created upside-down image: {new_image_path}')

    print("Upside-down image creation complete.")


def rename_class_files(base_directory, classes):
    """Renames files in each class directory to follow a consistent naming format."""
    for class_name in classes:
        class_path = os.path.join(base_directory, class_name)

        # Check if the directory exists
        if not os.path.isdir(class_path):
            print(f"Directory for class '{class_name}' not found.")
            continue

        # List all files in the current class directory
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        # Sort the files to ensure a consistent order
        files.sort()

        # Rename the files in the format: class_name_index.jpg
        for index, file in enumerate(files):
            # Check if the file has a valid image extension
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                new_name = f"{class_name}_{index}.jpg"
                old_path = os.path.join(class_path, file)
                new_path = os.path.join(class_path, new_name)

                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed '{file}' to '{new_name}'")

    print("Renaming complete.")


def delete_all_images_from_dataset(base_directory, classes):
    """Deletes all image files from each class directory."""
    for class_name in classes:
        class_path = os.path.join(base_directory, class_name)

        # Check if the directory exists
        if not os.path.isdir(class_path):
            print(f"Directory for class '{class_name}' not found.")
            continue

        # List all files in the current class directory
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        # Delete each image file in the directory
        deleted_files_count = 0
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                file_path = os.path.join(class_path, file)
                os.remove(file_path)
                deleted_files_count += 1

        print(f"Deleted {deleted_files_count} image(s) from class '{class_name}'.")

    print("Image deletion complete.")


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

def process_images_in_folder(input_folder, output_folder, class_name):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Dictionary to track the index for each class
    image_index = {}

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error reading image: {image_file}")
            continue
        
        # Initialize index for the class if not already present
        if class_name not in image_index:
            image_index[class_name] = 0  # Start index from 0 for each class
        
        index = image_index[class_name]
        image_index[class_name] += 1  # Increment the index after processing each image
        
        # Detect objects and get bounding boxes
        bounding_boxes = detect_objects(image_path)  # Replace with your object detection function
        
        if bounding_boxes:
            # Assuming we use the first bounding box if multiple are detected
            x1, y1, w, h = bounding_boxes[0]  # (x, y, w, h)
            x2, y2 = x1 + w, y1 + h  # Bottom-right coordinates
            # Create a new filename with the class name, presence flag, index, and coordinates
            new_filename = f"{class_name}_1_{index}_{x1}_{y1}_{x2}_{y2}.jpg"
            # image_with_boxes = draw_bounding_boxes(image, bounding_boxes)
            image_with_boxes = image  # If no drawing is needed, use the original image
        else:
            # No bounding boxes detected, set coordinates to zero
            new_filename = f"{class_name}_0_{index}_0_0_0_0.jpg"
            image_with_boxes = image  # Use the original image

        # Save the output image
        output_path = os.path.join(output_folder, new_filename)
        cv2.imwrite(output_path, image_with_boxes)
        
        print(f"Processed and saved {new_filename}")


def testDetection(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return
    
    # Detect objects and get bounding boxes
    bounding_boxes = detect_objects(image_path)
    
    # Draw the bounding boxes on the image
    image_with_boxes = draw_bounding_boxes(image, bounding_boxes)
    
    # Display the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axes
    plt.show()

import os
import shutil

def create_directories(base_dir):
    """
    Create 'positives' and 'negatives' directories within the base directory if they don't exist.
    """
    positive_dir = os.path.join(base_dir, 'positives')
    negative_dir = os.path.join(base_dir, 'negatives')

    if not os.path.exists(positive_dir):
        os.makedirs(positive_dir)
        print(f"Created directory: {positive_dir}")
    if not os.path.exists(negative_dir):
        os.makedirs(negative_dir)
        print(f"Created directory: {negative_dir}")

    return positive_dir, negative_dir

import os
import shutil

def move_files_to_directories(source_dir, positive_dir, negative_dir):
    """
    Move images to 'positives' and 'negatives' directories based on their naming convention.
    """
    for filename in os.listdir(source_dir):
        print(f"Checking filename: {filename}")  # Debug statement
        if filename.startswith("object_"):
            # Split the filename to separate the extension
            base_name, ext = os.path.splitext(filename)
            parts = base_name.split('_')
            print(f"Filename parts: {parts}")  # Debug statement to show all parts

            # Ensure the filename has the expected parts (label, index, coordinates)
            if len(parts) >= 7:  # Check the number of parts in the filename
                label = parts[1]
                print(f"Detected object type (parts[1]): {label}")  # Debug statement to show parts[1]

                # Check if it's a positive sample (object detected)
                if label == '1':
                    src_path = os.path.join(source_dir, filename)
                    dest_path = os.path.join(positive_dir, filename)
                    print(f"Moving positive sample from {src_path} to {dest_path}")  # Debug statement
                    try:
                        shutil.move(src_path, dest_path)
                    except Exception as e:
                        print(f"Error moving {filename}: {e}")
                # Check if it's a negative sample (no object detected)
                elif label == '0':
                    src_path = os.path.join(source_dir, filename)
                    dest_path = os.path.join(negative_dir, filename)
                    print(f"Moving negative sample from {src_path} to {dest_path}")  # Debug statement
                    try:
                        shutil.move(src_path, dest_path)
                    except Exception as e:
                        print(f"Error moving {filename}: {e}")
            else:
                print(f"Skipping invalid filename format: {filename}")



def generate_positive_annotations(directory, output_file):
    """
    Generate annotations for positive samples in the dataset.
    """
    with open(output_file, 'w') as f:
        for filename in os.listdir(directory):
            # Print debugging information for filenames
            print(f"Checking filename: {filename}")

            if filename.startswith("object_") and filename.endswith(('.jpg', '.jpeg', '.png')):
                parts = filename.split('_')
                print(f"Filename parts: {parts}")  # Debug statement to show parsed parts

                # Ensure that the filename format is correct and has at least 7 parts
                if len(parts) >= 7:
                    label = parts[1]
                    if label == '1':  # Only include positive samples
                        # Get the coordinates and image path
                        x1 = int(parts[3])
                        y1 = int(parts[4])
                        w = int(parts[5])
                        h = int(parts[6].split('.')[0])  # Remove file extension from the last part

                        image_path = os.path.join(directory, filename)
                        # Write the annotation to the file
                        f.write(f"{image_path} {x1} {y1} {w} {h}\n")
                        print(f"Writing to positives.txt: {image_path} {x1} {y1} {w} {h}")
                else:
                    print(f"Skipping invalid filename format: {filename}")
    print(f"Positive annotations saved to {output_file}")

def generate_negative_list(directory, output_file):
    """
    Generate a list of paths for negative samples in the dataset.
    """
    with open(output_file, 'w') as f:
        for filename in os.listdir(directory):
            print(f"Processing negative sample: {filename}")  # Debug statement
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add additional extensions if needed
                image_path = os.path.join(directory, filename)
                print(f"Writing to negatives.txt: {image_path}")  # Debug statement
                f.write(f"{image_path}\n")
    print(f"Negative list saved to {output_file}")

def process_dataset(source_dir, base_output_dir, pos_output, neg_output):
    """
    Process the dataset to create directories and prepare annotation files.
    """
    # Create positive and negative directories
    # positive_dir, negative_dir = create_directories(base_output_dir)

    # Move files to appropriate directories
    # print("Moving images to respective directories...")
    # move_files_to_directories(source_dir, positive_dir, negative_dir)

    # Generate the positive annotations file
    print("Generating positive annotations...")
    generate_positive_annotations("images/dataset_detection/positives", pos_output)

    # Generate the negative list file
    # print("Generating negative list...")
    # generate_negative_list(negative_dir, neg_output)

# Usage
source_directory = "images/dataset_detection/objects"
base_output_directory = "images/dataset_detection"
positive_output_file = os.path.join(base_output_directory, "positives.txt")
negative_output_file = os.path.join(base_output_directory, "negatives.txt")


# ---------------------- variables:
# Specify the directory containing the images
# directory = 'images/original_classes/empty'

# base_directory = 'images/original_classes'
# classes = ["butterknife", "choppingboard", "fork", "grater", "knife", "ladle", "plate", "roller", "spatula", "spoon"]

# #deletion dir
# dataset_directory = 'images/dataset'

# # apply bounding boxes to images
# input_folder_detected = 'images/dataset_detection_original/objects'
# output_folder_detected = 'images/dataset_detection/objects'
# os.makedirs(output_folder_detected, exist_ok=True)

def main():
    # process_images_in_folder(input_folder_detected, output_folder_detected, 'object')
    process_dataset(source_directory, base_output_directory, positive_output_file, negative_output_file)

    # create_upside_down_images(directory, 25)
    # rename_class_files(base_directory, classes = ["empty"])
    # delete_all_images_from_dataset(dataset_directory, classes)
    # process_images_in_folder(input_folder_empty, output_folder_empty, 'empty', has_object=0)
    # testDetection('images/dataset_detection_original/detected/choppingboard_61.jpg')



if __name__ == "__main__":
    main()




