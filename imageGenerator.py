import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
from PIL import Image, ImageOps  # Added ImageOps for padding

def augment_images(input_dir, output_dir, class_name, num_augmentations, augment_params):
    """
    Augment all images in the input directory and save the augmented images.

    Args:
        input_dir (str): Path to the directory containing input images for a specific class.
        output_dir (str): Directory to save augmented images.
        class_name (str): Class name for the images.
        num_augmentations (int): Number of augmented images to generate per original image.
        augment_params (dict): Dictionary of augmentation parameters for ImageDataGenerator.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Initialize ImageDataGenerator with the specified parameters
    datagen = ImageDataGenerator(
        **augment_params,
        fill_mode='nearest',  # Change to 'constant', 'reflect', or 'wrap' if desired
        cval=128  # Set a value that matches the background (e.g., mid-gray for grayscale)
    )

    # Counter for the number of augmented images generated
    total_images_generated = 0

    for image_file in image_files:
        # Load the image
        image_path = os.path.join(input_dir, image_file)
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        # Add padding to the image
        padding = 50  # Add 50-pixel padding on all sides
        padded_image = ImageOps.expand(image, border=padding, fill=128)  # Use gray padding (value=128)

        # Resize the padded image to 256x256 pixels
        resized_image = padded_image.resize((256, 256))

        # Convert the image to an array for augmentation
        image_array = img_to_array(resized_image)
        image_array = image_array.reshape((1,) + image_array.shape)  # Reshape for data generator

        # Generate augmented images
        counter = 0
        for batch in datagen.flow(image_array, batch_size=1):
            augmented_image = batch[0].astype(np.uint8)
            augmented_image = np.squeeze(augmented_image)  # Remove the extra channel for grayscale images

            # Generate a unique filename for each augmented image
            save_path = os.path.join(output_dir, f"{class_name}_{total_images_generated}.jpg")
            Image.fromarray(augmented_image).save(save_path)

            counter += 1
            total_images_generated += 1
            if counter >= num_augmentations:
                break

            # Stop when we reach 1000 images for the class
            if total_images_generated >= 1000:
                print(f"Reached 1000 augmented images for class '{class_name}'.")
                break

        if total_images_generated >= 1000:
            break

    print(f"Successfully created {total_images_generated} augmented images for class '{class_name}' in '{output_dir}'.")

if __name__ == "__main__":
    # Define the base directory containing subdirectories for each class
    input_base_dir = "images/original_classes"  # Base directory containing subdirectories for each class
    output_base_dir = "images/dataset"  # Directory where augmented images will be saved

    # Augmentation parameters
    augmentation_parameters = {
        "rotation_range": 40,  # Rotate images by up to 40 degrees
        "width_shift_range": 0.2,  # Shift image horizontally by up to 20%
        "height_shift_range": 0.2,  # Shift image vertically by up to 20%
        "shear_range": 0.2,  # Shear intensity (in degrees)
        "zoom_range": 0.2,  # Zoom image by up to 20%
        "horizontal_flip": True,  # Randomly flip images horizontally
        "vertical_flip": False,  # Do not flip images vertically
        "brightness_range": (0.5, 1.5),  # Random brightness adjustment between 0.5 and 1.5
    }

    # Iterate through each subdirectory in the input directory
    for class_name in os.listdir(input_base_dir):
        class_path = os.path.join(input_base_dir, class_name)
        if os.path.isdir(class_path):  # Ensure it is a directory (representing a class)
            output_dir = os.path.join(output_base_dir, class_name)  # Path to the output directory for the class

            # Call the function to augment images
            augment_images(
                input_dir=class_path,
                output_dir=output_dir,
                class_name=class_name,
                num_augmentations=20,  # Generate 20 augmented images per original image
                augment_params=augmentation_parameters
            )
