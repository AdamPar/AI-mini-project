import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
from PIL import Image

# first generate images then convert them to grayscale 128x128, 

def augment_images(input_image_path, output_dir, class_name, num_images, augment_params):
    """
    Augment an image using specified parameters and save the augmented images.

    Args:
        input_image_path (str): Path to the input image.
        output_dir (str): Directory to save augmented images.
        class_name (str): Class name for the image.
        num_images (int): Number of augmented images to generate.
        augment_params (dict): Dictionary of augmentation parameters for ImageDataGenerator.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the input image
    image = load_img(input_image_path)
    image_array = img_to_array(image)
    image_array = image_array.reshape((1,) + image_array.shape)  # Reshape for data generator

    # Initialize ImageDataGenerator with the specified parameters
    datagen = ImageDataGenerator(**augment_params)

    # Generate augmented images
    counter = 0
    for batch in datagen.flow(image_array, batch_size=1):
        # Save each generated image manually with a custom name
        augmented_image = batch[0].astype(np.uint8)

        pil_image = Image.fromarray(augmented_image)
        pil_image = pil_image.convert("L")  # Convert to grayscale
        pil_image = pil_image.resize((128, 128))  # Resize to 128x128

        save_path = os.path.join(output_dir, f"{class_name}_{counter}.jpg")
        pil_image.save(save_path)
        counter += 1
        if counter >= num_images:
            break

    print(f"Successfully created {num_images} augmented images in {output_dir}.")

if __name__ == "__main__":
    # Define parameters directly in the script
    input_image = "images/original_classes/spoon.jpg"  # Path to your input image
    output_dir = "images/dataset"  # Directory where augmented images will be saved
    class_name = "spoon"  # Replace with the class name (e.g., 'cat', 'dog')
    num_images = 100  # Number of augmented images to generate

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

    # Call the function to augment images
    augment_images(
        input_image_path=input_image,
        output_dir=output_dir,
        class_name=class_name,
        num_images=num_images,
        augment_params=augmentation_parameters
    )
