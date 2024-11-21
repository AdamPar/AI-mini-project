import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import numpy as np
from PIL import Image

# first convert image to grayscale 256x256, then generate images

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

    # Load the input image, convert to grayscale, and resize to 256x256 pixels
    image = Image.open(input_image_path).convert("L").resize((256, 256))
    image_array = img_to_array(image)
    image_array = image_array.reshape((1,) + image_array.shape)  # Reshape for data generator

    # Initialize ImageDataGenerator with the specified parameters
    datagen = ImageDataGenerator(**augment_params)

    # Generate augmented images
    counter = 0
    for batch in datagen.flow(image_array, batch_size=1):
        # Save each generated image manually with a custom name
        augmented_image = batch[0].astype(np.uint8)
        augmented_image = np.squeeze(augmented_image)  # Remove the extra channel for grayscale images
        save_path = os.path.join(output_dir, f"{class_name}_{counter}.jpg")
        Image.fromarray(augmented_image).save(save_path)
        counter += 1
        if counter >= num_images:
            break

    print(f"Successfully created {num_images} augmented images in {output_dir}.")

if __name__ == "__main__":
    # Define parameters
    class_names = ["butterknife", "choppingboard", "fork", "grater", "knife",
                   "ladle", "plate", "roller", "spatula", "spoon"]
    input_dir = "images/original_classes"  # Directory containing the source images
    output_dir = "images/dataset1"  # Directory where augmented images will be saved
    num_images = 10  # Number of augmented images to generate

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

     # Iterate through each class name and generate augmented images
    for class_name in class_names:
        input_image = os.path.join(input_dir, f"{class_name}.jpg")  # Path to the input image
        class_output_dir = os.path.join(output_dir, class_name)  # Separate output directory per class

        # Call the function to augment images
        augment_images(
            input_image_path=input_image,
            output_dir=class_output_dir,
            class_name=class_name,
            num_images=num_images,
            augment_params=augmentation_parameters
        )
