import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

class ImagePreprocessor:
    """A comprehensive utility class for image preprocessing and augmentation in computer vision tasks."""

    def __init__(self, target_size=(224, 224), normalize_mean=(0.485, 0.456, 0.406), normalize_std=(0.229, 0.224, 0.225)):
        """
        Initializes the ImagePreprocessor with target size and normalization parameters.

        Args:
            target_size (tuple): Desired output size for images (width, height).
            normalize_mean (tuple): Mean values for channel-wise normalization.
            normalize_std (tuple): Standard deviation values for channel-wise normalization.
        """
        self.target_size = target_size
        self.normalize_mean = np.array(normalize_mean).reshape((1, 1, 3))
        self.normalize_std = np.array(normalize_std).reshape((1, 1, 3))

    def preprocess_image(self, image_path, to_rgb=True, normalize=True):
        """
        Loads, resizes, and optionally normalizes an image for model input.

        Args:
            image_path (str): Path to the input image.
            to_rgb (bool): Whether to convert the image to RGB format.
            normalize (bool): Whether to normalize pixel values.

        Returns:
            np.ndarray: The processed image as a NumPy array.
        """
        image = Image.open(image_path)
        if to_rgb: # Ensure image is in RGB format
            image = image.convert("RGB")
        
        image = image.resize(self.target_size, Image.LANCZOS)
        image_np = np.array(image).astype(np.float32)

        if normalize:
            image_np = (image_np / 255.0 - self.normalize_mean) / self.normalize_std
        
        return image_np

    def augment_image(self, image_np, 
                      rotation_range=10, 
                      zoom_range=0.1, 
                      brightness_range=0.1, 
                      horizontal_flip=True, 
                      gaussian_blur=False):
        """
        Applies various random augmentations to an image NumPy array.

        Args:
            image_np (np.ndarray): Input image as a NumPy array (HWC, float32, normalized or 0-1).
            rotation_range (int): Max degrees for random rotation.
            zoom_range (float): Max factor for random zoom.
            brightness_range (float): Max factor for random brightness adjustment.
            horizontal_flip (bool): Whether to apply random horizontal flip.
            gaussian_blur (bool): Whether to apply random Gaussian blur.

        Returns:
            np.ndarray: Augmented image as a NumPy array.
        """
        image = Image.fromarray((image_np * 255).astype(np.uint8)) # Convert back to 0-255 for PIL ops

        # Random Rotation
        if rotation_range > 0:
            angle = np.random.uniform(-rotation_range, rotation_range)
            image = image.rotate(angle, resample=Image.BICUBIC, expand=False)

        # Random Zoom
        if zoom_range > 0:
            zoom_factor = np.random.uniform(1 - zoom_range, 1 + zoom_range)
            width, height = image.size
            new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            # Crop back to original size if zoomed in, or pad if zoomed out
            if zoom_factor > 1:
                left = (new_width - width) / 2
                top = (new_height - height) / 2
                right = (new_width + width) / 2
                bottom = (new_height + height) / 2
                image = image.crop((left, top, right, bottom))
            elif zoom_factor < 1:
                image = ImageOps.pad(image, (width, height), color=0)

        # Random Brightness
        if brightness_range > 0:
            enhancer = ImageEnhance.Brightness(image)
            brightness_factor = np.random.uniform(1 - brightness_range, 1 + brightness_range)
            image = enhancer.enhance(brightness_factor)

        # Random Horizontal Flip
        if horizontal_flip and np.random.rand() > 0.5:
            image = ImageOps.mirror(image)

        # Gaussian Blur (using OpenCV for simplicity)
        if gaussian_blur and np.random.rand() > 0.5:
            image_np_blur = np.array(image)
            image_np_blur = cv2.GaussianBlur(image_np_blur, (5, 5), 0)
            image = Image.fromarray(image_np_blur)

        # Convert back to normalized NumPy array
        image_np_augmented = np.array(image).astype(np.float32)
        image_np_augmented = (image_np_augmented / 255.0 - self.normalize_mean) / self.normalize_std

        return image_np_augmented

# Example Usage (not part of the class, for demonstration)
# if __name__ == "__main__":
#     preprocessor = ImagePreprocessor(target_size=(224, 224))
#     dummy_image_path = "path/to/your/image.jpg" # Replace with a real image path for testing
#     
#     # Preprocess an image
#     processed_img = preprocessor.preprocess_image(dummy_image_path)
#     print(f"Processed image shape: {processed_img.shape}, dtype: {processed_img.dtype}")
#
#     # Augment the processed image
#     augmented_img = preprocessor.augment_image(processed_img, rotation_range=20, gaussian_blur=True)
#     print(f"Augmented image shape: {augmented_img.shape}, dtype: {augmented_img.dtype}")
