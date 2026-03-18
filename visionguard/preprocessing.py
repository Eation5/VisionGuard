import cv2
import numpy as np

def preprocess_image(image, target_size=(224, 224), normalize=True):
    """Resizes and optionally normalizes an image for model input."""
    # Resize image
    resized_image = cv2.resize(image, target_size)
    
    # Convert to RGB if not already (OpenCV reads as BGR)
    if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to [0, 1]
    if normalize:
        resized_image = resized_image.astype(np.float32) / 255.0
    
    # Add batch dimension if needed (e.g., for PyTorch models expecting NCHW)
    # For TensorFlow/Keras, it might expect NHWC, so this step depends on the backend.
    # For simplicity, we return NHWC and let the model handle batching.
    return resized_image

def augment_image(image, rotation_range=10, zoom_range=0.1, brightness_range=0.1):
    """Applies random augmentations to an image."""
    # Simple rotation
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))

    # Simple zoom
    if zoom_range > 0:
        zoom_factor = np.random.uniform(1 - zoom_range, 1 + zoom_range)
        image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
        # Crop back to original size if zoomed in
        h, w = image.shape[:2]
        original_h, original_w = image.shape[:2] # Assuming original_h, original_w are available
        if zoom_factor > 1:
            start_h = max(0, (h - original_h) // 2)
            start_w = max(0, (w - original_w) // 2)
            image = image[start_h:start_h + original_h, start_w:start_w + original_w]
        elif zoom_factor < 1:
            # Pad if zoomed out
            pad_h = original_h - h
            pad_w = original_w - w
            top, bottom = pad_h // 2, pad_h - (pad_h // 2)
            left, right = pad_w // 2, pad_w - (pad_w // 2)
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        image = cv2.resize(image, (original_w, original_h)) # Ensure it's back to original size

    # Simple brightness adjustment
    if brightness_range > 0:
        brightness_factor = np.random.uniform(1 - brightness_range, 1 + brightness_range)
        image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

    return image
