import cv2

def draw_boxes(image, detections, class_names=None):
    """Draws bounding boxes and labels on an image."""
    img_copy = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        score = det["score"]

        # Draw rectangle
        color = (0, 255, 0) # Green color for bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        text = f"{label}: {score:.2f}"
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw filled rectangle for text background
        cv2.rectangle(img_copy, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
        
        # Put text on the image
        cv2.putText(img_copy, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return img_copy
