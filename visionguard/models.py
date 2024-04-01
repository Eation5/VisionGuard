import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image

class YOLOv5:
    """YOLOv5 object detection model wrapper."""
    def __init__(self, model_path="yolov5s.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
        print(f"Initializing YOLOv5 model from {model_path} on {device}")
        try:
            # In a real scenario, you would load the YOLOv5 model here.
            # For demonstration, we'll use a placeholder and simulate detections.
            # self.model = torch.hub.load(\'ultralytics/yolov5\', \'yolov5s\', pretrained=True).to(device)
            # self.model.eval()
            self.device = device
            self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
            print("YOLOv5 model placeholder initialized. Using dummy detections.")
        except Exception as e:
            print(f"Could not load YOLOv5 model (this is expected in a sandboxed environment): {e}")
            self.device = device
            self.classes = ["person", "car"]

    def predict(self, image):
        """Simulates object detection on an image. Returns bounding boxes, labels, and scores."""
        # In a real implementation, this would run inference with the YOLOv5 model.
        # results = self.model(image)
        # detections = results.pandas().xyxy[0].to_dict(orient="records")
        
        # For demonstration, we return dummy bounding boxes and labels.
        h, w, _ = image.shape
        detections = []

        if np.random.rand() > 0.5:
            x1, y1, x2, y2 = int(w * 0.1), int(h * 0.2), int(w * 0.4), int(h * 0.7)
            detections.append({
                "box": [x1, y1, x2, y2],
                "label": "person",
                "score": 0.95
            })
        
        if np.random.rand() > 0.3:
            x1, y1, x2, y2 = int(w * 0.5), int(h * 0.3), int(w * 0.9), int(h * 0.8)
            detections.append({
                "box": [x1, y1, x2, y2],
                "label": "car",
                "score": 0.88
            })

        return detections

class ImageClassifier:
    """Image classification model using a pre-trained ResNet."""
    def __init__(self, num_classes, model_name="resnet18", pretrained=True, device="cuda" if torch.cuda.is_available() else "cpu"):
        print(f"Initializing {model_name} classifier with {num_classes} classes on {device}.")
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        
        # Modify the final layer for the number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)
        self.model = self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode

    def predict(self, image_path):
        """Predicts the class of an image from its path."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

class SegmentationModel:
    """Semantic segmentation model using a pre-trained FCN ResNet."""
    def __init__(self, num_classes, model_name="fcn_resnet50", pretrained=True, device="cuda" if torch.cuda.is_available() else "cpu"):
        print(f"Initializing {model_name} segmentation model with {num_classes} classes on {device}.")
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if model_name == "fcn_resnet50":
            self.model = models.segmentation.fcn_resnet50(pretrained=pretrained)
        elif model_name == "deeplabv3_resnet101":
            self.model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        
        # Modify the classifier for the number of classes
        self.model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model = self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode

    def predict(self, image_path):
        """Performs semantic segmentation on an image from its path."""
        image = Image.open(image_path).convert("RGB")
        # Resize image to a common size for the model, e.g., 520x520
        input_image = image.resize((520, 520))
        input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)["out"]
        
        # Get the predicted mask (class with highest probability for each pixel)
        predicted_mask = output.argmax(1).squeeze(0).cpu().numpy()
        
        # Resize mask back to original image size for visualization if needed
        predicted_mask_resized = cv2.resize(predicted_mask.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST)
        
        return predicted_mask_resized
