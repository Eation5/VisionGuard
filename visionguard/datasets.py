import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    """A generic dataset for loading images from a directory."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # In a real scenario, you would also load labels if available
        label = 0 # Dummy label for demonstration
        return image, label



def get_default_transforms(image_size=(224, 224)):
    """Returns a set of default image transformations."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
