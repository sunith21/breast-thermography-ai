import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BreastThermalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        classes = {
            "normal": 0,
            "sick": 1
        }

        for cls_name, label in classes.items():
            cls_path = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Test the dataset
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    dataset = BreastThermalDataset("data/raw", transform=transform)

    print("Total images:", len(dataset))
    img, label = dataset[0]
    print("Image shape:", img.shape)
    print("Label:", label)
