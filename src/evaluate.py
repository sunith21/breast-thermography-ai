import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import numpy as np

from dataset import BreastThermalDataset
from model import get_model


def evaluate():
    data_dir = "data/raw"
    batch_size = 16
    val_split = 0.2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    dataset = BreastThermalDataset(data_dir, transform=transform)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = get_model(num_classes=2)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Metrics
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=["Normal", "Sick"]))

    roc_auc = roc_auc_score(all_labels, all_probs)
    print("ROC-AUC Score:", roc_auc)


if __name__ == "__main__":
    evaluate()
