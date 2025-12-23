import torch
import cv2
import numpy as np
import os
from torchvision import transforms
from PIL import Image

from model import get_model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output
        output.retain_grad()

    def generate(self, input_tensor, class_idx):
        # Enable gradients explicitly
        with torch.enable_grad():
            output = self.model(input_tensor)

            # Backprop for selected class
            score = output[:, class_idx].sum()
            self.model.zero_grad()
            score.backward()

            # Get gradients from retained activations
            gradients = self.activations.grad

            if gradients is None:
                raise RuntimeError("Gradients are None — Grad-CAM cannot be computed")

            # Global average pooling
            weights = gradients.mean(dim=(2, 3), keepdim=True)

            cam = (weights * self.activations).sum(dim=1)
            cam = torch.relu(cam)
            cam -= cam.min()
            cam /= (cam.max() + 1e-8)

            return cam



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(num_classes=2)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    target_layer = model.layer4
    cam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load ONE sick image
    img_path = os.path.join("data/raw/sick", os.listdir("data/raw/sick")[0])
    img = Image.open(img_path).convert("RGB")

    input_tensor = transform(img).unsqueeze(0).to(device)
    input_tensor.requires_grad_()
    output = model(input_tensor)
    class_idx = torch.argmax(output).item()

    heatmap = cam.generate(input_tensor, class_idx)
    heatmap = heatmap.squeeze().detach().cpu().numpy()

    heatmap = cv2.resize(heatmap, (224, 224))

    img_np = np.array(img.resize((224, 224)))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    cv2.imwrite("gradcam_output.jpg", overlay)

    print("Grad-CAM saved as gradcam_output.jpg")


if __name__ == "__main__":
    main()
