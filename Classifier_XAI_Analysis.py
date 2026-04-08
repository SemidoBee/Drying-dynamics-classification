import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# 1. Model Definition
class MyCNN(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout     = nn.Dropout(0.5)
        self.classifier  = nn.Linear(64, n_cls)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

# 2. XAI Tool: Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.fmap = None
        self.grad = None

        # Register hooks to capture feature maps and gradients
        self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.fmap = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.grad = grad_output[0].detach()

    def generate(self, class_idx):
        # Global average pooling on the gradients
        grads = self.grad.mean(dim=(2, 3), keepdim=True)
        # Weight the feature maps by the pooled gradients
        cam   = (grads * self.fmap).sum(dim=1)
        # Apply ReLU to retain only positive influences
        cam   = F.relu(cam)

        result = []
        for c in cam:
            c_min, c_max = c.min(), c.max()
            if c_max - c_min > 1e-6:
                c = (c - c_min) / (c_max - c_min) # Normalize to [0, 1]
            else:
                c = torch.zeros_like(c)
            result.append(c)

        return torch.stack(result, dim=0)

# 3. XAI Tool: Input Gradient Saliency
def compute_input_gradient(model, x, class_idx=None):
    model.eval()
    x = x.clone().detach().requires_grad_(True)

    out = model(x)
    if class_idx is None:
        class_idx = int(out.argmax())

    score = out[0, class_idx]
    score.backward()

    # Extract absolute gradients with respect to the input image
    grad = x.grad.detach()[0, 0]
    g = grad.abs()
    g_min, g_max = g.min(), g.max()

    # Normalize to [0, 1]
    if g_max - g_min > 1e-6:
        g = (g - g_min) / (g_max - g_min)
    else:
        g = torch.zeros_like(g)

    return g.cpu().numpy()

# 4. Main Analysis and Visualization
def analyze_image(img_path, model_path, class_names, output_dir, img_size=(115,40), device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} for XAI extraction")
    
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = MyCNN(len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess image
    val_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    pil_img = Image.open(img_path).convert("RGB")
    inp = val_tf(pil_img).unsqueeze(0).to(device)

    # Inference (Prediction)
    with torch.no_grad():
        logits = model(inp)
        probs  = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_cls = class_names[pred_idx]

    print(f"=== Prediction for {os.path.basename(img_path)} ===")
    for i, n in enumerate(class_names):
        print(f"  {n}: {probs[i]*100:.2f}%")
    print(f"→ Predicted Class: {pred_cls}")

    # Extract Grad-CAM (For reproducing Fig. 3F)
    target_layer = model.features[23] # Last Conv2d layer (Block 6)
    cam_extractor = GradCAM(model, target_layer)

    model.zero_grad()
    score = model(inp)[0, pred_idx]
    score.backward()
    cam = cam_extractor.generate(pred_idx)[0].cpu().numpy()

    # Extract Input Gradient Saliency (For reproducing Fig. 3E)
    grad_map = compute_input_gradient(model, inp, pred_idx)

    # Visualize and Save Results
    H, W = img_size
    
    # 1. Save Input Gradient Saliency Map
    plt.figure(figsize=(3, 6))
    plt.imshow(grad_map, cmap='hot')
    plt.axis("off")
    saliency_save_path = os.path.join(output_dir, f'saliency_map_{pred_cls}.png')
    plt.savefig(saliency_save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    # 2. Save Grad-CAM Overlay
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H))
    cam_resized = np.array(cam_resized) / 255.0
    
    base_img = np.array(pil_img.convert("L").resize((W, H))).astype(np.float32)
    base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min() + 1e-6)

    plt.figure(figsize=(3, 6))
    plt.imshow(base_img, cmap='gray')
    plt.imshow(cam_resized, cmap='jet', alpha=0.5)
    plt.axis("off")
    gradcam_save_path = os.path.join(output_dir, f'gradcam_overlay_{pred_cls}.png')
    plt.savefig(gradcam_save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    print(f"[Done] XAI results saved successfully to '{output_dir}'.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract XAI maps (Saliency & Grad-CAM) for Nature Communications submission")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input original RGB image")
    parser.add_argument('--model_weight', type=str, required=True, help="Path to trained .pth model weights")
    parser.add_argument('--output_dir', type=str, default='./xai_results', help="Directory to save the generated heatmaps")
    args = parser.parse_args()

    # 5 drying stages defined in the paper (Stage A-C, alpha, beta)
    class_names = ["A", "B", "C", "alpha", "beta"] 
    
    analyze_image(
        img_path=args.image_path,
        model_path=args.model_weight,
        class_names=class_names,
        output_dir=args.output_dir,
        img_size=(115, 40)
    )