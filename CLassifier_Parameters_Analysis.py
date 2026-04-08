import torch
import torch.nn as nn
import time
import numpy as np

# 1. Model Definition (Same as training code)
class MyCNN(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 5
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 6
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

# 2. Analysis Functions (Parameters, FLOPs, Speed)

def count_parameters(model):
    """Calculate the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_size):
    """Manually calculate FLOPs for Conv2d and Linear layers."""
    flops = 0
    x = torch.rand(1, *input_size)
    
    # Feature Extractor (Conv layers)
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            # Calculate output dimensions
            out_h = int((x.size(2) + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) / layer.stride[0] + 1)
            out_w = int((x.size(3) + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) / layer.stride[1] + 1)
            
            # MACs = Kernel_h * Kernel_w * In_C * Out_H * Out_W * Out_C
            macs = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels * out_h * out_w * layer.out_channels
            # FLOPs = 2 * MACs (Multiply + Add operations)
            flops += 2 * macs
            
            x = layer(x) # Forward pass for the next layer
        elif isinstance(layer, (nn.MaxPool2d, nn.BatchNorm2d, nn.ReLU)):
            x = layer(x) # FLOPs are negligible for these layers
            
    # Classifier (Linear layer)
    x = model.global_pool(x)
    x = torch.flatten(x, 1)
    
    layer = model.classifier
    macs = layer.in_features * layer.out_features
    flops += 2 * macs
    
    return flops

def measure_inference_speed(model, input_size, device):
    """Measure the average inference time (Latency) per image."""
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(1, *input_size).to(device)

    # 1. Warm-up
    print("Warming up GPU/CPU...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    # 2. Speed Measurement
    iterations = 1000
    print(f"Measuring average inference time over {iterations} runs on [{device}]...")
    
    if device.type == 'cuda':
        torch.cuda.synchronize() # Wait for GPU to finish previous tasks
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(dummy_input)
        torch.cuda.synchronize()
        end = time.time()
    else:
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(dummy_input)
        end = time.time()
        
    avg_time = (end - start) / iterations * 1000 # Convert to milliseconds
    return avg_time

# 3. Execution and Result Output
if __name__ == "__main__":
    print(" Computational Efficiency Analysis (CNN Framework) ")
    
    # Configuration
    n_cls = 5
    model = MyCNN(n_cls)
    
    # 1. Parameter Count
    total_params = count_parameters(model)
    print(f"\n[1] Model Parameters: {total_params:,} (Approx {total_params/1e6:.2f} M)")
    
    # 2. FLOPs Calculation (Based on Low-Res 115x40)
    input_res_low = (1, 115, 40)
    flops_low = count_flops(model, input_res_low)
    print(f"[2] FLOPs (Input {input_res_low}): {flops_low:,} ({flops_low/1e9:.4f} GFLOPs)")
    
    # FLOPs Calculation (Based on High-Res 460x160 for comparison)
    input_res_high = (1, 460, 160)
    flops_high = count_flops(model, input_res_high)
    print(f"    FLOPs (Input {input_res_high}): {flops_high:,} ({flops_high/1e9:.4f} GFLOPs)")
    print(f"    -> Theoretical Efficiency Gain: {flops_high/flops_low:.1f}x reduction in computational cost")

    # 3. Inference Speed Measurement
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    avg_time = measure_inference_speed(model, input_res_low, device)
    
    print(f"\n[3] Inference Speed Test (Device: {device})")
    print(f"    Average Latency per Image: {avg_time:.4f} ms")
    print(f"    Throughput: {1000/avg_time:.1f} FPS")