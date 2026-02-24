import torch
import intel_extension_for_pytorch as ipex

if torch.cuda.is_available():
    print(
        f"NVIDIA CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
elif torch.xpu.is_available():
    print(f"Intel GPU detected: {torch.xpu.get_device_name(0)}")
else:
    print("Intel GPU (XPU) not detected. Falling back to CPU.")
