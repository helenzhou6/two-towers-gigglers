import torch

print("PyTorch version:   ", torch.__version__)
print("CUDA toolkit ver.:  ", torch.version.cuda)
print("CUDA available?    ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Number of GPUs:    ", torch.cuda.device_count())
    print("Current GPU name:  ", torch.cuda.get_device_name(0))
