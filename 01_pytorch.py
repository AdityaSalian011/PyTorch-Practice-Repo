import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name() if torch.cuda.is_available() else 'No GPU Device.')