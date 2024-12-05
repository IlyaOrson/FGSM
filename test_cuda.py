import torch

print("Is cuda available from pytorch?\t", torch.cuda.is_available())

current_device = torch.cuda.current_device()
print("Device loaded from pytorch:\t", current_device)

print(
    "Device properties\t",
    torch.cuda.get_device_properties(current_device),
)

test_tensor = torch.rand(2, 3).cuda()
print("Load a tensor on device\t", test_tensor.device)
