import torch

print("Torch version: ", torch.__version__)
print("Is cuda available from pytorch? ", torch.cuda.is_available())

current_device = torch.cuda.current_device()
print("Device loaded from pytorch: ", current_device)

print("Device properties:")
print(torch.cuda.get_device_properties(current_device))

test_tensor = torch.rand(2, 3).cuda()
print("Load a tensor on device... ", test_tensor.device)
