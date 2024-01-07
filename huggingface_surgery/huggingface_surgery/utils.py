import torch

def getFreeCUDAMemory():
    t = torch.cuda.get_device_properties(0).total_memory/10**9
    r = torch.cuda.memory_reserved(0)/10**9
    a = torch.cuda.memory_allocated(0)/10**9
    print(f"Total: {t}, Reserved: {r}, Allocated: {a}")