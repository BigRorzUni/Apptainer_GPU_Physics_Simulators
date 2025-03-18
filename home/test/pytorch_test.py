import torch
import pytest

def test_pytorch_using_GPU():
    gpu_available = torch.cuda.is_available()

    if gpu_available:
        device = torch.device("cuda")

        tensor = torch.randn(3, 3).to(device)

        result = tensor @ tensor

        # Test that the result is still on the GPU
        assert result.device.type == "cuda"
    else:
        # Fail because the GPU is not available
        print("GPU backend not available for GPU")
        assert 0 == 1