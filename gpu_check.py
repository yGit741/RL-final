import torch

def check_gpu_availability():
    """
    Check if CUDA (GPU) is available and return relevant system information.
    
    Returns:
        dict: Dictionary containing information about GPU availability and configuration
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': 'cpu'
    }
    
    if info['cuda_available']:
        info['current_device'] = f"cuda:{torch.cuda.current_device()}"
        info['device_name'] = torch.cuda.get_device_name(0)
        info['device_capability'] = torch.cuda.get_device_capability(0)
    
    return info

def get_device():
    """
    Get the appropriate device (GPU if available, otherwise CPU).
    
    Returns:
        torch.device: The device to use for PyTorch operations
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def verify_cuda_operation():
    """
    Verify that CUDA operations work by performing a simple tensor operation.
    
    Returns:
        bool: True if CUDA operations work successfully, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        # Create a test tensor and move it to GPU
        x = torch.randn(10, 10).cuda()
        # Perform a simple operation
        y = x @ x.t()
        # Move back to CPU and check if operation was successful
        y = y.cpu()
        return True
    except Exception as e:
        print(f"CUDA operation failed: {str(e)}")
        return False

# Example usage:
if __name__ == "__main__":
    # Get system information
    gpu_info = check_gpu_availability()
    print("\nGPU System Information:")
    print("----------------------")
    for key, value in gpu_info.items():
        print(f"{key}: {value}")
    
    # Get the device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Verify CUDA operations
    if gpu_info['cuda_available']:
        cuda_works = verify_cuda_operation()
        print(f"\nCUDA operations {'working properly' if cuda_works else 'failed'}")
    else:
        print("\nCUDA is not available on this system")