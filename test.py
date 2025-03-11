if __name__ == "__main__":
    
    import sys
    print(sys.executable)

    import torch
    from gpu_check import check_gpu_availability, get_device, verify_cuda_operation
    
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
        if verify_cuda_operation():
            print("CUDA operations verified successfully.")
        else:
            print("CUDA operations failed.")
    else:
        print("CUDA is not available on this system.")