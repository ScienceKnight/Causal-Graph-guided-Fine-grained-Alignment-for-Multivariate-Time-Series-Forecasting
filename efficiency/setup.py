import subprocess
import sys

def install_dependencies():
    dependencies = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.10.0",
        "numpy>=1.24.0",
        "accelerate>=0.19.0",
        "sacrebleu>=2.3.0"
    ]
    
    for dep in dependencies:
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

def check_environment():
    import torch
    import transformers
    import datasets
    
    print("=== Environment Check ===")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Transformers Version: {transformers.__version__}")
    print(f"Datasets Version: {datasets.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")

if __name__ == "__main__":
    install_dependencies()
    check_environment()