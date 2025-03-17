#!/bin/bash

set -e
trap 'echo "❌ Error on line $LINENO"; exit 1' ERR

echo "🔹 Starting GPU Instance Setup on Ubuntu..."
echo "============================================"

echo "🔹 Updating system packages..."; sudo apt-get update -y && sudo apt-get upgrade -y && echo "✅ System packages updated successfully."

echo "🔹 Installing essential system dependencies..."; sudo apt-get install -y xvfb ffmpeg freeglut3-dev python3-pip python3-venv wget curl && echo "✅ System dependencies installed."

echo "🔹 Installing OpenGL and GPU dependencies..."; sudo apt-get install -y libgl1-mesa-dri libglx-mesa0 libosmesa6 libglu1-mesa freeglut3-dev libglvnd-dev mesa-utils && echo "✅ OpenGL dependencies installed."

echo "🔹 Installing X11 and Virtual Display dependencies..."; sudo apt-get install -y x11-utils x11-xserver-utils xserver-xorg-video-dummy && echo "✅ X11 dependencies installed."

echo "🔹 Installing additional system dependencies..."; sudo apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 && echo "✅ Additional dependencies installed."

echo "🔹 Installing NVIDIA CUDA Toolkit..."; sudo apt-get install -y nvidia-cuda-toolkit && export PATH=/usr/local/cuda/bin:$PATH && echo "✅ CUDA Toolkit installed."


echo "🔹 Downloading and installing Miniconda..."; wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && bash miniconda.sh -b -p $HOME/miniconda && rm miniconda.sh && export PATH="$HOME/miniconda/bin:$PATH" && echo "✅ Miniconda installed successfully."

echo "🔹 Initializing Conda..."; $HOME/miniconda/bin/conda init bash && export PATH="$HOME/miniconda/bin:$PATH" && source ~/.bashrc && echo "✅ Conda initialized."

echo "🔹 Checking Conda installation..."; conda --version || { echo "❌ Error: Conda installation failed!"; exit 1; } && echo "✅ Conda is installed correctly."

echo "🔹 Creating Conda environment 'rl_env' with Python 3.10..."; conda create --name rl_env python=3.10 -y && echo "✅ Conda environment 'rl_env' created."

echo "🔹 Adding Conda and Pip to PATH permanently..."; echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc && echo 'source $HOME/miniconda/bin/activate rl_env' >> ~/.bashrc && source ~/.bashrc && echo "✅ PATH updated for Conda and Pip."

echo "🔹 Activating Conda environment..."; . $HOME/miniconda/bin/activate rl_env && echo "✅ Environment 'rl_env' is now active."

echo "🔹 Checking Pip installation..."; python -m pip --version || { echo "❌ Error: Pip installation failed!"; exit 1; } && echo "✅ Pip is installed and available."

echo "🔹 Installing required Python packages..."; conda install -y numpy scipy matplotlib pandas gymnasium torch torchvision torchaudio opencv python-dotenv tqdm jupyter jupyterlab ipython pyglet pygame imageio imageio-ffmpeg pyvirtualdisplay minigrid && echo "✅ Python packages installed."

echo "🔹 Installing additional dependencies with Pip..."; python -m pip install 'imageio==2.4.0' && python -m pip install pyvirtualdisplay piglet && python -m pip install -U --no-cache-dir gdown --pre && python -m pip install minigrid && echo "✅ Additional Pip dependencies installed."

echo "🔹 Checking installed packages..."; python -c "import numpy, scipy, matplotlib, pandas, gymnasium, torch; print('✅ Packages imported successfully!')" || { echo "❌ Package import failed!"; exit 1; }

echo "🔹 Creating 'content' directory and downloading test video file..."; mkdir -p content && wget -q "https://www.dropbox.com/scl/fi/jhkb2y3jw8wgin9e26ooc/MiniGrid-MultiRoom-N6-v0_vid.mp4?rlkey=qtkrmmbk9aiote5z7w4bx6ixi&st=zbr4gk21&dl=1" -O content/MiniGrid-MultiRoom-N6-v0_vid.mp4 || echo "⚠️ Warning: File may not have downloaded."

source ~/.bashrc

echo "🔹 Testing GPU availability with PyTorch..."; python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print('✅ GPU is available for Torch!' if torch.cuda.is_available() else '❌ GPU not found or not accessible'); print('Using GPU device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" && echo "✅ PyTorch GPU check complete."

echo "============================================"
echo "🎉 GPU Instance Setup Complete! 🎉"
echo "To start using the environment, run: . $HOME/miniconda/bin/activate rl_env"
echo "============================================"
