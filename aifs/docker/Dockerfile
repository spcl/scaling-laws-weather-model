FROM nvcr.io/nvidia/ai-workbench/python-cuda117:1.0.3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /workspace

# Upgrade pip and install base build tools
RUN pip install --upgrade pip setuptools wheel

# This is based on py3.10 and anemoi-training requires torch2.9
RUN pip install \
  https://download.pytorch.org/whl/cu126/torch-2.9.0%2Bcu126-cp310-cp310-manylinux_2_28_aarch64.whl#sha256=31ef6cf39c85a368b09b4fcb92e520ea6dae0121faba28107d8eab6f78f67d51 \
  torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

RUN python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"


# Install Python dependencies
RUN pip install \
    tqdm numpy numba parameterized xarray xskillscore \
    timm jsbeautifier pynvml h5py wandb ruamel.yaml moviepy \
    tensorly tensorly-torch cupy-cuda12x torchdata trimesh \
    torch_harmonics pytorch-lightning==2.5.5

# Copy project files
COPY . .

RUN ls 

# Must use absolute path to install the packages
RUN cd /workspace && pip install ./graphs ./models ./training

RUN python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

# # Install testing, optional, do this outside docker build to test cuda
# RUN pip install pytest && cd training && pytest
