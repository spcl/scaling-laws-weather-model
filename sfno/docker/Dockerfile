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
WORKDIR /app

# Copy only requirements-related files first for better caching
COPY pyproject.toml ./

# Upgrade pip and install base build tools
RUN pip install --upgrade pip setuptools wheel

# RUN pip install torch==2.0.1+cu117 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN pip install \
  https://download.pytorch.org/whl/cu126/torch-2.6.0%2Bcu126-cp310-cp310-linux_aarch64.whl#sha256=48775b8544e6705aa72256117f33c5f0c3c1ab51cb7abef1989dcfc3cf2e6500 \
  torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

RUN python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"


# Install Python dependencies
RUN pip install \
    tqdm numpy numba parameterized xarray xskillscore \
    timm jsbeautifier pynvml h5py wandb ruamel.yaml moviepy \
    tensorly tensorly-torch cupy-cuda12x  torchdata \
 && pip install torch_harmonics \
 && pip install git+https://github.com/NVIDIA/modulus.git

# Copy project files
COPY . .

# Install in editable mode
RUN pip install -e .

RUN python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# # optional, do this outside docker build to test cuda
# RUN python3 -m pip install pytest pytest-cov parameterized && \
#     python3 -m pytest ./tests