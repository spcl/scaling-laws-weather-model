FROM nvcr.io/nvidia/modulus/modulus:24.04
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y git curl wget libeccodes0 libopenmpi3 libopenjp2-7 ncdu htop screen zip unzip git vim ffmpeg libjpeg-dev libpng-dev nvtop

RUN pip install --upgrade pip setuptools
RUN pip install --no-deps wandb lightning DeepSpeed flash-attn nvidia-dali-cuda120 torchmetrics huggingface-hub memory_profiler nvitop natsort scores==1.3.0


# Configure glymur
RUN mkdir -p /root/.config/glymur && printf "[library]\nopenjp2: /usr/lib/aarch64-linux-gnu/libopenjp2.so.7" > /root/.config/glymur/glymurrc
ENV XDG_CONFIG_HOME="/root/.config"

# Set up workspace
RUN mkdir -p /workspace
WORKDIR /workspace

# Set LD_LIBRARY_PATH for CUDA and HPC-X
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/hpcx/ucc/lib/:/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

CMD [ "/bin/bash" ]
