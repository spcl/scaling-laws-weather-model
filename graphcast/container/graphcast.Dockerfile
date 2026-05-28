FROM nvcr.io/nvidia/cuda:12.3.1-runtime-ubuntu22.04
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive\
        apt install -y python3 python3-pip git \
                       libexpat1 libglib2.0-0 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* 

RUN pip3 install --no-cache-dir --upgrade pip

RUN pip install --upgrade https://github.com/deepmind/graphcast/archive/master.zip \
         "jax[cuda]" google-cloud-storage gcsfs optax wandb importlib_resources
         

