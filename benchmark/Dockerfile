FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update -qq && \
    apt-get install -y -qq python3 python3-pip > /dev/null 2>&1 && \
    rm -rf /var/lib/apt/lists/*

# Install TensorFlow with CUDA + benchmark deps
COPY requirements_benchmark.txt .
RUN pip3 install --no-cache-dir "tensorflow[and-cuda]" && \
    pip3 install --no-cache-dir -r requirements_benchmark.txt

# Working directory will be mounted at runtime
WORKDIR /workspace
