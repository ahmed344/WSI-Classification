FROM nvcr.io/nvidia/pyg:26.01-py3

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER 1000

# Install mamba-ssm from the wheel matching nvcr.io/nvidia/pyg:26.01-py3.
RUN pip install --no-cache-dir \
    "https://github.com/state-spaces/mamba/releases/download/v2.3.1/mamba_ssm-2.3.1%2Bcu13torch26.01cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# Build torch-cluster against the NVIDIA PyTorch already present in the image.
RUN pip install --no-cache-dir --no-build-isolation torch-cluster

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir -r requirements.txt