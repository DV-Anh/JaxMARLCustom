FROM nvcr.io/nvidia/jax:23.10-py3

# Set build argument for CUDA usage
ARG USE_CUDA=False

# Default workdir
WORKDIR /app

# Install tmux
# USER root
RUN apt-get update && \
    apt-get install -y tmux

# Copy only the dependency files first
COPY pyproject.toml .
COPY requirements/requirements.txt ./requirements/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements/requirements.txt

# Copy the rest of the application code
COPY . .

# Disable preallocation and set other environment variables
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# For secrets and debug
ENV WANDB_API_KEY=""
ENV WANDB_ENTITY=""

EXPOSE 5088
ENV FLASK_RUN_PORT=5088
CMD ["python", "custom/backend/app.py"]
