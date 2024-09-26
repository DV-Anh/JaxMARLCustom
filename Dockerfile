FROM nvcr.io/nvidia/jax:23.10-py3

ARG USE_CUDA=False
WORKDIR /app

# Create a non-root user and group with IDs that won't conflict with host users
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser

# Install tmux as root before switching users
USER root
RUN apt-get update && \
    apt-get install -y tmux

# Switch to non-root user
USER appuser

COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser requirements/requirements.txt ./requirements/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements/requirements.txt
COPY --chown=appuser:appuser . .

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# For secrets and debug
ENV WANDB_API_KEY=""
ENV WANDB_ENTITY=""

# Expose the necessary port for Flask
EXPOSE 5088
ENV FLASK_RUN_PORT=5088

# Switch back to root user to ensure file permissions can be properly set
USER root

# Ensure proper permissions for non-root user on the /app directory
RUN chown -R appuser:appuser /app

# Switch to the non-root user to run the application
USER appuser

CMD ["python", "custom/backend/app.py"]