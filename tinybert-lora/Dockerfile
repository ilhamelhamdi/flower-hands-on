FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set working directory inside the container
WORKDIR /root

# Install uv (fast Python package manager)
ADD https://astral.sh/uv/install.sh /install.sh
RUN chmod +x /install.sh && /install.sh && rm /install.sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy configuration and lock files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv pip install --system -r pyproject.toml

# Copy the rest of the application code
COPY . .

# Ensure the project is installed in editable mode inside the container
RUN uv pip install --system -e .

# Set the default command (can be overridden by Singularity/Slurm)
CMD ["python", "-m", "tinybert_lora.train", "sst2"]