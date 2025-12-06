# tl-tensor-examples Docker image
# Provides Jones polynomial computation and knot identification
#
# Build:
#   docker build -t tl-tensor-examples .
#
# Build with KaHyPar optimizer (longer build time):
#   docker build --build-arg WITH_KAHYPAR=1 -t tl-tensor-examples .
#
# Build for multiple architectures:
#   docker buildx build --platform linux/amd64,linux/arm64 -t tl-tensor-examples .
#
# Run interactive:
#   docker run -it tl-tensor-examples python
#
# Run with mounted data:
#   docker run -v $(pwd)/data:/data tl-tensor-examples python /data/my_script.py
#
# Note: For best performance with narrow braids (≤5 strands), consider using
# the "topology" conda environment approach instead of Docker. See README.md
# for details on setting up the Sage+tl-tensor combined environment.

FROM python:3.11-slim

# Build argument for optional KaHyPar support
ARG WITH_KAHYPAR=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    cmake \
    libboost-program-options-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required for tl-tensor)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create working directory
WORKDIR /app

# Install Python dependencies
# Note: We install maturin first (needed to build tl-tensor from source if wheels unavailable)
RUN pip install --no-cache-dir maturin

# Install core dependencies
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    sympy \
    cotengra \
    opt_einsum

# Install tl-tensor (may build from source on ARM)
RUN pip install --no-cache-dir tl-tensor

# Install SnaPPy (works on both x86 and ARM via pip now)
# Note: jones_polynomial() requires Sage, but converters/volume/signature work
RUN pip install --no-cache-dir snappy || echo "SnaPPy installation failed, continuing without it"

# Optional: Build and install KaHyPar
# KaHyPar provides hypergraph partitioning for tensor contraction optimization.
# For TL-algebra tensor networks, the greedy optimizer is typically faster,
# but KaHyPar may be beneficial for other tensor network structures.
RUN if [ "$WITH_KAHYPAR" = "1" ]; then \
    git clone --recursive https://github.com/kahypar/kahypar.git /tmp/kahypar && \
    cd /tmp/kahypar && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DKAHYPAR_PYTHON_INTERFACE=ON && \
    make -j$(nproc) && \
    cp python/kahypar*.so $(python -c "import site; print(site.getsitepackages()[0])") && \
    cd / && rm -rf /tmp/kahypar; \
    fi

# Copy package files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY examples/ ./examples/

# Install the package
RUN pip install --no-cache-dir -e .

# Verify installation
RUN python -c "from tl_examples import compute_jones, has_kahypar; print(f'tl-tensor-examples installed successfully. KaHyPar: {has_kahypar()}')"

# Default command
CMD ["python"]
