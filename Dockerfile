
FROM python:3.11-slim

WORKDIR /app

# Configure the Python environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Install system libraries required by ML packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN python -m pip install --no-cache-dir \
    -r requirements.txt

# Copy project source code
COPY src/ ./src/

# Run baseline inference by default
CMD ["python", "src/baseline.py"]
