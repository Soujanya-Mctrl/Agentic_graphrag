FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and data
COPY . .

# Environment optimizations for single-threaded container execution
ENV PORT=8000
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
