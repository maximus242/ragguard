FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy RAGGuard library
COPY . /ragguard

# Copy healthcare example
COPY examples/healthcare-hipaa /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install RAGGuard from local copy
RUN pip install --no-cache-dir /ragguard

# Generate sample data
RUN python sample_data.py

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]
