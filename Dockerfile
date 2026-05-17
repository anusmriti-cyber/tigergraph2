FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (if it exists) or just install the ones we know
COPY backend/training/requirements.txt ./
# Fallback in case requirements.txt doesn't have everything
RUN pip install --no-cache-dir fastapi uvicorn pydantic huggingface_hub groq transformers torch sentence-transformers faiss-cpu networkx

# Copy the whole project
COPY . .

# Expose port
EXPOSE 7860

# Environment variables for Hugging Face Spaces
ENV HOST=0.0.0.0
ENV PORT=7860

# Run the app
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
