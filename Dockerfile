# Redrob Hackathon Stage 3 Sandbox Environment
# This Dockerfile provides a completely sandboxed, 100% reproducible environment 
# matching the Stage 3 CPU constraint (Python 3.11, no GPU).

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire repository (including the local ./models/ directory and pre-computed embeddings)
COPY . .

# The default command strictly matches the reproduce_command from submission_metadata.yaml
CMD ["python", "rank.py", "--candidates", "./candidates.jsonl", "--embeddings", "./embeddings_full.npz", "--out", "./submission.csv"]
