# Use a lightweight Python image
FROM python:3.9-slim

# Install FFmpeg and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Whisper and dependencies
RUN pip install --no-cache-dir \
    openai-whisper \
    torch --extra-index-url https://download.pytorch.org/whl/cpu \
    pydub

# Create directories for the application
WORKDIR /app
RUN mkdir -p /mnt/videos /mnt/processed_videos

# Copy the script into the container
COPY whisper_process.py /app/

# Set the script as the entry point
ENTRYPOINT ["python3", "whisper_process.py"]

