# Use a lightweight Python image
FROM python:3.9-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Install Whisper and dependencies
RUN pip install openai-whisper torch

# Create directories for the application
WORKDIR /app
RUN mkdir -p /mnt/videos
RUN mkdir -p /mnt/processed_videos

# Copy the script into the container
COPY whisper_process.py /app/

# Set the script as the entry point
ENTRYPOINT ["python3", "whisper_process.py"]
