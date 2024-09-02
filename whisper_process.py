import os
import time
import whisper
import subprocess

# Configuration
WATCH_FOLDER = "/mnt/videos"  # This is the path to the shared folder mounted in the Docker container
PROCESSED_FOLDER = "/mnt/processed_videos"  # Folder where processed videos will be moved
MODEL_SIZE = "base"  # Whisper model size - can be 'tiny', 'base', 'small', 'medium', 'large'

# Load Whisper model
model = whisper.load_model(MODEL_SIZE)

def process_video(file_path):
    """
    Process the video file to extract audio, transcribe using Whisper, and save the subtitles.
    """
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)
    
    # Extract audio from video
    audio_path = f"{WATCH_FOLDER}/{file_name}.wav"
    print(f"Extracting audio from {file_path}...")
    subprocess.run(['ffmpeg', '-i', file_path, '-q:a', '0', '-map', 'a', audio_path])
    
    # Transcribe audio using Whisper
    print(f"Transcribing audio for {file_name}...")
    result = model.transcribe(audio_path)
    
    # Save subtitles to SRT file
    srt_path = f"{WATCH_FOLDER}/{file_name}.srt"
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], start=1):
            start_time = whisper.utils.format_timestamp(segment['start'], always_include_hours=True)
            end_time = whisper.utils.format_timestamp(segment['end'], always_include_hours=True)
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{segment['text']}\n\n")
    
    print(f"Subtitles saved to {srt_path}")
    
    # Move processed video to another folder
    processed_path = os.path.join(PROCESSED_FOLDER, base_name)
    os.rename(file_path, processed_path)
    print(f"Moved processed video to {processed_path}")

def poll_folder():
    """
    Continuously polls the watch folder for new video files to process.
    """
    processed_files = set()

    while True:
        for file_name in os.listdir(WATCH_FOLDER):
            if file_name.endswith(".mp4") and file_name not in processed_files:
                file_path = os.path.join(WATCH_FOLDER, file_name)
                process_video(file_path)
                processed_files.add(file_name)

        # Wait before polling again
        time.sleep(10)

if __name__ == "__main__":
    print("Starting video processing application...")
    poll_folder()
