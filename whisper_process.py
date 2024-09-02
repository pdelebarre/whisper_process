import os
import time
import whisper
import subprocess
import logging
import shutil  # Import shutil for moving files across devices

# Configuration
WATCH_FOLDER = "/mnt/videos"  # Path to the shared folder mounted in the Docker container
PROCESSED_FOLDER = "/mnt/processed_videos"  # Folder where processed videos will be moved
MODEL_SIZE = "tiny"  # Whisper model size - can be 'tiny', 'base', 'small', 'medium', 'large'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info(f"Extracting audio from {file_path}...")
    try:
        subprocess.run(['ffmpeg', '-i', file_path, '-q:a', '0', '-map', 'a', audio_path], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting audio: {e}")
        return

    # Transcribe audio using Whisper
    logging.info(f"Transcribing audio for {file_name}...")
    try:
        result = model.transcribe(audio_path)  # Removed batch_size argument
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        return
    
    # Save subtitles to SRT file
    srt_path = f"{WATCH_FOLDER}/{file_name}.srt"
    try:
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], start=1):
                start_time = whisper.utils.format_timestamp(segment['start'], always_include_hours=True)
                end_time = whisper.utils.format_timestamp(segment['end'], always_include_hours=True)
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
        logging.info(f"Subtitles saved to {srt_path}")
    except IOError as e:
        logging.error(f"Error saving subtitles: {e}")
        return
    
    # Move processed video to another folder using shutil.move
    processed_path = os.path.join(PROCESSED_FOLDER, base_name)
    try:
        shutil.move(file_path, processed_path)
        logging.info(f"Moved processed video to {processed_path}")
    except OSError as e:
        logging.error(f"Error moving file: {e}")

def poll_folder():
    """
    Continuously polls the watch folder for new video files to process.
    """
    processed_files = set()

    while True:
        try:
            current_files = set(os.listdir(WATCH_FOLDER))
            new_files = [f for f in current_files if f.endswith(".mp4") and f not in processed_files]
            
            if new_files:
                for file_name in new_files:
                    file_path = os.path.join(WATCH_FOLDER, file_name)
                    process_video(file_path)
                    processed_files.add(file_name)
            
            # Wait before polling again
            time.sleep(10)
        except Exception as e:
            logging.error(f"Error during folder polling: {e}")

if __name__ == "__main__":
    logging.info("Starting video processing application...")
    poll_folder()
