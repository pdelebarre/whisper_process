from pydub import AudioSegment
from multiprocessing import Pool, cpu_count
import whisper
import os
import subprocess
import logging
import shutil
import asyncio

# Configuration
WATCH_FOLDER = "/mnt/videos"  
PROCESSED_FOLDER = "/mnt/processed_videos"  
MODEL_SIZE = "tiny"  
CHUNK_SIZE = 60 * 1000  # Split audio into 60-second chunks (in milliseconds)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Whisper model
model = whisper.load_model(MODEL_SIZE)

def transcribe_chunk(chunk_path):
    """
    Transcribes a single audio chunk using Whisper.
    """
    try:
        result = model.transcribe(chunk_path)
        return result['segments']
    except Exception as e:
        logging.error(f"Error transcribing chunk {chunk_path}: {e}")
        return []

def split_and_process_audio(audio_path):
    """
    Splits audio into chunks and processes each chunk in parallel.
    """
    audio = AudioSegment.from_wav(audio_path)
    chunks = [audio[i:i + CHUNK_SIZE] for i in range(0, len(audio), CHUNK_SIZE)]
    chunk_paths = []

    # Save chunks to files
    for i, chunk in enumerate(chunks):
        chunk_path = f"{audio_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)

    # Transcribe chunks in parallel using multiprocessing
    all_segments = []
    with Pool(cpu_count()) as pool:  # Utilize all CPU cores
        results = pool.map(transcribe_chunk, chunk_paths)
        for segments in results:
            all_segments.extend(segments)

    # Cleanup chunk files
    for path in chunk_paths:
        os.remove(path)

    return all_segments

async def process_video(file_path):
    """
    Process the video file to extract audio, transcribe using Whisper, and save the subtitles.
    """
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)
    
    loop = asyncio.get_event_loop()
    
    # Extract audio from video
    audio_path = f"{WATCH_FOLDER}/{file_name}.wav"
    logging.info(f"Extracting audio from {file_path}...")
    try:
        await loop.run_in_executor(None, lambda: subprocess.run(['ffmpeg', '-i', file_path, '-q:a', '0', '-map', 'a', audio_path], check=True))
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting audio: {e}")
        return

    # Split audio into chunks and transcribe
    logging.info(f"Transcribing audio for {file_name}...")
    all_segments = split_and_process_audio(audio_path)

    # Save subtitles to SRT file
    srt_path = f"{WATCH_FOLDER}/{file_name}.srt"
    try:
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(all_segments, start=1):
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

async def poll_folder():
    """
    Continuously polls the watch folder for new video files to process.
    """
    processed_files = set()

    while True:
        await asyncio.sleep(10)
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
    asyncio.run(poll_folder())

