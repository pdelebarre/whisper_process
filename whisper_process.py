import os
import asyncio
import whisper
import subprocess
import logging
import shutil
from pydub import AudioSegment
from multiprocessing import Pool, cpu_count

# Configuration
WATCH_FOLDER = "/mnt/videos"
PROCESSED_FOLDER = "/mnt/processed_videos"
MODEL_SIZE = "medium"  # Larger model might perform better
CHUNK_SIZE = 20 * 1000  # Split audio into 60-second chunks (in milliseconds)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Whisper model
model = whisper.load_model(MODEL_SIZE)

def transcribe_chunk(chunk_data):
    """
    Transcribes a single audio chunk using Whisper.
    """
    chunk_path, chunk_data = chunk_data
    try:
        # Save chunk data to a temporary file
        with open(chunk_path, 'wb') as f:
            f.write(chunk_data)

        result = model.transcribe(chunk_path)
        os.remove(chunk_path)  # Clean up the temporary file
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

    # Prepare chunks for parallel processing
    chunk_paths = []
    chunk_data_list = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"{audio_path}_chunk_{i}.wav"
        chunk_data_list.append((chunk_path, chunk.raw_data))
        chunk_paths.append(chunk_path)

    # Transcribe chunks in parallel using multiprocessing
    all_segments = []
    num_processes = min(cpu_count() * 2, len(chunk_paths))  # Increase processes if there are many chunks
    with Pool(num_processes) as pool:
        results = pool.map(transcribe_chunk, chunk_data_list)
        for segments in results:
            all_segments.extend(segments)

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
        await loop.run_in_executor(None, lambda: subprocess.run(['ffmpeg', '-i', file_path, '-q:a', '0', '-map', 'a', audio_path, '-threads', str(cpu_count())], check=True))
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
        try:
            await asyncio.sleep(10)  # Non-blocking sleep

            current_files = set(os.listdir(WATCH_FOLDER))
            new_files = [f for f in current_files if f.endswith(".mp4") and f not in processed_files]
            
            if new_files:
                tasks = []
                for file_name in new_files:
                    file_path = os.path.join(WATCH_FOLDER, file_name)
                    tasks.append(process_video(file_path))
                    processed_files.add(file_name)

                await asyncio.gather(*tasks)  # Await all processing tasks

        except Exception as e:
            logging.error(f"Error during folder polling: {e}")

if __name__ == "__main__":
    logging.info("Starting video processing application...")
    asyncio.run(poll_folder())
