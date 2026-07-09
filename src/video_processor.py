import os
import cv2
import base64
import logging
import urllib.parse
import tempfile
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video_processor")

def download_video(url: str) -> str:
    """
    Downloads a video from a URL to a temporary file.
    If the url is already a local path, it validates its existence and returns it directly.
    """
    # Check if it is a local file path
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or parsed.scheme == 'file':
        local_path = parsed.path if parsed.path else url
        if os.path.exists(local_path):
            logger.info(f"Using existing local video path: {local_path}")
            return local_path
        else:
            raise FileNotFoundError(f"Local video file not found: {local_path}")
    
    # Download from URL
    logger.info(f"Downloading video from: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to initiate download from {url}: {e}")
        raise
    
    # Save to a temporary file
    suffix = ".mp4"
    # Try to guess extension from URL
    path_without_query = parsed.path
    if path_without_query.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        suffix = os.path.splitext(path_without_query)[1]
        
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = temp_file.name
    temp_file.close()  # Close so other processes (OpenCV) can open it
    
    try:
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Video successfully downloaded to: {temp_path} (size: {os.path.getsize(temp_path)} bytes)")
        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        logger.error(f"Failed to save downloaded video: {e}")
        raise

def extract_uniform_frames(video_path: str, max_frames: int = 10, target_size: int = 512) -> list[dict]:
    """
    Extracts up to `max_frames` uniformly spaced frames from the video.
    Resizes each frame (max dimension: target_size) to reduce token count and API payload size.
    Returns a list of dicts: [{"timestamp": "3.2s", "base64_image": "..."}]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found for frame extraction: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0 or fps <= 0:
        # Fallback if properties are not available (read frames sequentially)
        logger.warning(f"Invalid total_frames ({total_frames}) or fps ({fps}). Reading sequentially...")
        frames = []
        timestamps = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # To avoid memory issues, save raw frames only if we need them
            frames.append(frame)
            count += 1
        cap.release()
        
        total_frames = len(frames)
        fps = 30.0  # Assumed fallback FPS
        logger.info(f"Read {total_frames} frames sequentially.")
        
        if total_frames == 0:
            raise ValueError("No frames could be read from the video.")
            
        # Select indexes uniformly
        step = max(1, total_frames // max_frames)
        selected_frames = []
        for i in range(0, total_frames, step):
            if len(selected_frames) >= max_frames:
                break
            frame = frames[i]
            timestamp_sec = i / fps
            selected_frames.append((frame, timestamp_sec))
    else:
        # Standard seek-based sampling
        duration_sec = total_frames / fps
        logger.info(f"Video stats: {total_frames} total frames, {fps:.2f} FPS, {duration_sec:.2f}s duration.")
        
        # Calculate frame indices to sample
        # We distribute the frames evenly across the duration
        indices = []
        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            # Generate max_frames evenly distributed indices
            indices = [int(i * (total_frames - 1) / (max_frames - 1)) for i in range(max_frames)]
            
        selected_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                timestamp_sec = idx / fps
                selected_frames.append((frame, timestamp_sec))
            else:
                logger.warning(f"Could not read frame at index {idx}")
        cap.release()

    # Process and encode selected frames
    processed_results = []
    for frame, ts in selected_frames:
        # Resize frame to optimize tokens
        h, w = frame.shape[:2]
        if max(h, w) > target_size:
            if w > h:
                new_w = target_size
                new_h = int(h * (target_size / w))
            else:
                new_h = target_size
                new_w = int(w * (target_size / h))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        # Encode as JPEG
        success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            logger.warning(f"Failed to encode frame at timestamp {ts:.2f}s")
            continue
            
        # Base64 encode
        b64_str = base64.b64encode(buffer).decode("utf-8")
        processed_results.append({
            "timestamp": f"{ts:.1f}s",
            "base64_image": b64_str
        })
        
    logger.info(f"Successfully extracted {len(processed_results)} frames.")
    return processed_results
