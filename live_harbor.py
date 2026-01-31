#!/usr/bin/env python3
"""
Live Harbor Detection - Port of Los Angeles EarthCam Feed
Uses ffmpeg for HLS streaming and YOLOv8 with Intel Arc GPU
"""

import subprocess
import numpy as np
import cv2
import time
import re
import requests
from collections import deque
from ultralytics import YOLO


def get_earthcam_stream_url(cam_id="portofla"):
    """Fetch fresh stream URL from EarthCam (tokens expire quickly)."""
    url = f"https://www.earthcam.com/usa/california/losangeles/port/?cam={cam_id}"
    resp = requests.get(url, timeout=10)

    # Extract stream URLs from JSON in page
    matches = re.findall(r'"stream":"([^"]+)"', resp.text)
    for m in matches:
        stream_url = m.replace('\\/', '/')
        # 23807 = San Pedro (portofla), 23808 = Wilmington (portofla2)
        if '23807' in stream_url:
            return stream_url
    # Fallback to first stream found
    if matches:
        return matches[0].replace('\\/', '/')
    raise ValueError("Could not find stream URL in page")


def create_ffmpeg_reader(stream_url, width=1280, height=720):
    """Create ffmpeg subprocess to read HLS stream."""
    cmd = [
        'ffmpeg',
        '-reconnect', '1',
        '-reconnect_streamed', '1',
        '-reconnect_delay_max', '5',
        '-headers', 'Referer: https://www.earthcam.com/',
        '-i', stream_url,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', '12',  # Limit to 12fps for stability
        '-an',  # No audio
        '-sn',  # No subtitles
        '-loglevel', 'warning',
        '-'
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)


def main():
    print("=" * 60)
    print("Port of Los Angeles - Live Harbor Detection")
    print("=" * 60)

    # Get fresh stream URL
    print("Fetching stream URL from EarthCam...")
    stream_url = get_earthcam_stream_url()
    print(f"Stream: {stream_url[:80]}...")

    # Load YOLO model (YOLOv8s for better detection)
    print("\nLoading YOLOv8s model with Intel Arc GPU...")
    model = YOLO('/home/mino/yolo/yolov8s_openvino_model', task='detect')

    # Warmup inference
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    model(dummy, device='intel:gpu', verbose=False)
    print("Model loaded and warmed up!")

    # Video settings
    width, height = 1280, 720
    frame_size = width * height * 3

    # Start ffmpeg
    print("\nConnecting to live stream...")
    process = create_ffmpeg_reader(stream_url, width, height)

    # FPS tracking
    fps_window = deque(maxlen=30)
    last_time = time.time()
    frame_count = 0
    detection_total = 0

    print("\n" + "=" * 60)
    print("LIVE DETECTION RUNNING - Press 'q' to quit, 's' for snapshot")
    print("=" * 60 + "\n")

    reconnect_attempts = 0
    max_reconnects = 10

    try:
        while True:
            try:
                # Read frame from ffmpeg
                raw = process.stdout.read(frame_size)
                if len(raw) != frame_size:
                    reconnect_attempts += 1
                    if reconnect_attempts > max_reconnects:
                        print(f"Too many reconnect attempts ({max_reconnects}), exiting...")
                        break
                    print(f"Stream ended or error, reconnecting ({reconnect_attempts}/{max_reconnects})...")
                    try:
                        process.kill()
                    except:
                        pass
                    time.sleep(3)
                    stream_url = get_earthcam_stream_url()
                    process = create_ffmpeg_reader(stream_url, width, height)
                    time.sleep(2)  # Give ffmpeg time to buffer
                    continue

                reconnect_attempts = 0  # Reset on successful read

                # Convert to numpy array
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                frame_count += 1

                # Run YOLO detection (lower confidence for night conditions)
                results = model(frame, device='intel:gpu', conf=0.15, verbose=False)[0]
            except Exception as e:
                print(f"Frame error: {e}, continuing...")
                time.sleep(1)
                continue

            # Calculate FPS
            current_time = time.time()
            fps_window.append(1.0 / (current_time - last_time + 1e-6))
            last_time = current_time
            fps = sum(fps_window) / len(fps_window)

            # Draw detections
            annotated = results.plot()
            detection_count = len(results.boxes)
            detection_total += detection_count

            # Draw overlay
            cv2.rectangle(annotated, (10, 10), (320, 100), (0, 0, 0), -1)
            cv2.rectangle(annotated, (10, 10), (320, 100), (0, 255, 0), 2)
            cv2.putText(annotated, f"FPS: {fps:.1f} (Intel Arc GPU)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated, f"Detections: {detection_count}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated, "Port of Los Angeles - LIVE", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display
            cv2.imshow("Port of LA - Live Harbor Detection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                snapshot = f"snapshot_portofla_{int(time.time())}.jpg"
                cv2.imwrite(snapshot, annotated)
                print(f"Saved: {snapshot}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        process.kill()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames, {detection_total} total detections")


if __name__ == "__main__":
    main()
