#!/usr/bin/env python3
"""
Harbor Surveillance Object Detection
Uses YOLOv8 with OpenVINO backend optimized for Intel Arc GPU

Usage:
    python harbor_detector.py --source <video_url_or_path>
    python harbor_detector.py --source rtsp://camera.example.com/stream
    python harbor_detector.py --source sample.mp4
"""

import argparse
import os
import time
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional, Protocol

import cv2
from ultralytics import YOLO

# Enable verbose OpenVINO logging if needed
# os.environ["OPENVINO_LOG_LEVEL"] = "DEBUG"


# ============================================================================
# Output Handlers (extensible for S3/Kinesis integration)
# ============================================================================

class FrameHandler(Protocol):
    """Protocol for frame output handlers - implement for S3/Kinesis/etc."""
    def write(self, frame, metadata: dict) -> None: ...
    def close(self) -> None: ...


class LocalFileHandler:
    """Saves annotated frames to local directory."""

    def __init__(self, output_dir: str, save_interval: int = 30):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval  # Save every N frames
        self.frame_count = 0

    def write(self, frame, metadata: dict) -> None:
        self.frame_count += 1
        if self.frame_count % self.save_interval == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = self.output_dir / f"frame_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)

    def close(self) -> None:
        print(f"[LocalFileHandler] Saved frames to {self.output_dir}")


class S3Handler:
    """Placeholder for S3 integration - implement when ready."""

    def __init__(self, bucket: str, prefix: str = "harbor-detections/"):
        self.bucket = bucket
        self.prefix = prefix
        # TODO: Initialize boto3 client
        # import boto3
        # self.s3 = boto3.client('s3')
        print(f"[S3Handler] Would upload to s3://{bucket}/{prefix}")

    def write(self, frame, metadata: dict) -> None:
        # TODO: Implement S3 upload
        # timestamp = datetime.now().isoformat()
        # key = f"{self.prefix}{timestamp}.jpg"
        # _, buffer = cv2.imencode('.jpg', frame)
        # self.s3.put_object(Bucket=self.bucket, Key=key, Body=buffer.tobytes())
        pass

    def close(self) -> None:
        pass


class KinesisVideoHandler:
    """Placeholder for Kinesis Video Streams integration."""

    def __init__(self, stream_name: str, region: str = "us-west-2"):
        self.stream_name = stream_name
        self.region = region
        # TODO: Initialize KVS producer
        print(f"[KinesisVideoHandler] Would stream to {stream_name}")

    def write(self, frame, metadata: dict) -> None:
        # TODO: Implement KVS frame push
        pass

    def close(self) -> None:
        pass


# ============================================================================
# Main Detector Class
# ============================================================================

class HarborDetector:
    """Real-time object detection for harbor surveillance."""

    # COCO classes relevant for harbor surveillance
    HARBOR_CLASSES = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        8: 'boat',
        14: 'bird',
        15: 'cat',
        16: 'dog',
    }

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        device: str = "auto",
        handlers: Optional[list] = None
    ):
        self.confidence = confidence
        self.handlers = handlers or []
        self.device = self._select_device(device)

        # Load OpenVINO model
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path, task='detect')
        print(f"Using device: {self.device}")

        # FPS calculation
        self.fps_window = deque(maxlen=30)
        self.last_time = time.time()

    def _select_device(self, requested: str) -> str:
        """Select best available OpenVINO device."""
        try:
            from openvino import Core
            core = Core()
            available = core.available_devices
            print(f"Available OpenVINO devices: {available}")

            if requested != "auto" and requested.upper() in available:
                selected = requested.upper()
            else:
                # Prefer GPU > NPU > CPU
                selected = "CPU"
                for preferred in ["GPU", "NPU", "CPU"]:
                    if preferred in available:
                        selected = preferred
                        break

            # Return ultralytics-compatible device string
            if selected == "CPU":
                return "cpu"
            else:
                return f"intel:{selected.lower()}"
        except Exception as e:
            print(f"Device detection failed: {e}, using CPU")
            return "cpu"

    def process_frame(self, frame):
        """Run detection on a single frame."""
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )[0]

        # Calculate FPS
        current_time = time.time()
        self.fps_window.append(1.0 / (current_time - self.last_time + 1e-6))
        self.last_time = current_time
        fps = sum(self.fps_window) / len(self.fps_window)

        # Draw detections
        annotated = results.plot()
        detection_count = len(results.boxes)

        # Add overlay
        self._draw_overlay(annotated, fps, detection_count)

        # Prepare metadata
        metadata = {
            'fps': fps,
            'detection_count': detection_count,
            'timestamp': datetime.now().isoformat(),
            'detections': self._extract_detections(results)
        }

        # Send to handlers
        for handler in self.handlers:
            handler.write(annotated, metadata)

        return annotated, metadata

    def _draw_overlay(self, frame, fps: float, count: int):
        """Draw FPS and detection count overlay."""
        h, w = frame.shape[:2]

        # Background rectangle
        cv2.rectangle(frame, (10, 10), (250, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, 80), (0, 255, 0), 2)

        # Text
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {count}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _extract_detections(self, results) -> list:
        """Extract detection data for metadata."""
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                'class_id': cls_id,
                'class_name': results.names.get(cls_id, 'unknown'),
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            })
        return detections

    def run(self, source: str, display: bool = True):
        """Process video stream or file."""
        print(f"Opening video source: {source}")
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"Video: {width}x{height} @ {source_fps:.1f} FPS")

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break

                frame_count += 1
                annotated, metadata = self.process_frame(frame)

                if display:
                    cv2.imshow("Harbor Surveillance", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Quit requested")
                        break
                    elif key == ord('s'):
                        # Manual snapshot
                        snapshot_path = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(snapshot_path, annotated)
                        print(f"Saved snapshot: {snapshot_path}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            for handler in self.handlers:
                handler.close()
            print(f"Processed {frame_count} frames")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Harbor Surveillance Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --source sample.mp4
  %(prog)s --source rtsp://camera.local/stream --conf 0.6
  %(prog)s --source https://example.com/harbor.mp4 --save-dir ./detections
        """
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Video source: file path, URL, or RTSP stream"
    )
    parser.add_argument(
        "--model", "-m",
        default="/home/mino/yolov8n_openvino_model",
        help="Path to OpenVINO model directory"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--save-dir",
        default="./detections",
        help="Directory to save annotated frames"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=30,
        help="Save every N frames (default: 30)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without display window (headless mode)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="OpenVINO device: auto, CPU, GPU, NPU (default: auto)"
    )
    # Future S3/Kinesis options (commented out until implemented)
    # parser.add_argument("--s3-bucket", help="S3 bucket for frame uploads")
    # parser.add_argument("--kvs-stream", help="Kinesis Video Stream name")

    args = parser.parse_args()

    # Setup handlers
    handlers = [
        LocalFileHandler(args.save_dir, args.save_interval)
    ]

    # Create detector and run
    detector = HarborDetector(
        model_path=args.model,
        confidence=args.conf,
        device=args.device,
        handlers=handlers
    )

    detector.run(
        source=args.source,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
