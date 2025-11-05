"""Entry point for the real-time Shelf Product Identifier."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

from src.realtime_identifier import RealTimeIdentifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the retail intelligence vision pipeline")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or IP stream URL (e.g. http://192.168.x.x:8080/video)",
    )
    parser.add_argument("--model", default="models/yolov8_best.pt", help="Path to YOLOv8 model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument("--save", action="store_true", help="Persist annotated frames to img/detections")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source: Union[int, str]
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    model_path = Path(args.model)
    identifier = RealTimeIdentifier(model_path=model_path, conf=args.conf)
    identifier.process_stream(source=source, save_frames=args.save)


if __name__ == "__main__":
    main()
