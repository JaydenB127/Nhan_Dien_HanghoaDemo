"""Entry point for realtime shelf product identification."""
from __future__ import annotations

import argparse

from src.realtime_identifier import RealtimeIdentifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime shelf product identifier")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (webcam index or IP stream URL)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8_best.pt",
        help="Path to YOLOv8 model weights",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated frames to img/detections",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identifier = RealtimeIdentifier(args.model)
    identifier.run(args.source, save_frames=args.save)


if __name__ == "__main__":
    main()
