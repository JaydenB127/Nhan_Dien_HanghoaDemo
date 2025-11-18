"""Command-line entry point for the real-time retail intelligence pipeline."""
from __future__ import annotations

import argparse

from src.realtime_identifier import (
    DEFAULT_SOFT_DRINK_CLASSES,
    RealTimeProductIdentifier,
    parse_source,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the application."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the real-time product identifier. Provide an IP camera URL or use "
            "the default webcam index 0."
        )
    )
    parser.add_argument(
        "--source",
        default="0",
        type=str,
        help="Video source. Use webcam index (e.g. 0) or an IP stream URL.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated frames into img/detections.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Detection confidence threshold for YOLOv8 (0-1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8_best.pt",
        help="Path to the YOLOv8 weight file.",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=None,
        help=(
            "Optional path to a newline-delimited list of class names. When not set, "
            "the detector filters to common soft drink brands: "
            f"{', '.join(DEFAULT_SOFT_DRINK_CLASSES)}"
        ),
    )
    return parser


def main() -> None:
    """Execute the CLI workflow."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        identifier = RealTimeProductIdentifier(
            model_path=args.model,
            conf_threshold=args.confidence,
            class_names=args.class_names,
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    identifier.start_stream(parse_source(args.source), save=args.save)


if __name__ == "__main__":
    main()
