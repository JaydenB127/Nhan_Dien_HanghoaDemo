"""Real-time product identification using YOLOv8."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Tuple, Union

import cv2  # type: ignore
from ultralytics import YOLO

DATA_LOG_PATH = Path("data/detection_log.csv")
DETECTION_IMAGE_DIR = Path("img/detections")

if TYPE_CHECKING:  # pragma: no cover - used only for static typing
    import numpy as np


@dataclass
class DetectionSummary:
    """Container for detection metrics calculated per frame."""

    counts: Dict[str, int]
    product_confidence: Dict[str, float]
    average_confidence: float


class RealTimeIdentifier:
    """Encapsulates a YOLOv8 based real-time detection pipeline.

    The class keeps the interface small so it can be orchestrated from ``main.py``
    or imported into the Streamlit dashboard. A single instance of
    :class:`RealTimeIdentifier` can be reused across different video sources.
    """

    def __init__(self, model_path: Union[str, Path] = "models/yolov8_best.pt", conf: float = 0.25) -> None:
        self.model = YOLO(str(model_path))
        self.conf = conf
        DATA_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        DETECTION_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    def process_stream(self, source: Union[int, str], save_frames: bool = False) -> None:
        """Run the detection pipeline on ``source`` until interrupted.

        Args:
            source: Camera index or IP stream URL.
            save_frames: When ``True`` the processed frames with bounding boxes are
                persisted in :data:`img/detections`.
        """

        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                annotated_frame, summary = self._process_frame(frame)
                self._log_detections(summary)

                overlay = self._render_overlay(annotated_frame, summary)
                cv2.imshow("Shelf Product Identifier", overlay)

                if save_frames:
                    self._persist_frame(overlay)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

    def _process_frame(self, frame: "np.ndarray") -> Tuple["np.ndarray", DetectionSummary]:
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        annotated = frame.copy()
        counts: Counter[str] = Counter()
        confidence_sums: Dict[str, float] = defaultdict(float)
        detections: Dict[str, int] = defaultdict(int)

        if results:
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for xyxy, conf, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
                    confidence = float(conf)
                    if confidence < self.conf:
                        continue
                    product_name = self.model.names.get(int(cls), f"cls_{int(cls)}")
                    counts[product_name] += 1
                    detections[product_name] += 1
                    confidence_sums[product_name] += confidence
                    annotated = self._draw_box(annotated, xyxy, product_name, confidence)

        average_confidence = float(sum(confidence_sums.values()) / sum(detections.values())) if detections else 0.0
        product_confidence = {
            product: confidence_sums[product] / max(detections[product], 1) for product in detections
        }
        summary = DetectionSummary(dict(counts), product_confidence, average_confidence)
        return annotated, summary

    @staticmethod
    def _draw_box(frame: "np.ndarray", xyxy: Iterable[float], product_name: str, confidence: float) -> "np.ndarray":
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{product_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

    @staticmethod
    def _render_overlay(frame: "np.ndarray", summary: DetectionSummary) -> "np.ndarray":
        overlay = frame.copy()
        y_offset = 30
        cv2.putText(
            overlay,
            f"Average confidence: {summary.average_confidence:.2f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        for product, count in summary.counts.items():
            y_offset += 30
            cv2.putText(
                overlay,
                f"{product}: {count}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
        return overlay

    def _log_detections(self, summary: DetectionSummary) -> None:
        timestamp = datetime.utcnow().isoformat()
        file_exists = DATA_LOG_PATH.exists()
        with DATA_LOG_PATH.open("a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists or DATA_LOG_PATH.stat().st_size == 0:
                writer.writerow(["timestamp", "product_name", "confidence", "count"])
            for product, count in summary.counts.items():
                confidence = summary.product_confidence.get(product, summary.average_confidence)
                writer.writerow([timestamp, product, f"{confidence:.4f}", count])

    def _persist_frame(self, frame: "np.ndarray") -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        path = DETECTION_IMAGE_DIR / f"detection_{timestamp}.jpg"
        cv2.imwrite(str(path), frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time shelf product identifier")
    parser.add_argument("--source", default=0, help="Camera index or IP stream URL", type=str)
    parser.add_argument("--conf", default=0.25, type=float, help="Confidence threshold for YOLO detections")
    parser.add_argument("--model", default="models/yolov8_best.pt", help="Path to YOLOv8 weights")
    parser.add_argument("--save", action="store_true", help="Persist annotated frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source: Union[int, str]
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    identifier = RealTimeIdentifier(model_path=args.model, conf=args.conf)
    identifier.process_stream(source=source, save_frames=args.save)


if __name__ == "__main__":
    main()
