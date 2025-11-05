"""Realtime product identification using YOLOv8."""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


DETECTION_LOG_PATH = Path("data/detection_log.csv")
DETECTION_IMAGE_DIR = Path("img/detections")


@dataclass
class Detection:
    """Single detection result."""

    product_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]


class RealtimeIdentifier:
    """Wrapper class around YOLOv8 for realtime product identification."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = YOLO(model_path)
        DETECTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        DETECTION_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_log_header()

    def _ensure_log_header(self) -> None:
        if not DETECTION_LOG_PATH.exists() or DETECTION_LOG_PATH.stat().st_size == 0:
            with DETECTION_LOG_PATH.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "product_name", "confidence", "count"])

    def run(self, source: str, save_frames: bool = False) -> None:
        cap = self._open_video_source(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self._predict(frame)
                annotated_frame, stats = self._draw(frame, detections)

                self._log_detections(stats)

                if save_frames:
                    self._save_frame(annotated_frame)

                cv2.imshow("Shelf Product Identifier", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _open_video_source(self, source: str) -> cv2.VideoCapture:
        if source.isdigit():
            return cv2.VideoCapture(int(source))
        return cv2.VideoCapture(source)

    def _predict(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(frame, verbose=False)
        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return detections

        for box in boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            product_name = names.get(cls_id, f"class_{cls_id}")
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detections.append(Detection(product_name, conf, (x1, y1, x2, y2)))
        return detections

    def _draw(
        self, frame: np.ndarray, detections: Iterable[Detection]
    ) -> Tuple[np.ndarray, Dict[str, Tuple[float, int, float]]]:
        annotated = frame.copy()
        confidence_by_product: Dict[str, List[float]] = defaultdict(list)
        counts: Counter[str] = Counter()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            confidence = detection.confidence
            product_name = detection.product_name
            counts[product_name] += 1
            confidence_by_product[product_name].append(confidence)

            label = f"{product_name}: {confidence:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        avg_confidence = self._average_confidence(confidence_by_product)
        summary_lines = [f"Avg confidence: {avg_confidence:.2f}"]
        for product_name, count in counts.items():
            mean_conf = np.mean(confidence_by_product[product_name]) if count else 0.0
            summary_lines.append(f"{product_name}: {count} ({mean_conf:.2f})")

        for idx, line in enumerate(summary_lines):
            cv2.putText(
                annotated,
                line,
                (10, 30 + idx * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (50, 200, 255),
                2,
                cv2.LINE_AA,
            )

        stats = {
            product_name: (
                float(np.mean(confidence_by_product[product_name])),
                count,
                avg_confidence,
            )
            for product_name, count in counts.items()
        }
        return annotated, stats

    def _average_confidence(self, confidence_by_product: Dict[str, List[float]]) -> float:
        if not confidence_by_product:
            return 0.0
        all_confidences = [c for values in confidence_by_product.values() for c in values]
        return float(np.mean(all_confidences)) if all_confidences else 0.0

    def _log_detections(self, stats: Dict[str, Tuple[float, int, float]]) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with DETECTION_LOG_PATH.open("a", newline="") as file:
            writer = csv.writer(file)
            for product_name, (confidence, count, _avg) in stats.items():
                writer.writerow([timestamp, product_name, f"{confidence:.4f}", count])

    def _save_frame(self, frame: np.ndarray) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = DETECTION_IMAGE_DIR / f"detection_{timestamp}.jpg"
        cv2.imwrite(str(output_path), frame)


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
