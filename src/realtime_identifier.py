"""Real-time product identification module using YOLOv8.

This module encapsulates the logic required to run a YOLOv8 model on a
video stream (webcam or IP camera) and log the resulting detections to disk.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path("models/yolov8_best.pt")
DEFAULT_LOG_PATH = Path("data/detection_log.csv")
DEFAULT_SAVE_DIR = Path("img/detections")


class RealTimeProductIdentifier:
    """High level interface for streaming detections with YOLOv8.

    Parameters
    ----------
    model_path:
        Path to the trained YOLOv8 model weights.
    log_path:
        CSV file path used to persist detection information.
    save_dir:
        Directory where annotated frames are saved when the ``save`` option
        is enabled.
    conf_threshold:
        Minimum confidence threshold for retaining detections.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        log_path: Union[str, Path] = DEFAULT_LOG_PATH,
        save_dir: Union[str, Path] = DEFAULT_SAVE_DIR,
        conf_threshold: float = 0.4,
    ) -> None:
        self.model_path = Path(model_path)
        self.log_path = Path(log_path)
        self.save_dir = Path(save_dir)
        self.conf_threshold = conf_threshold

        self._ensure_directories()
        self.model = self._load_model(self.model_path)

    def _ensure_directories(self) -> None:
        """Create output directories and seed the detection log if required."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self.log_path.open("w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["timestamp", "product_name", "confidence", "count"])

    @staticmethod
    def _load_model(model_path: Path) -> YOLO:
        """Load the YOLO model from disk.

        Parameters
        ----------
        model_path:
            Path to the YOLO weights file.

        Returns
        -------
        YOLO
            Instantiated YOLO model ready for inference.
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {model_path}. Please place the trained "
                "model in the models/ directory."
            )
        return YOLO(model_path.as_posix())

    def start_stream(self, source: Union[int, str], save: bool = False) -> None:
        """Start the real-time detection loop.

        Parameters
        ----------
        source:
            Either an integer representing the webcam index or a string with the
            URL to an IP camera stream.
        save:
            When ``True`` the annotated frames are stored under ``self.save_dir``.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        frame_index = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections, average_confidence = self._infer_frame(frame)
                self._log_detections(detections)
                annotated = self._draw_detections(frame, detections, average_confidence)

                if save:
                    output_path = self.save_dir / f"frame_{frame_index:06d}.jpg"
                    cv2.imwrite(output_path.as_posix(), annotated)

                cv2.imshow("Shelf Product Identifier", annotated)
                frame_index += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _infer_frame(self, frame: np.ndarray) -> Tuple[Dict[str, int], float]:
        """Run inference on a single frame and return detection statistics."""
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        if not results:
            return {}, 0.0

        result = results[0]
        boxes = result.boxes
        names = result.names or self.model.names
        detections: Dict[str, int] = Counter()
        confidences: list[float] = []

        if boxes is None or len(boxes) == 0:
            self._last_result_boxes = None
            self._last_result_names = names
            return {}, 0.0

        for box in boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            cls_id = int(box.cls[0])
            label = self._label_from_names(names, cls_id)
            detections[label] += 1
            confidences.append(conf)

        average_confidence = float(np.mean(confidences)) if confidences else 0.0
        self._last_result_boxes = boxes
        self._last_result_names = names
        return detections, average_confidence

    def _log_detections(self, detections: Dict[str, int]) -> None:
        """Append the detections of the current frame to the CSV log."""
        if not detections:
            return

        timestamp = datetime.utcnow().isoformat()
        with self.log_path.open("a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            for label, count in detections.items():
                confidence = 0.0
                if hasattr(self, "_last_result_boxes"):
                    confidence = self._mean_confidence_for_label(label)
                writer.writerow([timestamp, label, confidence, count])

    def _mean_confidence_for_label(self, label: str) -> float:
        """Compute the mean confidence for a given label in the last frame."""
        boxes = getattr(self, "_last_result_boxes", None)
        names = getattr(self, "_last_result_names", {})
        confidences = []
        if boxes is None:
            return 0.0

        for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                if self._label_from_names(names, cls_id) == label:
                    confidences.append(float(box.conf[0]))
        return float(np.mean(confidences)) if confidences else 0.0

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: Dict[str, int],
        average_confidence: float,
    ) -> np.ndarray:
        """Overlay bounding boxes, labels and summary statistics on the frame."""
        annotated = frame.copy()
        boxes = getattr(self, "_last_result_boxes", None)
        names = getattr(self, "_last_result_names", {})

        if boxes is not None:
            for idx, box in enumerate(boxes):
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue
                cls_id = int(box.cls[0])
                label = self._label_from_names(names, cls_id)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label}: {conf:.2f}"
                cv2.putText(
                    annotated,
                    text,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        summary_y = 20
        cv2.putText(
            annotated,
            f"Avg confidence: {average_confidence:.2f}",
            (10, summary_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        summary_y += 25

        for label, count in detections.items():
            cv2.putText(
                annotated,
                f"{label}: {count}",
                (10, summary_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            summary_y += 25

        return annotated

    @staticmethod
    def _label_from_names(names: Union[Dict[int, str], Tuple[str, ...], list[str]], cls_id: int) -> str:
        """Resolve a class id into a readable label."""
        if isinstance(names, dict):
            return names.get(cls_id, f"class_{cls_id}")
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return str(names[cls_id])
        return f"class_{cls_id}"


def parse_source(source: str) -> Union[int, str]:
    """Convert the ``--source`` argument into a value understood by OpenCV."""
    if source.isdigit():
        return int(source)
    return source


def main() -> None:
    """Command-line entry point for the real-time identifier."""
    parser = argparse.ArgumentParser(description="Run the real-time product identifier")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source. Use an integer for webcam index or an IP camera URL.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist annotated frames under img/detections.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Minimum detection confidence threshold (0-1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH.as_posix(),
        help="Path to the YOLOv8 model weights.",
    )
    args = parser.parse_args()

    identifier = RealTimeProductIdentifier(
        model_path=args.model,
        conf_threshold=args.confidence,
    )
    identifier.start_stream(parse_source(args.source), save=args.save)


if __name__ == "__main__":
    main()
