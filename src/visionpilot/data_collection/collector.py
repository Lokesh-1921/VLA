from __future__ import annotations

import json
import time
from pathlib import Path

from visionpilot.actions.adapter import ContinuousToDiscreteAdapter
from visionpilot.actions.schema import DiscreteAction


class WebcamCollector:
    """Keyboard-labelled custom-data collector; capture happens only after explicit Space."""

    KEY_ACTIONS = {
        ord("w"): DiscreteAction.MOVE_FORWARD, ord("a"): DiscreteAction.MOVE_LEFT,
        ord("s"): DiscreteAction.MOVE_BACKWARD, ord("d"): DiscreteAction.MOVE_RIGHT,
        ord("q"): DiscreteAction.LOOK_LEFT, ord("e"): DiscreteAction.LOOK_RIGHT, ord("x"): DiscreteAction.STOP,
    }

    def __init__(self, output: str | Path, instruction: str, camera_index: int = 0) -> None:
        self.output = Path(output)
        self.instruction, self.camera_index = instruction, camera_index
        self.frames = self.output / "frames"
        self.records = self.output / "samples.jsonl"

    def run(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Collector requires opencv-python") from exc
        self.frames.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open camera index {self.camera_index}")
        current = DiscreteAction.STOP
        count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera frame capture failed")
                message = f"{self.instruction} | label={current.value} | Space save | Esc quit"
                cv2.putText(frame, message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.imshow("VisionPilot Collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key in self.KEY_ACTIONS:
                    current = self.KEY_ACTIONS[key]
                if key == 32:
                    stamp = time.time_ns()
                    image_path = self.frames / f"{stamp}.jpg"
                    cv2.imwrite(str(image_path), frame)
                    record = {
                        "image": str(image_path), "instruction": self.instruction,
                        "action": ContinuousToDiscreteAdapter.keyboard_action(current).tolist(),
                        "episode_id": f"webcam-{stamp}", "timestep": 0, "source": "visionpilot_custom",
                        "embodiment": "webcam_simulation", "metadata": {"label": current.value, "timestamp_ns": stamp},
                    }
                    with self.records.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    count += 1
        finally:
            capture.release()
            cv2.destroyAllWindows()
        print(f"Saved {count} samples to {self.records}")
