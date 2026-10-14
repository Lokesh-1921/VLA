from __future__ import annotations

from visionpilot.actions.executor import SimulatedExecutor
from visionpilot.inference.pipeline import InferencePipeline


def run_webcam_demo(pipeline: InferencePipeline, instruction: str, camera_index: int = 0) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Webcam demo requires opencv-python. Install `pip install -e '.[webcam]'`.") from exc
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")
    executor = SimulatedExecutor()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera frame capture failed")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pipeline.infer(rgb, instruction)
            accepted = executor.execute(result.demo_action, result.confidence)
            text = f"ACTION: {result.demo_action.value} | conf={result.confidence:.2f} | {result.latency_ms:.1f}ms | simulated={accepted}"
            cv2.putText(frame, instruction, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, text, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("VisionPilot — SIMULATED ACTIONS ONLY (Esc to quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        executor.emergency_stop()
        capture.release()
        cv2.destroyAllWindows()
