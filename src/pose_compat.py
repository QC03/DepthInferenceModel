"""Compatibility wrapper for legacy and MediaPipe Tasks Pose APIs."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mediapipe as mp

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
MODEL_PATH = Path(".cache") / "mediapipe" / "pose_landmarker_lite.task"


def _ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as error:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            "MediaPipe Pose 모델을 다운로드하지 못했습니다. "
            f"인터넷 연결을 확인한 뒤 다시 실행하세요: {MODEL_URL}"
        ) from error
    return MODEL_PATH


class PoseEstimator:
    """Expose the old ``process``/``close`` interface on all MediaPipe versions."""

    def __init__(self, *, static_image_mode: bool, model_complexity: int) -> None:
        del model_complexity
        self._legacy = hasattr(mp, "solutions")
        if self._legacy:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=1,
            )
            self._static_image_mode = static_image_mode
            return

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(_ensure_model())),
            running_mode=(
                vision.RunningMode.IMAGE
                if static_image_mode
                else vision.RunningMode.VIDEO
            ),
            num_poses=1,
        )
        self._pose = vision.PoseLandmarker.create_from_options(options)
        self._static_image_mode = static_image_mode
        self._timestamp_ms = 0

    def process(self, image: Any) -> Any:
        if self._legacy:
            return self._pose.process(image)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image,
        )
        if self._static_image_mode:
            result = self._pose.detect(mp_image)
        else:
            self._timestamp_ms += 33
            result = self._pose.detect_for_video(mp_image, self._timestamp_ms)
        landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        return SimpleNamespace(
            pose_landmarks=SimpleNamespace(landmark=landmarks) if landmarks else None
        )

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
