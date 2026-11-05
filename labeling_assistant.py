"""Create relative-depth annotations from images with MediaPipe Pose.

Example:
    python labeling_assistant.py --input-dir data/raw_images
    python labeling_assistant.py --input-dir data/raw_images --interactive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

KEYPOINT_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
]
SKELETON = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 7), (2, 8), (7, 8)]
LANDMARK_INDEX = {
    name: getattr(mp.solutions.pose.PoseLandmark, name.upper()).value
    for name in KEYPOINT_NAMES
}


def _label_view(image: np.ndarray, points: list[list[float]], depths: list[float], selected: int) -> np.ndarray:
    """Render a fullscreen-friendly labeling view with editable depth values."""
    screen_width, screen_height = 1280, 720
    left_width, panel_height = screen_width // 2, screen_height // 2
    left = cv2.resize(image, (left_width, screen_height))
    top = np.full((panel_height, screen_width - left_width, 3), (28, 28, 28), dtype=np.uint8)
    bottom = top.copy()
    cv2.putText(top, "X-Y SKELETON", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (235, 235, 235), 2)
    cv2.putText(bottom, "Z-Y LABEL EDITOR", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (235, 235, 235), 2)
    width = screen_width - left_width
    for start, end in SKELETON:
        p1 = (int(points[start][0] * (width - 40) + 20), int(points[start][1] * (panel_height - 65) + 48))
        p2 = (int(points[end][0] * (width - 40) + 20), int(points[end][1] * (panel_height - 65) + 48))
        cv2.line(top, p1, p2, (150, 150, 150), 2)
    max_depth = max(max(abs(value) for value in depths), 0.25)
    for index, (x, y) in enumerate(points):
        color = (0, 220, 255) if index == selected else (50, 180, 50)
        cv2.circle(top, (int(x * (width - 40) + 20), int(y * (panel_height - 65) + 48)), 8, color, -1)
        z_x = int((0.5 + 0.42 * np.clip(depths[index] / max_depth, -1, 1)) * (width - 40) + 20)
        z_y = int(y * (panel_height - 65) + 48)
        cv2.circle(bottom, (z_x, z_y), 8, color, -1)
        cv2.putText(bottom, f"{index + 1} {KEYPOINT_NAMES[index]} {depths[index]:+.3f}", (12, 42 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1)
    for start, end in SKELETON:
        p1 = (int((0.5 + 0.42 * np.clip(depths[start] / max_depth, -1, 1)) * (width - 40) + 20), int(points[start][1] * (panel_height - 65) + 48))
        p2 = (int((0.5 + 0.42 * np.clip(depths[end] / max_depth, -1, 1)) * (width - 40) + 20), int(points[end][1] * (panel_height - 65) + 48))
        cv2.line(bottom, p1, p2, (150, 150, 150), 2)
    controls = "1-9 select | UP/DOWN +/-0.01 | SHIFT +/-0.10 | SPACE/S save-next | Q quit"
    cv2.putText(left, controls, (15, screen_height - 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((np.hstack((left[:panel_height], top)), np.hstack((left[panel_height:], bottom))))


def _relative_depth(landmarks: Any, image_width: int, image_height: int) -> tuple[list[list[float]], list[float], list[float]]:
    left_hip = landmarks[LANDMARK_INDEX["left_hip"]]
    right_hip = landmarks[LANDMARK_INDEX["right_hip"]]
    waist_z = (left_hip.z + right_hip.z) / 2.0
    shoulder_distance = max(
        np.hypot(
            (landmarks[LANDMARK_INDEX["left_shoulder"]].x - landmarks[LANDMARK_INDEX["right_shoulder"]].x) * image_width,
            (landmarks[LANDMARK_INDEX["left_shoulder"]].y - landmarks[LANDMARK_INDEX["right_shoulder"]].y) * image_height,
        ),
        1.0,
    )
    points: list[list[float]] = []
    depths: list[float] = []
    weights: list[float] = []
    for name in KEYPOINT_NAMES:
        landmark = landmarks[LANDMARK_INDEX[name]]
        points.append([float(np.clip(landmark.x, 0.0, 1.0)), float(np.clip(landmark.y, 0.0, 1.0))])
        # MediaPipe z is normalized by image width. This follows z_i-z_waist.
        depths.append(float((landmark.z - waist_z) / shoulder_distance * image_width))
        weights.append(float(np.clip(landmark.visibility, 0.1, 0.5)))
    weights[KEYPOINT_NAMES.index("left_hip")] = 1.0
    weights[KEYPOINT_NAMES.index("right_hip")] = 1.0
    return points, depths, weights


def create_annotations(input_dir: Path, output_path: Path, interactive: bool = False) -> int:
    """Detect pose landmarks and save annotations; return the number saved."""
    records: list[dict[str, Any]] = []
    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[skip] unreadable image: {image_path}")
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if not result.pose_landmarks:
                print(f"[skip] no pose detected: {image_path}")
                continue
            points, depths, weights = _relative_depth(result.pose_landmarks.landmark, image.shape[1], image.shape[0])
            if interactive:
                selected = 0
                edited: set[int] = set()
                cv2.namedWindow("Manual Relative Depth Labeling", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Manual Relative Depth Labeling", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                while True:
                    cv2.imshow("Manual Relative Depth Labeling", _label_view(image, points, depths, selected))
                    key = cv2.waitKey(0)
                    if key in (ord("q"), 27):
                        cv2.destroyAllWindows()
                        return len(records)
                    if ord("1") <= key <= ord("9"):
                        selected = key - ord("1")
                    elif key in (ord("w"), 2490368, 63232):
                        depths[selected] += 0.1 if key == ord("w") else 0.01
                        edited.add(selected)
                    elif key in (ord("s"), 2621440, 63233):
                        depths[selected] -= 0.1 if key == ord("s") else 0.01
                        edited.add(selected)
                    elif key in (ord(" "), 13, 10):
                        break
                for edited_index in edited:
                    weights[edited_index] = 1.0
            records.append(
                {
                    "image_path": str(image_path.as_posix()),
                    "keypoints_2d": points,
                    "relative_depth_gt": depths,
                    "confidence_weights": weights,
                }
            )
    if interactive:
        cv2.destroyAllWindows()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw_images"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations.json"))
    parser.add_argument("--interactive", action="store_true", help="Preview each label and press q to stop.")
    args = parser.parse_args()
    print(f"Saved {create_annotations(args.input_dir, args.output, args.interactive)} annotations.")


if __name__ == "__main__":
    main()
