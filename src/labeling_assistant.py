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
from PIL import Image, ImageDraw, ImageFont

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
KOREAN_NAMES = ["코", "왼쪽 어깨", "오른쪽 어깨", "왼쪽 팔꿈치", "오른쪽 팔꿈치", "왼쪽 손목", "오른쪽 손목", "왼쪽 골반", "오른쪽 골반"]
SKELETON = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 7), (2, 8), (7, 8)]
LANDMARK_INDEX = {
    name: getattr(mp.solutions.pose.PoseLandmark, name.upper()).value
    for name in KEYPOINT_NAMES
}


def _draw_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    size: int = 18,
    color: tuple[int, int, int] = (240, 240, 240),
    stroke_width: int = 2,
) -> None:
    """Draw text with Korean glyphs and a black outline."""
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", size)
        canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(canvas).text(
            position,
            text,
            font=font,
            fill=color[::-1],
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0),
        )
        image[:] = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    except (OSError, ValueError):
        fallback = text.encode("ascii", "replace").decode()
        cv2.putText(image, fallback, (position[0] + 1, position[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, fallback, position, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _label_view(image: np.ndarray, points: list[list[float]], depths: list[float], selected: int, processed: int, remaining: int) -> np.ndarray:
    """Render a fullscreen-friendly labeling view with editable depth values."""
    screen_width, screen_height = 1280, 720
    left_width, panel_height = screen_width // 2, screen_height // 2
    left_top = cv2.resize(image, (left_width, panel_height))
    left_bottom = np.full((panel_height, left_width, 3), (28, 28, 28), dtype=np.uint8)
    right_top = np.full((panel_height, left_width, 3), (28, 28, 28), dtype=np.uint8)
    right_bottom = right_top.copy()
    _draw_text(left_top, "원본 + X-Y", (18, 30), 22)
    _draw_text(left_bottom, "Z-X 스켈레톤", (18, 30), 22)
    _draw_text(right_top, "X-Y 스켈레톤", (18, 30), 22)
    _draw_text(right_bottom, "Z-Y 깊이", (18, 30), 22)
    width = left_width
    panel_scale = (width - 40, panel_height - 65)
    for start, end in SKELETON:
        p1 = (int(points[start][0] * panel_scale[0] + 20), int(points[start][1] * panel_scale[1] + 48))
        p2 = (int(points[end][0] * panel_scale[0] + 20), int(points[end][1] * panel_scale[1] + 48))
        cv2.line(left_top, p1, p2, (150, 150, 150), 2)
        cv2.line(right_top, p1, p2, (150, 150, 150), 2)
    max_depth = max(max(abs(value) for value in depths), 0.25)
    for index, (x, y) in enumerate(points):
        color = (0, 220, 255) if index == selected else (50, 180, 50)
        xy_point = (int(x * panel_scale[0] + 20), int(y * panel_scale[1] + 48))
        cv2.circle(left_top, xy_point, 8, color, -1)
        cv2.circle(right_top, xy_point, 8, color, -1)
        _draw_text(left_top, KOREAN_NAMES[index], (xy_point[0] + 10, xy_point[1] - 8), 16, (255, 255, 255))
        _draw_text(right_top, KOREAN_NAMES[index], (xy_point[0] + 10, xy_point[1] - 8), 16, (255, 255, 255))
        # Z-X is a top-down view: horizontal image X, vertical depth Z.
        zx_x = int(x * panel_scale[0] + 20)
        zx_y = int((0.5 + 0.42 * np.clip(depths[index] / max_depth, -1, 1)) * panel_scale[1] + 48)
        cv2.circle(left_bottom, (zx_x, zx_y), 8, color, -1)
        # Z-Y keeps image Y vertical and uses depth Z horizontally.
        zy_x = int((0.5 + 0.42 * np.clip(depths[index] / max_depth, -1, 1)) * panel_scale[0] + 20)
        zy_y = int(y * panel_scale[1] + 48)
        cv2.circle(right_bottom, (zy_x, zy_y), 8, color, -1)
        _draw_text(right_bottom, f"{index + 1} {KOREAN_NAMES[index]} {depths[index]:+.3f}", (12, 58 + index * 24), 15)
    for start, end in SKELETON:
        zx_p1 = (int(points[start][0] * panel_scale[0] + 20), int((0.5 + 0.42 * np.clip(depths[start] / max_depth, -1, 1)) * panel_scale[1] + 48))
        zx_p2 = (int(points[end][0] * panel_scale[0] + 20), int((0.5 + 0.42 * np.clip(depths[end] / max_depth, -1, 1)) * panel_scale[1] + 48))
        zy_p1 = (int((0.5 + 0.42 * np.clip(depths[start] / max_depth, -1, 1)) * panel_scale[0] + 20), int(points[start][1] * panel_scale[1] + 48))
        zy_p2 = (int((0.5 + 0.42 * np.clip(depths[end] / max_depth, -1, 1)) * panel_scale[0] + 20), int(points[end][1] * panel_scale[1] + 48))
        cv2.line(left_bottom, zx_p1, zx_p2, (150, 150, 150), 2)
        cv2.line(right_bottom, zy_p1, zy_p2, (150, 150, 150), 2)
    _draw_text(left_top, f"처리: {processed} | 남은 프레임: {remaining}", (18, 58), 18, (0, 255, 255))
    _draw_text(left_bottom, "1-9 선택 | 드래그: X-Y 이동 | 위/아래 +/-0.01 | W/S +/-0.10", (15, screen_height // 2 - 18), 15, (0, 255, 255))
    return np.vstack((np.hstack((left_top, right_top)), np.hstack((left_bottom, right_bottom))))


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
        for frame_index, image_path in enumerate(image_paths):
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
                selected_state = {"index": 0}
                edited: set[int] = set()
                drag_state = {"index": None}
                cv2.namedWindow("Manual Relative Depth Labeling", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Manual Relative Depth Labeling", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
                    if x >= 640 or y >= 360:
                        if event == cv2.EVENT_LBUTTONUP:
                            drag_state["index"] = None
                        return
                    normalized = ((x - 20) / 600.0, (y - 48) / 295.0)
                    if event == cv2.EVENT_LBUTTONDOWN:
                        distances = [((point[0] - normalized[0]) ** 2 + (point[1] - normalized[1]) ** 2, index) for index, point in enumerate(points)]
                        nearest_distance, nearest_index = min(distances)
                        if nearest_distance < 0.04:
                            selected_state["index"] = nearest_index
                            drag_state["index"] = nearest_index
                    elif event == cv2.EVENT_MOUSEMOVE and drag_state["index"] is not None:
                        point_index = int(drag_state["index"])
                        points[point_index][0] = float(np.clip(normalized[0], 0.0, 1.0))
                        points[point_index][1] = float(np.clip(normalized[1], 0.0, 1.0))
                        edited.add(point_index)
                    elif event == cv2.EVENT_LBUTTONUP:
                        drag_state["index"] = None
                cv2.setMouseCallback("Manual Relative Depth Labeling", on_mouse)
                while True:
                    selected = int(selected_state["index"])
                    cv2.imshow("Manual Relative Depth Labeling", _label_view(image, points, depths, selected, frame_index + 1, len(image_paths) - frame_index - 1))
                    key = cv2.waitKey(30)
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
