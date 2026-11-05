"""Run relative-depth inference with a fullscreen 2x2 visualization UI."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from model import ResNet18RelativeDepthEstimator

KEYPOINT_NAMES = ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip"]
MEDIAPIPE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]
SKELETON = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 7), (2, 8), (7, 8)]


def predict(model: torch.nn.Module, frame: np.ndarray, device: torch.device) -> np.ndarray:
    resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (224, 224))
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().div(255.0)
    tensor = ((tensor - 0.5) / 0.5).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(tensor).squeeze(0).cpu().numpy()


def _fit_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _panel(title: str, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), (28, 28, 28), dtype=np.uint8)
    cv2.putText(panel, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
    return panel


def compose_view(frame: np.ndarray, depths: np.ndarray, landmarks: object | None, window_size: tuple[int, int]) -> np.ndarray:
    """Build the requested fullscreen layout: video left, x-y/z-y right."""
    screen_width, screen_height = window_size
    left_width = screen_width // 2
    right_width = screen_width - left_width
    panel_height = screen_height // 2
    left = _fit_image(frame, (left_width, screen_height))
    top = _panel("X-Y SKELETON", right_width, panel_height)
    bottom = _panel("Z-Y RELATIVE DEPTH", right_width, screen_height - panel_height)
    points: list[tuple[float, float]] = []
    if landmarks is not None:
        points = [(float(landmarks.landmark[i].x), float(landmarks.landmark[i].y)) for i in MEDIAPIPE_INDICES]
        for start, end in SKELETON:
            p1 = (int(points[start][0] * (right_width - 40) + 20), int(points[start][1] * (panel_height - 65) + 48))
            p2 = (int(points[end][0] * (right_width - 40) + 20), int(points[end][1] * (panel_height - 65) + 48))
            cv2.line(top, p1, p2, (150, 150, 150), 2)
    depth_range = max(float(np.max(np.abs(depths))), 0.25)
    for index, depth in enumerate(depths):
        if index >= len(points):
            break
        x, y = points[index]
        color = (40, int(np.clip(180 - depth * 100, 40, 220)), int(np.clip(180 + depth * 100, 40, 255)))
        cv2.circle(top, (int(x * (right_width - 40) + 20), int(y * (panel_height - 65) + 48)), 7, color, -1)
        z_x = int((0.5 + 0.42 * float(np.clip(depth / depth_range, -1.0, 1.0))) * (right_width - 40) + 20)
        z_y = int(y * (screen_height - panel_height - 65) + 48)
        cv2.circle(bottom, (z_x, z_y), 7, color, -1)
        cv2.putText(bottom, f"{index + 1}:{KEYPOINT_NAMES[index]} {depth:+.3f}", (12, 42 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    for start, end in SKELETON:
        if len(points) > max(start, end):
            p1 = (int((0.5 + 0.42 * np.clip(depths[start] / depth_range, -1, 1)) * (right_width - 40) + 20), int(points[start][1] * (bottom.shape[0] - 65) + 48))
            p2 = (int((0.5 + 0.42 * np.clip(depths[end] / depth_range, -1, 1)) * (right_width - 40) + 20), int(points[end][1] * (bottom.shape[0] - 65) + 48))
            cv2.line(bottom, p1, p2, (150, 150, 150), 2)
    cv2.line(bottom, (right_width // 2, 42), (right_width // 2, bottom.shape[0]), (80, 80, 80), 1)
    return np.vstack((np.hstack((left[:panel_height], top)), np.hstack((left[panel_height:], bottom))))


def _next_output(output_dir: Path, suffix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    while (output_dir / f"result_{index}{suffix}").exists():
        index += 1
    return output_dir / f"result_{index}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--output", type=Path, help="Optional explicit output path.")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ResNet18RelativeDepthEstimator(checkpoint["num_keypoints"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open {args.input}")
    is_video = args.input.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    output_path = args.output or _next_output(args.output_dir, ".mp4" if is_video else ".jpg")
    screen_size = (1280, 720)
    writer: cv2.VideoWriter | None = None
    if is_video:
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), capture.get(cv2.CAP_PROP_FPS) or 30.0, (screen_size[0], screen_size[1]))
    if not args.no_display:
        cv2.namedWindow("Relative Depth Studio", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Relative Depth Studio", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        depths = predict(model, frame, device)
        view = compose_view(frame, depths, result.pose_landmarks, screen_size)
        if writer is not None:
            writer.write(view)
        if not args.no_display:
            cv2.imshow("Relative Depth Studio", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if not is_video:
            cv2.imwrite(str(output_path), view)
            break
    capture.release()
    pose.close()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"Saved result: {output_path}")


if __name__ == "__main__":
    main()
