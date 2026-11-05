"""Extract still images from a video for pose labeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(video: Path, output_dir: Path, every_n_frames: int) -> int:
    """Write every Nth frame and return the number of frames written."""
    if every_n_frames < 1:
        raise ValueError("every_n_frames must be positive.")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open {video}")
    output_dir.mkdir(parents=True, exist_ok=True)
    index = written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % every_n_frames == 0:
            output_path = output_dir / f"frame_{written:04d}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                capture.release()
                raise OSError(f"Unable to write {output_path}")
            written += 1
        index += 1
    capture.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw_images"))
    parser.add_argument("--every-n-frames", type=int, default=6)
    args = parser.parse_args()
    print(f"Extracted {extract_frames(args.video, args.output_dir, args.every_n_frames)} frames.")


if __name__ == "__main__":
    main()
