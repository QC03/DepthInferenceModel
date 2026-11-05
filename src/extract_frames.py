"""Extract still images from a video for pose labeling."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2


def _archive_existing_frames(output_dir: Path, annotations_path: Path) -> Path | None:
    """Move the current frame set and its annotations into a numbered archive."""
    images = sorted(p for p in output_dir.glob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images and not annotations_path.is_file():
        return None
    archive_index = 1
    while (output_dir / f"pastFrame_{archive_index}").exists():
        archive_index += 1
    archive_dir = output_dir / f"pastFrame_{archive_index}"
    archive_dir.mkdir(parents=True)
    for image in images:
        shutil.move(str(image), str(archive_dir / image.name))
    if annotations_path.is_file():
        records: list[dict[str, Any]] = json.loads(annotations_path.read_text(encoding="utf-8"))
        for record in records:
            image_path = Path(str(record["image_path"]))
            record["image_path"] = (Path("data") / "raw_images" / archive_dir.name / image_path.name).as_posix()
        (archive_dir / "annotations.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        annotations_path.unlink()
    return archive_dir


def extract_frames(video: Path, output_dir: Path, every_n_frames: int, annotations_path: Path) -> int:
    """Write every Nth frame and return the number of frames written."""
    if every_n_frames < 1:
        raise ValueError("every_n_frames must be positive.")
    archived = _archive_existing_frames(output_dir, annotations_path)
    if archived:
        print(f"Archived previous frames and annotations to {archived}")
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
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations.json"))
    args = parser.parse_args()
    print(f"Extracted {extract_frames(args.video, args.output_dir, args.every_n_frames, args.annotations)} frames.")


if __name__ == "__main__":
    main()
