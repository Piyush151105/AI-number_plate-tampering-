"""Generate synthetic sample plate images for demo and training."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from config.settings import settings

STATES = ["KA", "DL", "MH", "TN", "GJ", "UP", "RJ", "WB"]
LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")


def random_plate_text() -> str:
    return (
        f"{random.choice(STATES)}"
        f"{random.randint(1, 99):02d}"
        f"{''.join(random.choices(LETTERS, k=random.choice([2, 3])))}"
        f"{random.randint(1000, 9999)}"
    )


def draw_plate(text: str, tampered: bool = False) -> np.ndarray:
    width, height = 420, 100
    plate = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(plate, (4, 4), (width - 5, height - 5), (20, 20, 180), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    thickness = 2
    size = cv2.getTextSize(text, font, scale, thickness)[0]
    x = (width - size[0]) // 2
    y = (height + size[1]) // 2
    cv2.putText(plate, text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    if tampered:
        # Simulate pasted character / paint tampering
        patch_x = random.randint(80, width - 140)
        cv2.rectangle(plate, (patch_x, 25), (patch_x + 55, 75), (245, 245, 245), -1)
        fake_char = random.choice(LETTERS + [str(random.randint(0, 9))])
        cv2.putText(plate, fake_char, (patch_x + 12, 68), font, 1.3, (30, 30, 30), 2)
        noise = np.random.normal(0, 18, plate.shape).astype(np.int16)
        plate = np.clip(plate.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Place plate on a simple car rear background
    canvas = np.full((360, 640, 3), 55, dtype=np.uint8)
    y_offset = 200
    x_offset = (canvas.shape[1] - width) // 2
    canvas[y_offset : y_offset + height, x_offset : x_offset + width] = plate
    return canvas


def generate_samples(count_per_class: int = 10) -> None:
    authentic_dir = settings.sample_dir / "authentic"
    tampered_dir = settings.sample_dir / "tampered"
    authentic_dir.mkdir(parents=True, exist_ok=True)
    tampered_dir.mkdir(parents=True, exist_ok=True)

    for index in range(count_per_class):
        text = random_plate_text()
        authentic = draw_plate(text, tampered=False)
        tampered = draw_plate(text, tampered=True)
        cv2.imwrite(str(authentic_dir / f"authentic_{index:03d}.jpg"), authentic)
        cv2.imwrite(str(tampered_dir / f"tampered_{index:03d}.jpg"), tampered)

    print(f"Generated {count_per_class} authentic and {count_per_class} tampered samples in {settings.sample_dir}")


if __name__ == "__main__":
    generate_samples()
