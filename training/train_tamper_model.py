"""Train the CNN tampering classifier on sample images."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from config.settings import settings
from ml.models.tamper_cnn import TamperCNN


class PlateDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.samples: list[tuple[Path, int]] = []
        for label, folder in enumerate(["authentic", "tampered"]):
            for image_path in sorted((root / folder).glob("*.jpg")):
                self.samples.append((image_path, label))

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 224)),
                transforms.RandomHorizontalFlip(p=0.1),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to read {path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(rgb), torch.tensor(label, dtype=torch.long)


def train(epochs: int = 12, batch_size: int = 8, learning_rate: float = 1e-3) -> None:
    dataset = PlateDataset(settings.sample_dir)
    if len(dataset) < 4:
        raise RuntimeError("Not enough samples. Run: python scripts/generate_samples.py")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if settings.use_gpu and torch.cuda.is_available() else "cpu")

    model = TamperCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / max(total, 1)
        print(f"Epoch {epoch}: loss={running_loss / total:.4f}, accuracy={accuracy:.2%}")

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), settings.model_dir / "tamper_cnn.pth")
    print(f"Saved model to {settings.model_dir / 'tamper_cnn.pth'}")


if __name__ == "__main__":
    train()
