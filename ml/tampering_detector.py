"""Multi-signal tampering analysis engine."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from config.settings import settings
from ml.feature_extractor import (
    background_smudge_score,
    character_alignment_score,
    color_inconsistency,
    edge_density,
    normalize_plate,
    smudge_patch_score,
    texture_variance,
)
from ml.models.tamper_cnn import TamperCNN
from ml.schemas import OCRResult, TamperAnalysis, TamperSignal


class TamperingDetector:
    def __init__(self) -> None:
        self.threshold = settings.tamper_threshold
        self.model = TamperCNN()
        self.model_path = settings.model_dir / "tamper_cnn.pth"
        self.device = torch.device("cuda" if settings.use_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self._load_weights()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def analyze(self, plate_image: np.ndarray, ocr: OCRResult | None = None) -> TamperAnalysis:
        signals: list[TamperSignal] = []

        edge_score = edge_density(plate_image)
        signals.append(
            TamperSignal(
                name="Edge Anomaly",
                score=edge_score,
                description="Unusually high edge density may indicate pasted or altered characters.",
                severity="high" if edge_score > 0.22 else "low",
            )
        )

        texture_score = texture_variance(plate_image)
        signals.append(
            TamperSignal(
                name="Texture Inconsistency",
                score=texture_score,
                description="Irregular local texture patterns across the plate surface.",
                severity="medium" if texture_score > 0.015 else "low",
            )
        )

        alignment_score = character_alignment_score(plate_image)
        signals.append(
            TamperSignal(
                name="Character Misalignment",
                score=alignment_score,
                description="Characters not aligned on a common baseline suggest manual tampering.",
                severity="high" if alignment_score > 0.35 else "low",
            )
        )

        color_score = color_inconsistency(plate_image)
        signals.append(
            TamperSignal(
                name="Color Mismatch",
                score=color_score,
                description="Left/right color differences can indicate partial plate replacement.",
                severity="medium" if color_score > 0.25 else "low",
            )
        )

        patch_score = smudge_patch_score(plate_image)
        signals.append(
            TamperSignal(
                name="Dark Patch / Tape",
                score=patch_score,
                description="Oversized irregular dark blobs suggest tape, paint, or sticker tampering.",
                severity="high" if patch_score > 0.5 else "low",
            )
        )

        smudge_score = background_smudge_score(plate_image)
        signals.append(
            TamperSignal(
                name="Background Smudge",
                score=smudge_score,
                description="Uneven tone or gray residue on the plate background around characters.",
                severity="high" if smudge_score > 0.5 else "low",
            )
        )

        cnn_score = self._cnn_probability(plate_image)
        signals.append(
            TamperSignal(
                name="CNN Tamper Classifier",
                score=cnn_score,
                description="Deep learning model trained to distinguish authentic vs tampered plates.",
                severity="high" if cnn_score > 0.65 else "low",
            )
        )

        if ocr is not None:
            ocr_risk = max(0.0, 0.65 - ocr.confidence)
            if not ocr.text.strip():
                ocr_risk = max(ocr_risk, 0.55)
            elif not ocr.is_valid_format:
                ocr_risk = max(ocr_risk, 0.25)
            signals.append(
                TamperSignal(
                    name="OCR Confidence / Format",
                    score=ocr_risk,
                    description="Very low OCR confidence or unreadable plate text increases suspicion.",
                    severity="high" if ocr_risk > 0.45 else "low",
                )
            )

        weights = {
            "Edge Anomaly": 0.08,
            "Texture Inconsistency": 0.06,
            "Character Misalignment": 0.08,
            "Color Mismatch": 0.06,
            "Dark Patch / Tape": 0.28,
            "Background Smudge": 0.18,
            "CNN Tamper Classifier": 0.16,
            "OCR Confidence / Format": 0.10,
        }
        tamper_score = sum(signal.score * weights.get(signal.name, 0.1) for signal in signals)
        tamper_score = float(np.clip(tamper_score, 0.0, 1.0))

        high_signals = sum(1 for signal in signals if signal.severity == "high")
        forensic_tamper = patch_score >= 0.52 and smudge_score >= 0.48
        strong_patch = patch_score >= 0.62
        cnn_and_patch = cnn_score >= 0.68 and patch_score >= 0.35

        is_tampered = (
            tamper_score >= self.threshold
            and (forensic_tamper or strong_patch or cnn_and_patch or high_signals >= 3)
        ) or (patch_score >= 0.72 and smudge_score >= 0.55)

        verdict = (
            "TAMPERED — Multiple forensic indicators suggest plate alteration."
            if is_tampered
            else "AUTHENTIC — No strong tampering indicators detected."
        )
        return TamperAnalysis(
            tamper_score=round(tamper_score, 4),
            is_tampered=is_tampered,
            verdict=verdict,
            signals=signals,
        )

    def _cnn_probability(self, plate_image: np.ndarray) -> float:
        rgb = cv2.cvtColor(normalize_plate(plate_image), cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        return float(probs[1].item())

    def _load_weights(self) -> None:
        if self.model_path.exists():
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            return

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
