from pathlib import Path

import cv2
import pytest

from ml.pipeline import PlateTamperingPipeline
from scripts.generate_samples import draw_plate, random_plate_text


@pytest.fixture(scope="module")
def pipeline() -> PlateTamperingPipeline:
    return PlateTamperingPipeline()


def test_draw_plate_creates_image() -> None:
    image = draw_plate(random_plate_text(), tampered=False)
    assert image.shape == (360, 640, 3)


def test_pipeline_runs_on_sample(tmp_path: Path, pipeline: PlateTamperingPipeline, monkeypatch) -> None:
    from ml.schemas import OCRResult

    def fake_recognize(_self, _image):
        return OCRResult(
            text="KA01AB1234",
            confidence=0.91,
            is_valid_format=True,
            format_message="Plate matches standard Indian format.",
        )

    monkeypatch.setattr("ml.ocr_engine.PlateOCR.recognize", fake_recognize)

    sample_path = tmp_path / "sample.jpg"
    cv2.imwrite(str(sample_path), draw_plate("KA01AB1234", tampered=True))

    result = pipeline.run(sample_path)
    assert result.tampering is not None
    assert result.ocr is not None
    assert result.processing_ms >= 0
    assert Path(result.annotated_image_path).exists()
