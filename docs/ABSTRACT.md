# Abstract

Vehicle number plate tampering is a growing concern in traffic management, toll evasion, and criminal investigations. Perpetrators alter characters using paint, stickers, or replacement plates to disguise vehicle identity. This project presents an **AI-enabled automated system** that detects such tampering from still images of vehicles.

The proposed system uses a three-stage pipeline: (1) license plate localization using computer vision, (2) optical character recognition with Indian plate format validation, and (3) multi-signal tampering analysis combining forensic heuristics and a convolutional neural network (CNN).

Experimental evaluation on synthetic and sample datasets demonstrates the system's ability to flag tampered plates with interpretable forensic signals. The solution is deployed as a REST API with a web dashboard, making it suitable for integration with traffic monitoring infrastructure.

**Keywords:** Number Plate Recognition, Tampering Detection, Computer Vision, OCR, Deep Learning, Forensic Analysis
