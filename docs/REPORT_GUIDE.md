# Final Year Project Report Sections

Use these files as starting content for your project report / PPT.

## Suggested Report Outline

1. **Introduction** — Problem, motivation, scope
2. **Literature Survey** — ANPR, tampering detection papers
3. **System Analysis** — Requirements (functional/non-functional)
4. **System Design** — See `SYSTEM_ARCHITECTURE.md`
5. **Implementation** — Module-wise description
6. **Methodology** — See `METHODOLOGY.md`
7. **Results & Discussion** — Accuracy, sample outputs, limitations
8. **Conclusion & Future Work**
9. **References**
10. **Appendix** — Code snippets, API docs, screenshots

## Demo Checklist for Viva

- [ ] Show project folder structure
- [ ] Explain plate detection algorithm
- [ ] Explain OCR + Indian format validation
- [ ] Explain each tampering signal
- [ ] Run API via `python run_api.py`
- [ ] Upload authentic vs tampered sample in dashboard
- [ ] Show annotated output image
- [ ] Discuss CNN training (`training/train_tamper_model.py`)
- [ ] Mention real-world improvements (YOLO, CCTV integration)

## Sample Viva Questions

**Q: Why combine heuristics and CNN?**  
A: Heuristics are interpretable for forensic reports; CNN captures complex tampering patterns heuristics miss.

**Q: How do you handle low-light images?**  
A: Preprocessing (bilateral filter, Otsu threshold). Future: train on augmented low-light data.

**Q: Can this work in real time?**  
A: Current design is batch image analysis. With GPU + YOLO optimization, 5–15 FPS is achievable.
