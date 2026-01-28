# Citation

If you use this work in your research, please cite:

```bibtex
@misc{chinese-ai-detector-2026,
  title={Chinese AI-Generated Text Detection with Boundary Markers},
  author={AnxForever},
  year={2026},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/AnxForever/ai-text-detector-mix}},
  note={A Chinese AI text detection system with 98.71\% accuracy, featuring boundary marker mechanism and token-level boundary detection}
}
```

## Key Contributions

1. **Boundary Marker Mechanism**: Introduced `[SEP]` token to explicitly mark human-AI boundaries, improving C2 detection by 14%

2. **Two-Stage Detection Architecture**: Combined coarse-grained classification with fine-grained boundary localization

3. **Token-Level Boundary Detection**: Achieved 96.69% token classification accuracy with <10 character actual error

## Performance Metrics

- Overall Accuracy: **98.71%**
- C2 (Continuation) Detection: **93.84%** (improved from 79.82%)
- C3 (Rewriting) Detection: **100%**
- C4 (Polishing) Detection: **92.89%**
- Token Classification: **96.69%**
- Boundary Localization: **49.51%** (±5 tokens)

## Related Work

This work builds upon:
- BERT (Devlin et al., 2019)
- RoBERTa (Liu et al., 2019)
- Chinese BERT (Cui et al., 2020)
- DetectGPT and related AI text detection methods

## Contact

For questions or collaboration:
- GitHub Issues: https://github.com/AnxForever/ai-text-detector-mix/issues
- Email: [To be added]

## License

- Code: MIT License
- Models: MIT License
- Dataset: CC BY-NC-SA 4.0 (Academic use only)
