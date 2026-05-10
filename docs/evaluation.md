# Evaluation

Evaluation separates system checks from quality claims.

## Gates

- MARI-AAS checks Andaluh accent/style.
- MARI-PAS checks persona behavior after the persona stage.
- Qwen-Andaluh gates track Spanish leakage, directness, technical correctness,
  preambles, repetition, and generation artifacts.

Run the Qwen-Andaluh gate:

```bash
marichatmen eval qwen-gate --samples .artifacts/reports/samples.jsonl
```

## Reports

Generated reports belong under `.artifacts/reports` or another
`MCM_ARTIFACT_ROOT`. The public repo should only keep docs and small prompt
examples.
