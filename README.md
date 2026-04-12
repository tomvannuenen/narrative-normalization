# Voice Under Revision

Replication materials for "Voice Under Revision: Large Language Models and the Normalization of Personal Narrative"

## Overview

This repository contains the code, computed linguistic markers, and statistical results to reproduce the analysis in our paper. The study examines how LLM rewriting systematically transforms personal narratives across 13 linguistic markers grounded in computational stylistics and register studies.

## Repository Structure

```
├── data/
│   ├── processed/
│   │   ├── sample.parquet                    # 300 sampled stories (originals)
│   │   ├── markers_story.parquet             # Computed markers for originals
│   │   ├── markers_*.parquet                 # Computed markers for rewrites (9 files)
│   │   ├── rewrites_*.parquet                # LLM rewrites (3 files)
│   │   └── stylometric/                      # Stylometric features for Delta analysis
│   └── robustness_tests/
│       ├── self_consistency/                 # 3-run consistency test data
│       ├── protocol/                         # System vs user message comparison
│       └── temperature/                      # Temperature sensitivity (0.0, 0.7, 1.0)
├── results/
│   ├── comparison_*.csv                      # Per-marker statistical results
│   ├── summary_*.csv                         # Dimension-level summaries
│   └── refined_marker_effects.csv            # 13-marker effect sizes
├── figures/
│   ├── fig_narrative_normalization.*         # Main figure: direction and magnitude
│   ├── fig_radar_conditions.*                # Radar plots by prompt condition
│   └── fig_stylometric_convergence.*         # PCA and Delta attribution analysis
├── src/
│   ├── config.py                             # Configuration and paths
│   ├── markers.py                            # Linguistic marker computation
│   ├── empathic_markers.py                   # Narrative-specific markers
│   ├── stylometric_markers.py                # Delta-style stylometric features
│   ├── stats.py                              # Statistical analysis
│   ├── visualize.py                          # Visualization functions
│   ├── rewriter.py                           # LLM API calls
│   └── data_loader_empathic.py               # Data loading utilities
├── scripts/
│   ├── generate_narrative_normalization_figure.py  # Main figure
│   ├── generate_radar_plot.py                      # Radar visualization
│   ├── generate_stylometric_figure.py              # PCA/Delta figure
│   ├── generate_tables.py                          # Paper tables
│   ├── compute_stylometric_markers.py              # Compute Delta features
│   ├── analyze_stylometric_convergence.py          # Delta attribution analysis
│   ├── analyze_self_consistency.py                 # Robustness: consistency
│   ├── analyze_protocol_robustness.py              # Robustness: prompt format
│   ├── analyze_temperature_sensitivity.py          # Robustness: temperature
│   ├── run_self_consistency_test.py                # Generate consistency data
│   ├── run_protocol_robustness_test.py             # Generate protocol data
│   └── run_temperature_sensitivity_test.py         # Generate temperature data
├── run_empathic_pipeline.py                  # Main analysis pipeline
└── requirements.txt
```

## Key Findings

The study measures normalization using 13 markers across four categories:

| Category | Markers | Key Finding |
|----------|---------|-------------|
| Function Words | MFW coverage, Function word ratio | Largest effects (d = -1.13 to -1.76) |
| Vocabulary | MTLD, Honoré's R, Yule's K, Word length, Trigram entropy | Systematic inflation (d = 0.91 to 1.74) |
| Syntax & Punct. | Sentence length, Comma freq, Dash freq | Elaboration increases |
| Register | Contractions, First-person pronouns, Emotion words | Voice markers deflate |

All three models (GPT-5.4, Claude Sonnet 4.6, Gemini 3.1 Pro) push all 13 markers in the same direction with 100% directional agreement.

## Data Availability

**Included in this repository:**
- Original 300 sampled stories from EmpathicStories corpus (`sample.parquet`)
- LLM-generated rewrites for all 3 models × 3 conditions (`rewrites_*.parquet`)
- All computed linguistic markers (13 markers × 300 stories × 3 models × 3 conditions)
- Stylometric features for Delta analysis
- Robustness test data (self-consistency, protocol sensitivity, temperature)
- Statistical comparison results (effect sizes, p-values)

**Source data:**
- EmpathicStories corpus: [Shen et al. (2023)](https://github.com/behavioral-data/EmpathicStories)

**Regenerating rewrites (optional):**
To regenerate rewrites from scratch (~$25-30 in API fees), see "Full Reproduction" below.

## Reproducing the Analysis

### Prerequisites

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Using Pre-computed Results

The statistical results are already computed and available in `results/`. To regenerate figures and tables:

```bash
# Generate main figure (3-panel: direction, model comparison, voice-preserving reduction)
python scripts/generate_narrative_normalization_figure.py

# Generate radar plots showing normalization pattern across conditions
python scripts/generate_radar_plot.py

# Generate stylometric convergence analysis (PCA + Delta attribution)
python scripts/generate_stylometric_figure.py

# Generate all paper tables
python scripts/generate_tables.py
```

Figures are saved to `figures/`. Table output is printed to console.

### Robustness Analyses

```bash
# Self-consistency (ICC across 3 independent runs)
python scripts/analyze_self_consistency.py

# Protocol sensitivity (system vs user message placement)
python scripts/analyze_protocol_robustness.py

# Temperature stability (0.0, 0.7, 1.0)
python scripts/analyze_temperature_sensitivity.py

# Stylometric convergence (Delta-based attribution accuracy)
python scripts/analyze_stylometric_convergence.py
```

### Technical Notes

**Sign convention:** In comparison files, positive Cohen's d means `original > rewrite` (rewrite decreases the feature). Scripts flip the sign for interpretability in figures where positive = inflation.

**13 markers:** The study uses 13 markers grounded in stylometry and register research:
- Function words: MFW coverage, Function word ratio
- Vocabulary: MTLD, Honoré's R, Yule's K, Mean word length, Character trigram entropy
- Syntax: Mean sentence length, Comma frequency, Dash frequency
- Register: Contraction density, First-person pronoun density, Emotion word density

**Direction agreement:** 100% of markers move in the same direction across all models and conditions (generic vs rewrite-only). Voice-preserving reduces magnitude by ~32% but does not change direction.

### Full Reproduction (requires API keys)

To regenerate rewrites from scratch (costs ~$25-30 in API fees):

```bash
# Set API keys
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# Run full pipeline for each condition
python run_empathic_pipeline.py --stage rewrite --prompt generic
python run_empathic_pipeline.py --stage analyze --prompt generic
python run_empathic_pipeline.py --stage compare --prompt generic

# Repeat for voice_preserving and rewrite_only conditions
```

## Citation

```bibtex
@article{vannuenen2026voice,
  title={Voice Under Revision: Large Language Models and the Normalization of Personal Narrative},
  author={van Nuenen, Tom},
  journal={Digital Scholarship in the Humanities},
  year={2026},
  publisher={Oxford University Press}
}
```

## License

Code: MIT License
Data: See EmpathicStories license for original corpus

## Contact

Tom van Nuenen
D-Lab, UC Berkeley
tomvannuenen@berkeley.edu
