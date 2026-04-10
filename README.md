# Narrative Normalization

Replication materials for "Narrative Normalization: How Large Language Models Reshape Personal Stories"

## Overview

This repository contains the code, computed linguistic markers, and statistical results to reproduce the analysis in our paper.

## Repository Structure

```
├── data/
│   └── processed/
│       ├── sample.parquet              # 300 sampled stories (originals)
│       ├── markers_story.parquet       # Computed markers for originals
│       └── markers_*.parquet           # Computed markers for rewrites
├── results/
│   ├── comparison_*.csv                # Per-marker statistical results
│   └── summary_*.csv                   # Dimension-level summaries
├── figures/
│   └── fig_narrative_normalization.*   # Main figure: Normalization effects
├── src/
│   ├── config.py                       # Configuration and paths
│   ├── markers.py                      # Linguistic marker computation
│   ├── empathic_markers.py             # Narrative-specific markers
│   ├── stats.py                        # Statistical analysis
│   ├── visualize.py                    # Visualization functions
│   ├── rewriter.py                     # LLM API calls
│   └── data_loader_empathic.py         # Data loading utilities
├── scripts/
│   ├── generate_narrative_normalization_figure.py  # Main figure
│   └── generate_tables.py                          # Paper tables
├── run_empathic_pipeline.py            # Main analysis pipeline
└── requirements.txt
```

## Data Availability

**Included in this repository:**
- Original 300 sampled stories from EmpathicStories corpus
- All computed linguistic markers (49 markers × 300 stories × 3 models × 3 conditions)
- Statistical comparison results (effect sizes, p-values)

**Not included (API Terms of Service):**
- LLM-generated rewrites cannot be redistributed
- Rewrites can be regenerated using the provided code with API access

**Source data:**
- EmpathicStories corpus: [Shen et al. (2023)](https://github.com/behavioral-data/EmpathicStories)

## Reproducing the Analysis

### Prerequisites

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Using Pre-computed Results

The statistical results are already computed and available in `results/`. To regenerate figure and tables:

```bash
# Generate main figure
python scripts/generate_narrative_normalization_figure.py

# Generate all paper tables
python scripts/generate_tables.py
```

Figure is saved to `figures/`. Table output is printed to console.

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
@article{vannuenen2026narrative,
  title={Narrative Normalization: How Large Language Models Reshape Personal Stories},
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
