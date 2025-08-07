# Dissertation Figure Generation

## Overview
This directory contains the scripts for generating publication-quality figures for the dissertation Results section.

## Main Script
- `create_dissertation_figures.py`: Generates three essential figures based on statistical analysis results

## Generated Figures

### 1. Model Comparison Box Plot (`model_comparison_boxplot.png`)
- **Purpose**: Shows overall performance distributions across three model variants
- **Key Statistics**: 
  - Baseline: M = 0.574, SD = 0.260
  - DPO-Synthetic: M = 0.564, SD = 0.231
  - DPO-Hybrid: M = 0.581, SD = 0.201
- **Visualization**: Box plots with quartiles, whiskers, outliers, and mean markers

### 2. Effect Size Forest Plot (`effect_size_forest_plot.png`)
- **Purpose**: Academic standard visualization for effect sizes with confidence intervals
- **Key Values**:
  - Baseline vs DPO-Synthetic: d = -0.040 [-0.271, 0.190]
  - Baseline vs DPO-Hybrid: d = 0.031 [-0.199, 0.261]
  - DPO-Synthetic vs DPO-Hybrid: d = 0.079 [-0.151, 0.309]
- **Interpretation**: All effect sizes are negligible (|d| < 0.2)

### 3. Model-Specific Improvements (`model_specific_improvements.png`)
- **Purpose**: Shows heterogeneous responses across different model architectures
- **Key Findings**:
  - M0001 (TinyLlama): -3.4% (Synthetic), -3.5% (Hybrid)
  - M0002 (Vicuna-7B): +12.4% (Synthetic), +16.7% (Hybrid)
  - M0003 (Phi-3): +3.8% (Synthetic), +3.1% (Hybrid)
  - M0005 (StableLM): -8.1% (Synthetic), -10.4% (Hybrid)

## Data Sources
- Primary: `results_context/master_files/statistical_values_master.json`
- Baseline data: `output/multi_topic_results/20250722_061212/complete_results.json`
- DPO-Synthetic data: `output/multi_topic_results/20250722_123509/complete_results.json`
- DPO-Hybrid data: `output/multi_topic_results/20250731_164142/complete_results.json`

## Usage
```bash
cd analysis
python create_dissertation_figures.py
```

## Output Location
All figures are saved to: `../report/figures/`

## Requirements
- Python 3.x
- matplotlib
- seaborn
- numpy
- json (standard library)
- pathlib (standard library)

## Figure Specifications
- Size: 10x6 inches (suitable for 0.8\textwidth in LaTeX)
- Resolution: 300 DPI
- Style: Academic publication standard
- Colors: Colorblind-friendly palette

## LaTeX Integration
The figures are referenced in `report/sections/results.tex` using:
```latex
\includegraphics[width=0.8\textwidth]{figures/figure_name.png}
```

## Last Updated
2025-08-06