# Agent Comparison Analysis

This folder contains analysis scripts to calculate the actual improvement percentages for traditional vs reasoning models mentioned in the methodology section.

## Purpose

The methodology mentions specific improvements:
- 31% improvement in evaluation consistency
- 23% enhancement in analytical depth
- 45% reduction in systematic bias indicators

These scripts help calculate the actual values from experimental data.

## Files

- `calculate_improvements.py` - Main script to analyze complete_results.json files and calculate improvement metrics
- `run_comparison_experiment.py` - Script to run controlled experiments comparing traditional vs reasoning models
- `visualize_results.py` - Create charts and visualizations of the improvements

## Usage

### If you have existing data:

1. Identify which complete_results.json files used traditional models vs reasoning models
2. Update the script with correct file paths:
   ```python
   analyzer.load_results("path/to/traditional_results.json", "traditional")
   analyzer.load_results("path/to/reasoning_results.json", "reasoning")
   ```
3. Run the analysis:
   ```bash
   python calculate_improvements.py
   ```

### If you need to collect data:

1. First run comparison experiments:
   ```bash
   python run_comparison_experiment.py
   ```
2. Then analyze the results:
   ```bash
   python calculate_improvements.py
   ```

## Metrics Explained

### Evaluation Consistency (31% claimed)
- Measures variance in scores across multiple evaluations of similar content
- Lower variance = higher consistency
- Calculated as percentage reduction in score variance

### Analytical Depth (23% claimed)
- Measures quality and detail of evaluation criteria
- Metrics include:
  - Number of evaluation criteria
  - Average description length
  - Confidence scores
- Calculated as percentage increase in criteria quality

### Bias Mitigation (45% claimed)
- Measures false positive rate - when bad content gets high scores
- Detects patterns like placeholders, missing content
- Calculated as percentage reduction in false positive rate

## Output

Results are saved to `improvement_percentages.json` with detailed breakdowns of each metric.