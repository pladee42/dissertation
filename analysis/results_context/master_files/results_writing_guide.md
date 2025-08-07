# Results Section Writing Guide
**Generated**: 2025-01-31  
**Purpose**: Complete reference for Stage 6 Results section writing  

## Statistical Values for Text

### Opening Overview Statistics
- Sample size: N = 145 (balanced design)
- Three model variants: Baseline, DPO-Synthetic, DPO-Hybrid
- Overall score range: 0.000 to 1.000
- No missing data, complete evaluation coverage

### Descriptive Statistics (for Table 1 reference)
- Baseline: M = 0.574, SD = 0.260
- DPO-Synthetic: M = 0.564, SD = 0.231
- DPO-Hybrid: M = 0.581, SD = 0.201

### Statistical Test Results (for Table 2 reference)
- ANOVA: F(2, 432) = 0.199, p = 0.820, η² = 0.001
- All pairwise comparisons: p > 0.05 (non-significant)
- All effect sizes: |d| < 0.2 (negligible)

### Effect Size Details
- Baseline vs DPO-Synthetic: d = -0.040, 95% CI [-0.271, 0.19]
- Baseline vs DPO-Hybrid: d = 0.031, 95% CI [-0.199, 0.261]  
- DPO-Synthetic vs DPO-Hybrid: d = 0.079, 95% CI [-0.151, 0.309]

## LaTeX Table References (Ready for Copy-Paste)

### Table References for Text
```latex
Table~\ref{tab:descriptive-statistics} presents descriptive statistics...
Table~\ref{tab:statistical-comparisons} shows the pairwise comparisons...
Table~\ref{tab:model-specific} details individual model performance...
Table~\ref{tab:category-analysis} presents category-based results...
Table~\ref{tab:methodology-validation} documents validation outcomes...
```

### Figure References for Text
```latex
Figure~\ref{fig:effect-size-forest} displays effect sizes with confidence intervals...
Figure~\ref{fig:model-comparison} shows the distribution overlap...
Figure~\ref{fig:anova-summary} presents ANOVA results...
Figure~\ref{fig:methodology-validation} documents prediction failures...
```

## Section Structure Template

### 1. Opening Overview
```latex
Statistical analysis of the three model variants (Baseline, DPO-Synthetic, DPO-Hybrid) 
was conducted on N = 145 email evaluations using a balanced design. 
All analyses revealed no statistically significant differences between variants.
```

### 2. Descriptive Statistics Section
```latex
Descriptive statistics are presented in Table~\ref{tab:descriptive-statistics}. 
The baseline model achieved M = 0.574 (SD = 0.260), 
while DPO-Synthetic and DPO-Hybrid variants showed similar performance...
```

### 3. Statistical Comparisons Section  
```latex
Pairwise statistical comparisons (Table~\ref{tab:statistical-comparisons}) revealed no significant 
differences between any model variants. The omnibus ANOVA was non-significant, 
F(2, 432) = 0.199, p = 0.820, η² = 0.001...
```

### 4. Effect Size Analysis
```latex
Effect size analysis (Figure~\ref{fig:effect-size-forest}) confirmed negligible differences. 
All Cohen's d values were below 0.2, indicating negligible practical significance...
```

### 5. Methodology Validation
```latex
Methodology validation (Table~\ref{tab:methodology-validation}) revealed complete failure 
of theoretical predictions. All predicted effect sizes substantially overestimated actual effects...
```

## Key Interpretive Phrases

### For Statistical Equivalence
- "No statistically significant differences"
- "Effect sizes were negligible (|d| < 0.2)"
- "Substantial overlap in confidence intervals"
- "Performance equivalence across variants"

### For Methodology Validation Failure
- "Methodology predictions failed validation"
- "Large discrepancies between predicted and observed effects"
- "Theoretical framework not supported by empirical data"
- "Optimization approaches proved ineffective"

## Critical Numbers to Double-Check
- Sample size: 145
- ANOVA F-statistic: 0.199
- ANOVA p-value: 0.820
- ANOVA η²: 0.001
- Largest effect size: 0.079

## Available Context Files
- Complete statistical data: `statistical_values_master.json`
- All figure descriptions: `figures_complete_list.json`
- LaTeX tables: `latex_tables_ready.tex`
- Stage summaries: `stage1-4_analysis_summary.md` files

---
*This guide provides all necessary information for comprehensive Results section writing with accurate statistical reporting.*