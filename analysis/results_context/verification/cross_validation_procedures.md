# Cross-Validation Procedures

## Purpose
Ensure consistency and accuracy of data across all stages of the Results section implementation.

## Cross-Validation Types

### 1. Statistical Values Cross-Validation

#### Procedure
1. **Source Verification**: Compare statistical values against original data files
2. **Registry Consistency**: Check that all registries contain identical values for the same metrics
3. **Methodology Alignment**: Verify results align with methodology predictions
4. **Mathematical Consistency**: Ensure derived values (e.g., confidence intervals) are correctly calculated

#### Key Checks
- Effect sizes match between Stage 1 analysis and Stage 4 tables
- Descriptive statistics consistent across all summary files
- ANOVA results identical in analysis output and visualization metadata
- Model-specific values sum/average correctly to overall results

### 2. Figure-Data Cross-Validation

#### Procedure
1. **Visual-Numerical Match**: Confirm figures display the same values as in registries
2. **Metadata Accuracy**: Verify figure metadata reflects actual figure content
3. **File Existence**: Ensure all referenced figures exist at specified paths
4. **Resolution Quality**: Confirm figures meet publication standards

#### Key Checks
- Box plot medians match descriptive statistics
- Forest plot effect sizes match statistical analysis results
- Bar chart heights correspond to tabulated means
- Figure legends accurately describe the data shown

### 3. Table-Analysis Cross-Validation

#### Procedure
1. **Data Source Tracing**: Verify each table value traces back to original analysis
2. **Statistical Notation**: Ensure consistent notation across all tables
3. **Precision Consistency**: Check decimal places and rounding are consistent
4. **LaTeX Formatting**: Verify tables compile correctly and display properly

#### Key Checks
- Table 1 descriptive statistics match Stage 1 analysis
- Table 2 effect sizes identical to Stage 1 calculations
- Table 3 model-specific data consistent with Stage 3 analysis
- Table 5 validation status reflects actual results

### 4. Temporal Consistency Cross-Validation

#### Procedure
1. **Timestamp Verification**: Ensure analysis reflects most recent data
2. **File Version Control**: Confirm using correct versions of input files
3. **Sequential Integrity**: Verify each stage builds on previous stage outputs
4. **Update Propagation**: Ensure changes propagate through all dependent files

#### Key Checks
- Input data files are the specified versions from the dates mentioned
- Registry updates reflect the most recent analysis runs
- Summary files contain information from the correct stage executions
- Master files incorporate all stage updates

## Implementation Protocol

### Before Each Stage
1. Run pre-stage cross-validation checks
2. Verify all input dependencies are current
3. Clear any cached or outdated intermediate files
4. Update master context tracker with verification status

### During Stage Execution
1. Implement real-time consistency checks where possible
2. Log any data inconsistencies immediately
3. Halt execution if critical inconsistencies detected
4. Document all intermediate results for later verification

### After Each Stage
1. Run comprehensive cross-validation suite
2. Compare all outputs against expected patterns
3. Verify consistency with previous stages
4. Update all tracking and registry files

### Cross-Stage Integration Points

#### Stage 1 → Stage 2
- Statistical values used in figure generation must match Stage 1 outputs
- Figure metadata must reference correct statistical results
- Visualization parameters must be consistent with analysis results

#### Stage 1 → Stage 3
- Detailed analysis must use same raw data as Stage 1
- Model-specific results must aggregate to Stage 1 overall results
- Category analysis must be consistent with overall statistical patterns

#### Stage 1-3 → Stage 4
- All table values must trace to specific analysis results
- Statistical notation must be consistent across tables
- Methodology validation must reflect actual analysis outcomes

#### Stage 1-4 → Stage 5
- Consolidated results must include all previous stage outputs
- Master statistical file must contain comprehensive value set
- Writing guide must reference all created figures and tables

#### Stage 5 → Stage 6
- Results writing must use only consolidated context files
- All statistical values must match master registry
- Figure and table references must be accurate and complete

## Automated Cross-Validation Checks

### File Existence Verification
```bash
# Check all expected files exist
python verification/check_file_existence.py
```

### Statistical Value Consistency
```bash
# Verify statistical values across all files
python verification/check_statistical_consistency.py
```

### Figure-Data Alignment
```bash
# Compare figure metadata with analysis results
python verification/check_figure_data_alignment.py
```

## Error Resolution Protocol

### When Cross-Validation Fails
1. **Immediate Stop**: Halt current stage execution
2. **Issue Documentation**: Record specific inconsistency details
3. **Root Cause Analysis**: Identify source of inconsistency
4. **Targeted Re-execution**: Re-run only affected components
5. **Full Re-validation**: Verify fix resolves all related issues
6. **Update Tracking**: Record resolution in master context tracker

### Common Issues and Solutions

#### Statistical Value Mismatches
- **Cause**: Different analysis runs, rounding differences, or data source changes
- **Solution**: Re-run analysis with identical parameters, ensure consistent precision

#### Figure-Data Inconsistencies
- **Cause**: Figure generated from outdated data or wrong data source
- **Solution**: Regenerate figures using current registry data

#### Missing Files
- **Cause**: Stage execution failed partially or files moved/deleted
- **Solution**: Re-execute failed stage components, verify file paths

#### Formatting Inconsistencies
- **Cause**: Different formatting applied to same underlying data
- **Solution**: Standardize formatting rules, apply consistently across all outputs

## Quality Assurance

### Cross-Validation Success Criteria
- **100% File Existence**: All expected files present and accessible
- **Statistical Consistency**: All statistical values identical across references
- **Figure Accuracy**: All figures reflect correct underlying data
- **Table Precision**: All tables contain correct values with consistent formatting
- **Methodology Alignment**: All results consistent with methodology predictions

### Final Cross-Validation Report
Before proceeding to Results writing (Stage 6), generate comprehensive cross-validation report:
- Summary of all cross-validation checks performed
- Status of each consistency verification
- Resolution of any identified issues
- Confidence level in data integrity
- Approval for proceeding to Results writing