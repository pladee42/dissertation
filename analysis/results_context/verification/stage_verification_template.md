# Stage Verification Template

## Stage [X] Verification Checklist

### Pre-Stage Requirements
- [ ] Previous stage completed successfully
- [ ] All required input files exist and are accessible
- [ ] Context tracking files updated from previous stage

### Stage Execution Verification
- [ ] All planned tasks completed
- [ ] Expected output files generated
- [ ] No critical errors encountered during execution
- [ ] Statistical analyses (if applicable) completed without warnings

### Output File Verification
- [ ] All expected output files exist at correct locations
- [ ] File sizes are reasonable (not empty or corrupted)
- [ ] File formats are correct (JSON, PNG, MD, TEX as expected)
- [ ] File permissions allow read access

### Data Quality Verification
- [ ] Statistical values are within reasonable ranges
- [ ] No null or undefined values where data should exist
- [ ] Figures display correctly and contain expected elements
- [ ] Tables contain proper statistical notation and formatting

### Context Preservation Verification
- [ ] Stage output file created in `/analysis/results_context/stage_outputs/`
- [ ] Stage summary file created in `/analysis/results_context/summaries/`
- [ ] Registry files updated with new data
- [ ] Master context tracker updated with stage status

### Cross-Validation Checks
- [ ] Statistical values consistent with methodology predictions
- [ ] Figure metadata matches actual figure content
- [ ] No conflicts between different output files
- [ ] Data sources properly documented

### Stage-Specific Checks

#### Stage 1: Statistical Analysis
- [ ] T-test results for all three pairwise comparisons
- [ ] Effect sizes calculated with confidence intervals
- [ ] ANOVA results with F-statistic and η²
- [ ] All statistical values stored in registry

#### Stage 2: Visualization Generation
- [ ] All figures saved to `/report/figures/` directory
- [ ] Figures are high-resolution and publication-ready
- [ ] Figure metadata registry updated with key values
- [ ] Validation subdirectory created and populated

#### Stage 3: Detailed Analysis
- [ ] Model-specific analysis completed for all 7 models
- [ ] Category analysis completed for all 4 categories
- [ ] Custom visualizations generated and saved
- [ ] Detailed analysis registry updated

#### Stage 4: Tables Creation
- [ ] All 5 LaTeX tables properly formatted
- [ ] Statistical notation correct and consistent
- [ ] Table data matches statistical analysis results
- [ ] LaTeX tables ready file created

#### Stage 5: Context Consolidation
- [ ] All stage outputs consolidated successfully
- [ ] Master statistical values file created
- [ ] Writing guide contains all necessary information
- [ ] Comprehensive verification completed

#### Stage 6: Results Writing
- [ ] Complete Results section written
- [ ] All figures and tables properly referenced
- [ ] Statistical values match registry data
- [ ] No context information missing

### Final Verification
- [ ] Stage marked as completed in master context tracker
- [ ] Verification timestamp recorded
- [ ] Next stage can proceed with current outputs
- [ ] All quality checks passed

### Verification Status
- **Stage**: [X]
- **Verifier**: [Name/System]
- **Date**: [YYYY-MM-DD]
- **Time**: [HH:MM]
- **Status**: [PASS/FAIL/PARTIAL]
- **Notes**: [Any additional notes or concerns]

### Issue Resolution
If any verification checks fail:
1. Document the specific issue
2. Determine root cause
3. Re-execute failed components
4. Re-run verification
5. Update master context tracker with resolution