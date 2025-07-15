# DPO Data Preparation Guide

## Overview

The `prepare_data.sh` script converts multi-topic evaluation results into DPO (Direct Preference Optimization) training format. It processes `complete_results.json` files and creates preference pairs for training.

## Quick Start

### **Recommended: Run from dpo/ directory**
```bash
cd dpo
sbatch slurm/prepare_data.sh
```
This uses the default most recent data folder.

## Available Data Folders

Based on your `output/multi_topic_results/` directory:

| Folder | Date | Description |
|--------|------|-------------|
| `20250715_035107` | July 15, 2025 | **Most recent** (default) |
| `20250714_202647` | July 14, 2025 | Recent results |
| `20250714_043247` | July 14, 2025 | Previous run |
| `20250713_233406` | July 13, 2025 | Earlier results |

## Usage Examples

### **1. Use Default (Most Recent Data)**
```bash
# From dpo/ directory
cd dpo
sbatch slurm/prepare_data.sh
```

### **2. Specify Different Data Folder**
```bash
# Use specific date folder
cd dpo
sbatch slurm/prepare_data.sh ../output/multi_topic_results/20250714_043247

# Use older data
cd dpo  
sbatch slurm/prepare_data.sh ../output/multi_topic_results/20250713_233406
```

### **3. Run from SLURM Directory**
```bash
# From dpo/slurm/ directory
cd dpo/slurm
sbatch prepare_data.sh ../../output/multi_topic_results/20250715_035107
```

## Path Structure

### **Input Data Location:**
```
dissertation/
└── output/
    └── multi_topic_results/
        └── YYYYMMDD_HHMMSS/
            ├── complete_results.json  # Main data file
            └── topic_summary.csv      # Summary stats
```

### **Output Location:**
```
dissertation/
└── dpo/
    └── outputs/
        └── datasets/
            └── dpo_data_YYYYMMDD_HHMMSS.jsonl  # Generated training data
```

## What the Script Does

1. **Reads** `complete_results.json` from specified input folder
2. **Extracts** email evaluation data with rankings
3. **Creates** preference pairs (chosen vs rejected emails)
4. **Outputs** JSONL format suitable for DPO training
5. **Timestamps** output file automatically

## Data Format

### **Input Format (complete_results.json):**
```json
{
  "topic_id": "T0001",
  "emails": [
    {
      "model": "tinyllama",
      "content": "...",
      "rank": 1,
      "final_score": 8.5
    }
  ]
}
```

### **Output Format (dpo_data_TIMESTAMP.jsonl):**
```json
{"prompt": "Generate email for: Children's Hospital...", "chosen": "Dear supporters...", "rejected": "Hi there..."}
{"prompt": "Generate email for: Food Bank...", "chosen": "Dear friends...", "rejected": "Hello..."}
```

## Resource Requirements

- **Memory**: 16GB (handles large evaluation datasets)
- **Time**: ~1 hour (depends on data size)
- **CPUs**: 4 CPUs for data processing
- **Storage**: ~100-500MB output (depends on dataset size)

## Monitoring Progress

### **Check Job Status:**
```bash
squeue -u $USER
```

### **View Log Files:**
```bash
# Check output log
tail -f outputs/logs/data_prep_JOBID.out

# Check error log  
tail -f outputs/logs/data_prep_JOBID.err
```

### **Verify Output:**
```bash
# Check if output file was created
ls -la outputs/datasets/

# Preview first few lines
head outputs/datasets/dpo_data_*.jsonl

# Count training examples
wc -l outputs/datasets/dpo_data_*.jsonl
```

## Troubleshooting

### **Common Issues:**

#### **1. File Not Found Error**
```bash
# Check if input folder exists
ls ../output/multi_topic_results/20250715_035107/

# Verify complete_results.json exists
ls ../output/multi_topic_results/20250715_035107/complete_results.json
```

#### **2. Permission Errors**
```bash
# Ensure outputs directory exists and is writable
mkdir -p outputs/datasets outputs/logs
chmod 755 outputs/datasets outputs/logs
```

#### **3. Environment Issues**
```bash
# Verify conda environment
conda env list | grep dis-venv3

# Check if required packages are installed
pip list | grep -E "(pandas|json|yaml)"
```

### **Expected Output Size:**
- **Small dataset** (~10 topics): 100-200 training examples
- **Medium dataset** (~50 topics): 500-1000 training examples  
- **Large dataset** (~100+ topics): 1000+ training examples

## Next Steps

After successful data preparation:

1. **Verify output quality:**
   ```bash
   head -5 outputs/datasets/dpo_data_*.jsonl
   ```

2. **Start DPO training:**
   ```bash
   sbatch slurm/train_single.sh tinyllama outputs/datasets/dpo_data_*.jsonl
   ```

3. **Train multiple models:**
   ```bash
   sbatch slurm/train_all_sequential.sh outputs/datasets/dpo_data_*.jsonl
   ```

## File Organization

Keep your data preparation organized:

```
dpo/
├── outputs/
│   └── datasets/
│       ├── dpo_data_20250715_120000.jsonl  # Today's data
│       ├── dpo_data_20250714_180000.jsonl  # Yesterday's data
│       └── ...
└── slurm/
    └── prepare_data.sh
```

**Tip**: Use descriptive names or move older datasets to avoid confusion.

---

**Quick Reference:**
- **Default command**: `sbatch slurm/prepare_data.sh`
- **Latest data**: `../output/multi_topic_results/20250715_035107`
- **Output location**: `outputs/datasets/dpo_data_TIMESTAMP.jsonl`