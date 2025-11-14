# saRNA Aligner - Statistical Analysis Pipeline for Small Activating RNA Sequences

A comprehensive Python toolkit for analyzing small activating RNA (saRNA) sequences with statistical analysis, machine learning classification, and publication-ready visualizations.

## 🚀 Quick Start

```python
from saRNA_aligner_20251009 import saRNAAligner, run_statistical_analysis

# Initialize and run analysis
aligner = saRNAAligner(input_file_path="designs.xlsx")
sequences, labels = aligner.load_and_process_data()
results = run_statistical_analysis(aligner.sequences_df)
```

## 📋 Features

- **Statistical Analysis**: Fisher's exact tests, bootstrap confidence intervals, permutation testing
- **Machine Learning**: Random Forest feature importance, K-means clustering, PCA visualization  
- **Motif Discovery**: K-mer enrichment analysis, positional motif patterns
- **Professional Visualizations**: Publication-ready figures (300 DPI, PNG/PDF)
- **Automated Reporting**: Statistical summaries with recommendations

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/saRNA-aligner.git
cd saRNA-aligner

# Install dependencies
pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl
```

## 📊 Input Format

Excel file with the following structure:

| #saRNA_ID | Gene_Target | #saRNA_2 | Gene_Target_positive_control |
|-----------|-------------|----------|------------------------------|
| 001 | AGCTGATCGATCGATCGAT | 002 | GCTAGCTAGCTAGCTAGC |
| 003 | CGTAGCTAGCTAGCTAGCT | 004 | ATCGATCGATCGATCGATC |

- Columns starting with `#`: saRNA identifiers
- Gene target columns: Start with target gene name
- Positive controls: Include "positive control" in column name
- Sequences: Must be exactly 19 nucleotides (A, C, G, U)

## 💻 Usage Examples

### Basic Analysis
```python
from saRNA_aligner_20251009 import saRNAAligner

# Load and analyze sequences
aligner = saRNAAligner(input_file_path="designs.xlsx")
sequences, labels = aligner.load_and_process_data()

print(f"Loaded {len(sequences)} sequences")
print(f"Positive controls: {sum(1 for l in labels if l == 'POS_CTRL')}")
print(f"Test sequences: {sum(1 for l in labels if l == 'NON_CTRL')}")
```

### Statistical Analysis
```python
# Run complete statistical pipeline
results = run_statistical_analysis(aligner.sequences_df)

# Access results
fisher_results = results['fisher_results']
bootstrap_results = results['bootstrap_results']
motif_results = results['motif_results']

# Find significant positions
sig_positions = fisher_results[fisher_results['A_significant'] | 
                               fisher_results['C_significant'] |
                               fisher_results['G_significant'] |
                               fisher_results['U_significant']]
print(f"Significant positions: {len(sig_positions)}")
```

### Machine Learning Analysis
```python
# Encode sequences for ML
X = aligner.encode_sequences(sequences)
y = [1 if label == 'POS_CTRL' else 0 for label in labels]

# Feature importance analysis
aligner.analyze_feature_importance(X, y)

# Clustering
cluster_results = aligner.perform_clustering(X)
print(f"Optimal clusters: {cluster_results['optimal_k']}")
```

### Motif Analysis
```python
# Analyze k-mer patterns
for k in [2, 3, 4]:
    positional_counts = aligner.count_positional_motifs(sequences, n=k)
    # Process results...

# Analyze stability patterns (Strong=G,C; Weak=A,U)
stability_counts = aligner.count_positional_motifs(
    sequences, n=3, use_stability=True
)
```

## 📁 Output Structure

```
saRNA_Aligner_Analysis_YYYYMMDD_HHMMSS/
├── sequence_analysis_results.csv
├── fisher_exact_test_results.csv
├── bootstrap_confidence_intervals.csv
├── motif_enrichment_results.csv
├── nucleotide_frequency_heatmap.png/pdf
├── confidence_intervals.png/pdf
├── effect_sizes_heatmap.png/pdf
├── feature_importance.png/pdf
└── statistical_summary.txt
```

## 📊 Analysis Pipeline

1. **Data Loading**: Excel file → DataFrame with sequence validation
2. **Statistical Testing**: Position-wise nucleotide frequency analysis
3. **Effect Size Estimation**: Bootstrap confidence intervals (10,000 iterations)
4. **Motif Discovery**: K-mer enrichment with significance testing
5. **Machine Learning**: Random Forest classification and clustering
6. **Visualization**: High-quality plots for publication

## 🔬 Statistical Methods

### Fisher's Exact Test
- Position-wise nucleotide frequency comparison
- Bonferroni correction for multiple testing
- P-value < 0.05 for significance

### Bootstrap Analysis
- 10,000 bootstrap iterations
- 95% confidence intervals
- Effect size calculation

### Permutation Testing
- Non-parametric significance testing
- 1,000 permutations default
- Global pattern detection

## 🎨 Visualization Options

```python
# Customize visualization settings
aligner.fig_dpi = 600  # Higher resolution
aligner.colors = {
    'nucleotides': {
        'A': '#FF6B6B',  # Custom colors
        'C': '#4ECDC4',
        'G': '#45B7D1',
        'U': '#FFA07A'
    }
}
```

## 📦 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=1.0.0
openpyxl>=3.0.9
```

