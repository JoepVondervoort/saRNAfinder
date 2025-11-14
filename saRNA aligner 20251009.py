# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
from itertools import product
from scipy.stats import fisher_exact, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('default')
sns.set_palette("husl")

class saRNAAligner:
    """Professional saRNA sequence analysis tool."""
    
    def __init__(self, input_file_path=None, working_dir=None):
        """Initialize the saRNA Aligner."""
        # Fix for Jupyter notebook compatibility
        if working_dir is None:
            try:
                # Try to get the script directory (works in .py files)
                self.script_dir = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                # Fallback for Jupyter notebooks
                self.script_dir = os.getcwd()
                print(f"Running in Jupyter environment. Using current working directory: {self.script_dir}")
        else:
            self.script_dir = working_dir
            
        self.input_file = input_file_path or os.path.join(self.script_dir, "designs.xlsx")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.script_dir, f'saRNA_Aligner_Analysis_{self.timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Professional figure settings
        self.fig_dpi = 300
        self.fig_format = 'both'  # Save both PNG and PDF
        
        # Color schemes
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#F39C12',
            'accent': '#E74C3C',
            'neutral': '#5D6D7E',
            'success': '#27AE60',
            'nucleotides': {'A': '#FF6B6B', 'C': '#4ECDC4', 'G': '#45B7D1', 'U': '#FFA07A'}
        }
        
    def encode_sequences(self, sequences):
        """Encode RNA sequences using one-hot encoding."""
        base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        encoded = np.zeros((len(sequences), 19 * 4))
        
        for i, seq in enumerate(sequences):
            for j, nucleotide in enumerate(seq):
                if nucleotide in base_map:
                    encoded[i, j * 4 + base_map[nucleotide]] = 1
                    
        return encoded
    
    def map_to_stability(self, sequence):
        """Map nucleotides to stability categories (Strong: G,C; Weak: A,U)."""
        return ''.join(['S' if nuc in 'GC' else 'W' for nuc in sequence])
    
    def count_positional_motifs(self, sequences, n, use_stability=False):
        """Count n-mer motifs at each position across sequences."""
        if not sequences:
            return []
            
        seq_length = len(sequences[0])
        positional_counts = [Counter() for _ in range(seq_length - n + 1)]
        
        for seq in sequences:
            processed_seq = self.map_to_stability(seq) if use_stability else seq
            for i in range(len(processed_seq) - n + 1):
                motif = processed_seq[i:i + n]
                positional_counts[i][motif] += 1
                
        return positional_counts
    
    def load_and_process_data(self):
        """Load and process the Excel data file."""
        print(f"Loading data from: {self.input_file}")
        
        try:
            df = pd.read_excel(self.input_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        except Exception as e:
            raise Exception(f"Error loading Excel file: {e}")
        
        sequences, labels, seq_labels, target_genes, sarna_numbers = [], [], [], [], []
        
        for idx, column in enumerate(df.columns):
            column_str = str(column).strip()
            
            # Skip columns starting with #
            if column_str.startswith("#"):
                continue
            
            is_positive_control = "positive control" in column_str.lower()
            target_gene = column_str.split(" ")[0].split("_")[0].strip()
            
            # Find corresponding saRNA number column
            sarna_number_col = None
            if idx > 0 and str(df.columns[idx - 1]).strip().startswith("#"):
                sarna_number_col = df.columns[idx - 1]
            
            # Process sequences in this column
            for row_idx, value in enumerate(df[column]):
                if pd.isnull(value):
                    continue
                    
                sequence = str(value).strip().upper()
                if len(sequence) == 19:  # Valid saRNA sequence length
                    label = "POS_CTRL" if is_positive_control else "NON_CTRL"
                    sequences.append(sequence)
                    labels.append(label)
                    seq_labels.append(f"{column}_{row_idx + 1}")
                    target_genes.append(target_gene)
                    
                    # Extract saRNA number
                    if sarna_number_col is not None:
                        sarna_num = df.loc[row_idx, sarna_number_col]
                        sarna_num = str(sarna_num).strip() if not pd.isnull(sarna_num) else "Unknown"
                    else:
                        sarna_num = "Unknown"
                    sarna_numbers.append(sarna_num)
        
        # Create comprehensive dataframe
        self.sequences_df = pd.DataFrame({
            'Sequence': sequences,
            'Type': labels,
            'Label': seq_labels,
            'Target_Gene': target_genes,
            'saRNA_Number': sarna_numbers
        })
        
        print(f"Loaded {len(sequences)} sequences:")
        print(f"  - Positive controls: {sum(1 for l in labels if l == 'POS_CTRL')}")
        print(f"  - Test sequences: {sum(1 for l in labels if l == 'NON_CTRL')}")
        
        return sequences, labels
    
    def save_figure(self, fig, filename, tight_layout=True):
        """Save figure in high quality with professional formatting."""
        if tight_layout:
            fig.tight_layout()
        
        png_path = os.path.join(self.output_dir, f"{filename}.png")
        pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
        
        fig.savefig(png_path, dpi=self.fig_dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        fig.savefig(pdf_path, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"Saved: {filename}")
    
    def analyze_feature_importance(self, X, y):
        """Analyze positional importance using Random Forest."""
        print("Analyzing positional feature importance...")
        
        # Train Random Forest classifier
        rf = RandomForestClassifier(n_estimators=500, random_state=42, 
                                  max_depth=10, min_samples_split=5)
        rf.fit(X, y)
        
        # Calculate positional importance
        feature_importances = rf.feature_importances_.reshape(19, 4)
        positional_importance = feature_importances.sum(axis=1)
        
        # Get nucleotide recommendations
        pos_sequences = self.sequences_df[self.sequences_df['Type'] == 'POS_CTRL']['Sequence'].tolist()
        non_sequences = self.sequences_df[self.sequences_df['Type'] == 'NON_CTRL']['Sequence'].tolist()
        
        nucleotide_analysis = self._analyze_nucleotide_preferences(pos_sequences, non_sequences)
        
        # Create professional visualization
        self._plot_feature_importance(positional_importance, nucleotide_analysis)
        
        return positional_importance, nucleotide_analysis
    
    def _analyze_nucleotide_preferences(self, pos_sequences, non_sequences):
        """Analyze nucleotide preferences at each position."""
        base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        inv_base_map = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
        
        # Count nucleotides in positive controls
        pos_counts = np.zeros((19, 4))
        for seq in pos_sequences:
            for i, nt in enumerate(seq):
                if nt in base_map:
                    pos_counts[i, base_map[nt]] += 1
        
        # Count nucleotides in non-controls
        non_counts = np.zeros((19, 4))
        for seq in non_sequences:
            for i, nt in enumerate(seq):
                if nt in base_map:
                    non_counts[i, base_map[nt]] += 1
        
        # Calculate frequencies
        pos_freqs = pos_counts / pos_counts.sum(axis=1, keepdims=True)
        non_freqs = non_counts / non_counts.sum(axis=1, keepdims=True)
        
        # Get recommendations
        recommended_nucleotides = [inv_base_map[np.argmax(row)] for row in pos_freqs]
        pos_freq_values = pos_freqs.max(axis=1)
        non_freq_values = [non_freqs[i, base_map[nt]] for i, nt in enumerate(recommended_nucleotides)]
        
        return {
            'recommended_nucs': recommended_nucleotides,
            'pos_freqs': pos_freq_values,
            'non_freqs': non_freq_values,
            'pos_counts_matrix': pos_counts,
            'non_counts_matrix': non_counts
        }
    
    def _plot_feature_importance(self, importance_scores, nucleotide_data):
        """Create professional feature importance plot."""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        positions = np.arange(1, 20)
        
        # Create gradient color scheme
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, 19))
        
        bars = ax.bar(positions, importance_scores, color=colors, 
                     edgecolor='white', linewidth=0.8, alpha=0.8)
        
        # Add nucleotide annotations
        for i, (score, nuc, pos_freq, non_freq) in enumerate(zip(
            importance_scores, 
            nucleotide_data['recommended_nucs'],
            nucleotide_data['pos_freqs'],
            nucleotide_data['non_freqs']
        )):
            # Nucleotide label
            ax.text(i + 1, score + 0.01, nuc, ha='center', va='bottom', 
                   fontsize=14, fontweight='bold', 
                   color=self.colors['nucleotides'].get(nuc, 'black'))
            
            # Frequency information
            ax.text(i + 1, score + 0.005, f"Ctrl: {pos_freq:.2f}\nTest: {non_freq:.2f}", 
                   ha='center', va='bottom', fontsize=9, 
                   color='dimgray', alpha=0.8)
        
        # Styling
        ax.set_xlabel('Nucleotide Position', fontsize=14, fontweight='bold')
        ax.set_ylabel('Feature Importance Score', fontsize=14, fontweight='bold')
        ax.set_title('saRNA Positional Importance Analysis\nRecommended Nucleotides for Optimal Activity', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(positions)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Add subtle background
        ax.set_facecolor('#FAFAFA')
        
        self.save_figure(fig, 'positional_importance_analysis')
    
    def create_motif_heatmaps(self, pos_sequences, non_sequences):
        """Create comprehensive motif analysis heatmaps."""
        print("Generating motif analysis heatmaps...")
        
        for use_stability in [False, True]:
            motif_type = 'Stability' if use_stability else 'Nucleotide'
            alphabet = 'SW' if use_stability else 'ACGU'
            
            for n in range(1, 5):
                self._create_single_heatmap(pos_sequences, non_sequences, n, 
                                          use_stability, motif_type, alphabet)
    
    def _create_single_heatmap(self, pos_sequences, non_sequences, n, 
                              use_stability, motif_type, alphabet):
        """Create a single motif heatmap."""
        pos_counts = self.count_positional_motifs(pos_sequences, n, use_stability)
        non_counts = self.count_positional_motifs(non_sequences, n, use_stability)
        
        motifs = [''.join(combo) for combo in product(alphabet, repeat=n)]
        heatmap_data = []
        
        for position in range(len(pos_counts)):
            row = []
            for motif in motifs:
                pos_total = sum(pos_counts[position].values())
                non_total = sum(non_counts[position].values())
                
                pos_freq = pos_counts[position][motif] / pos_total if pos_total > 0 else 0
                non_freq = non_counts[position][motif] / non_total if non_total > 0 else 0
                
                difference_percent = (pos_freq - non_freq) * 100
                row.append(difference_percent)
            
            heatmap_data.append(row)
        
        # Create heatmap
        heatmap_df = pd.DataFrame(
            heatmap_data, 
            columns=motifs, 
            index=[f'Position {i + 1}' for i in range(len(heatmap_data))]
        )
        
        # Determine figure size based on motif count
        fig_width = max(12, len(motifs) * 0.8)
        fig_height = max(8, len(heatmap_data) * 0.5)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create heatmap with professional styling
        sns.heatmap(heatmap_df, 
                   annot=n <= 2,  # Only annotate for smaller motifs
                   fmt='.1f',
                   cmap='RdBu_r',
                   center=0,
                   cbar_kws={'label': 'Frequency Difference (%)'},
                   linewidths=0.5,
                   ax=ax)
        
        ax.set_title(f'{motif_type} {n}-mer Motif Analysis\nPositional Frequency Differences (Control vs Test)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Motifs', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sequence Position', fontsize=12, fontweight='bold')
        
        filename = f"{motif_type.lower()}_{n}mer_motif_heatmap"
        self.save_figure(fig, filename)
    
    def create_pca_analysis(self, X, y, labels):
        """Create professional PCA analysis."""
        print("Performing PCA analysis...")
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        
        # Create professional PCA plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Main PCA plot
        unique_labels = list(set(labels))
        colors = [self.colors['primary'], self.colors['secondary']]
        markers = ['o', '^']
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax1.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       c=colors[i], marker=markers[i], s=80, alpha=0.7, 
                       label=f'{label} (n={sum(mask)})', edgecolors='white', linewidth=0.5)
        
        ax1.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                      fontsize=12, fontweight='bold')
        ax1.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                      fontsize=12, fontweight='bold')
        ax1.set_title('PCA Analysis of saRNA Sequences', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#FAFAFA')
        
        # Variance explained plot
        all_components = PCA().fit(X)
        cumvar = np.cumsum(all_components.explained_variance_ratio_)
        
        ax2.plot(range(1, min(21, len(cumvar) + 1)), cumvar[:20], 
                marker='o', linewidth=2, markersize=6, color=self.colors['accent'])
        ax2.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7, label='95% Variance')
        ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
        ax2.set_title('Explained Variance by Components', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_facecolor('#FAFAFA')
        
        self.save_figure(fig, 'pca_comprehensive_analysis')
        
        return pca_result, pca.explained_variance_ratio_
    
    def create_clustering_analysis(self, X, y):
        """Perform clustering analysis with confusion matrix."""
        print("Performing clustering analysis...")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Create confusion matrix
        cm = confusion_matrix(y, cluster_labels)
        
        # Professional confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Cluster 0', 'Cluster 1'],
                   yticklabels=['Non-Control', 'Positive Control'],
                   cbar_kws={'label': 'Count'},
                   linewidths=1, ax=ax)
        
        ax.set_title('Clustering Performance\nConfusion Matrix', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted Cluster', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        
        # Calculate accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        ax.text(0.5, -0.15, f'Clustering Accuracy: {accuracy:.1%}', 
               transform=ax.transAxes, ha='center', fontsize=12, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        self.save_figure(fig, 'clustering_confusion_matrix')
        
        return cluster_labels, cm
    
    def generate_summary_report(self, importance_scores, nucleotide_data, pca_variance, cm):
        """Generate comprehensive Excel report."""
        print("Generating comprehensive Excel report...")
        
        output_excel = os.path.join(self.output_dir, f"saRNA_Aligner_Report_{self.timestamp}.xlsx")
        
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            # Sequence data
            pos_ctrl_df = self.sequences_df[self.sequences_df['Type'] == 'POS_CTRL']
            test_seq_df = self.sequences_df[self.sequences_df['Type'] == 'NON_CTRL']
            
            pos_ctrl_df.to_excel(writer, sheet_name='Positive_Controls', index=False)
            test_seq_df.to_excel(writer, sheet_name='Test_Sequences', index=False)
            
            # Feature importance analysis
            importance_df = pd.DataFrame({
                'Position': range(1, 20),
                'Importance_Score': importance_scores,
                'Recommended_Nucleotide': nucleotide_data['recommended_nucs'],
                'Control_Frequency': nucleotide_data['pos_freqs'],
                'Test_Frequency': nucleotide_data['non_freqs']
            })
            importance_df.to_excel(writer, sheet_name='Positional_Analysis', index=False)
            
            # PCA results
            pca_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(pca_variance))],
                'Explained_Variance_Ratio': pca_variance
            })
            pca_df.to_excel(writer, sheet_name='PCA_Analysis', index=False)
            
            # Confusion matrix
            cm_df = pd.DataFrame(cm, 
                               index=['Non_Control', 'Positive_Control'],
                               columns=['Cluster_0', 'Cluster_1'])
            cm_df.to_excel(writer, sheet_name='Clustering_Results')
            
            # Summary statistics
            summary_df = pd.DataFrame({
                'Metric': ['Total_Sequences', 'Positive_Controls', 'Test_Sequences', 
                          'Clustering_Accuracy', 'Top_2_PC_Variance'],
                'Value': [len(self.sequences_df), len(pos_ctrl_df), len(test_seq_df),
                         f"{np.trace(cm) / np.sum(cm):.1%}",
                         f"{sum(pca_variance[:2]):.1%}"]
            })
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        print(f"Excel report saved: {output_excel}")
    
    def run_complete_analysis(self):
        """Execute the complete saRNA analysis pipeline."""
        print("="*60)
        print("saRNA Aligner - Professional Sequence Analysis")
        print("="*60)
        print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        try:
            # Load and process data
            sequences, labels = self.load_and_process_data()
            
            # Encode sequences for machine learning
            X = self.encode_sequences(sequences)
            y = np.array([1 if lab == "POS_CTRL" else 0 for lab in labels])
            
            # Feature importance analysis
            importance_scores, nucleotide_data = self.analyze_feature_importance(X, y)
            
            # Create motif heatmaps
            pos_sequences = self.sequences_df[self.sequences_df['Type'] == 'POS_CTRL']['Sequence'].tolist()
            non_sequences = self.sequences_df[self.sequences_df['Type'] == 'NON_CTRL']['Sequence'].tolist()
            self.create_motif_heatmaps(pos_sequences, non_sequences)
            
            # PCA analysis
            pca_result, pca_variance = self.create_pca_analysis(X, y, labels)
            
            # Clustering analysis
            cluster_labels, cm = self.create_clustering_analysis(X, y)
            
            # Generate comprehensive report
            self.generate_summary_report(importance_scores, nucleotide_data, pca_variance, cm)
            
            print()
            print("="*60)
            print("Analysis completed successfully!")
            print(f"Results saved in: {self.output_dir}")
            print("="*60)
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


def main():
    """Main execution function."""
    try:
        # Initialize and run saRNA Aligner
        aligner = saRNAAligner(input_file_path=r"C:\Users\joep-\Python scripts\Designs.xlsx")
        aligner.run_complete_analysis()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    
    return 0




# %%
# For Jupyter notebook execution
if __name__ == "__main__":
    # You can run the analysis in Jupyter with:
    main()
    
    # Or create an instance directly:
    # aligner = saRNAAligner(input_file_path=r"C:\Users\joep-\Python scripts\designs.xlsx")
    # aligner.run_complete_analysis()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact, chi2_contingency, binomtest
from scipy import stats
from collections import Counter
import warnings
import os
import glob
from datetime import datetime
warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_sequences_from_excel(file_path="designs.xlsx"):
    """Load and process saRNA sequences from Excel file."""
    
    print(f"Loading data from: {file_path}")
    
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print(f"Current directory: {os.getcwd()}")
        print("Files in current directory:")
        for f in os.listdir():
            if f.endswith(('.xlsx', '.xls')):
                print(f"  - {f}")
        return None
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None
    
    sequences, labels, seq_labels, target_genes, sarna_numbers = [], [], [], [], []
    
    for idx, column in enumerate(df.columns):
        column_str = str(column).strip()
        
        # Skip columns starting with #
        if column_str.startswith("#"):
            continue
        
        is_positive_control = "positive control" in column_str.lower()
        target_gene = column_str.split(" ")[0].split("_")[0].strip()
        
        # Find corresponding saRNA number column
        sarna_number_col = None
        if idx > 0 and str(df.columns[idx - 1]).strip().startswith("#"):
            sarna_number_col = df.columns[idx - 1]
        
        # Process sequences in this column
        for row_idx, value in enumerate(df[column]):
            if pd.isnull(value):
                continue
                
            sequence = str(value).strip().upper()
            if len(sequence) == 19:  # Valid saRNA sequence length
                label = "POS_CTRL" if is_positive_control else "NON_CTRL"
                sequences.append(sequence)
                labels.append(label)
                seq_labels.append(f"{column}_{row_idx + 1}")
                target_genes.append(target_gene)
                
                # Extract saRNA number
                if sarna_number_col is not None:
                    sarna_num = df.loc[row_idx, sarna_number_col]
                    sarna_num = str(sarna_num).strip() if not pd.isnull(sarna_num) else "Unknown"
                else:
                    sarna_num = "Unknown"
                sarna_numbers.append(sarna_num)
    
    # Create comprehensive dataframe
    sequences_df = pd.DataFrame({
        'Sequence': sequences,
        'Type': labels,
        'Label': seq_labels,
        'Target_Gene': target_genes,
        'saRNA_Number': sarna_numbers
    })
    
    print(f"Loaded {len(sequences)} sequences:")
    print(f"  - Positive controls: {sum(1 for l in labels if l == 'POS_CTRL')}")
    print(f"  - Test sequences: {sum(1 for l in labels if l == 'NON_CTRL')}")
    print()
    
    return sequences_df

# =============================================================================
# STATISTICAL ANALYSIS CLASS
# =============================================================================

class StatisticalSaRNAAnalysis:
    """Statistically appropriate analysis for small saRNA datasets."""
    
    def __init__(self, sequences_df, output_dir=None):
        self.sequences_df = sequences_df
        self.pos_sequences = sequences_df[sequences_df['Type'] == 'POS_CTRL']['Sequence'].tolist()
        self.test_sequences = sequences_df[sequences_df['Type'] == 'NON_CTRL']['Sequence'].tolist()
        
        # Setup output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            self.output_dir = f'Statistical_saRNA_Analysis_{self.timestamp}'
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # High-quality figure settings
        self.fig_dpi = 300
        
        print(f"Dataset Summary:")
        print(f"Positive Controls: {len(self.pos_sequences)}")
        print(f"Test Sequences: {len(self.test_sequences)}")
        print(f"Total: {len(sequences_df)}")
        print(f"Output directory: {self.output_dir}")
        print()
    
    def save_figure(self, fig, filename, tight_layout=True):
        """Save figure in high quality TIFF and PDF formats."""
        if tight_layout:
            fig.tight_layout()
        
        # Save as high-quality TIFF
        tiff_path = os.path.join(self.output_dir, f"{filename}.tiff")
        fig.savefig(tiff_path, dpi=self.fig_dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='tiff')
        
        # Save as PDF
        pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
        fig.savefig(pdf_path, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', format='pdf')
        
        print(f"Saved: {filename}.tiff and {filename}.pdf")
        
        # Close figure to free memory
        plt.close(fig)
        
        return tiff_path, pdf_path
        
    def fisher_exact_analysis(self):
        """Perform Fisher's exact test for each nucleotide at each position."""
        print("=== Fisher's Exact Test Analysis ===")
        
        nucleotides = ['A', 'C', 'G', 'U']
        results = []
        
        for pos in range(19):  # 19 positions
            print(f"Position {pos + 1}:")
            pos_results = {'position': pos + 1}
            
            for nuc in nucleotides:
                # Count occurrences
                pos_count = sum(1 for seq in self.pos_sequences if seq[pos] == nuc)
                pos_total = len(self.pos_sequences)
                test_count = sum(1 for seq in self.test_sequences if seq[pos] == nuc)
                test_total = len(self.test_sequences)
                
                # Create contingency table
                table = [[pos_count, pos_total - pos_count],
                        [test_count, test_total - test_count]]
                
                # Fisher's exact test
                odds_ratio, p_value = fisher_exact(table)
                
                # Calculate frequencies
                pos_freq = pos_count / pos_total
                test_freq = test_count / test_total
                
                pos_results[f'{nuc}_pos_freq'] = pos_freq
                pos_results[f'{nuc}_test_freq'] = test_freq
                pos_results[f'{nuc}_odds_ratio'] = odds_ratio
                pos_results[f'{nuc}_p_value'] = p_value
                pos_results[f'{nuc}_significant'] = p_value < 0.05
                
                print(f"  {nuc}: Ctrl={pos_freq:.2f}, Test={test_freq:.2f}, "
                      f"OR={odds_ratio:.2f}, p={p_value:.4f}")
            
            results.append(pos_results)
            print()
        
        return pd.DataFrame(results)
    
    def bootstrap_confidence_intervals(self, n_bootstrap=1000):
        """Calculate bootstrap confidence intervals for nucleotide frequencies."""
        print("=== Bootstrap Confidence Intervals ===")
        
        nucleotides = ['A', 'C', 'G', 'U']
        results = []
        
        for pos in range(19):
            for nuc in nucleotides:
                # Bootstrap for positive controls
                pos_freqs = []
                for _ in range(n_bootstrap):
                    sample = np.random.choice(self.pos_sequences, size=len(self.pos_sequences), replace=True)
                    freq = sum(1 for seq in sample if seq[pos] == nuc) / len(sample)
                    pos_freqs.append(freq)
                
                pos_ci_lower = np.percentile(pos_freqs, 2.5)
                pos_ci_upper = np.percentile(pos_freqs, 97.5)
                pos_mean = np.mean(pos_freqs)
                
                # Bootstrap for test sequences
                test_freqs = []
                for _ in range(n_bootstrap):
                    sample = np.random.choice(self.test_sequences, size=len(self.test_sequences), replace=True)
                    freq = sum(1 for seq in sample if seq[pos] == nuc) / len(sample)
                    test_freqs.append(freq)
                
                test_ci_lower = np.percentile(test_freqs, 2.5)
                test_ci_upper = np.percentile(test_freqs, 97.5)
                test_mean = np.mean(test_freqs)
                
                results.append({
                    'position': pos + 1,
                    'nucleotide': nuc,
                    'pos_mean': pos_mean,
                    'pos_ci_lower': pos_ci_lower,
                    'pos_ci_upper': pos_ci_upper,
                    'test_mean': test_mean,
                    'test_ci_lower': test_ci_lower,
                    'test_ci_upper': test_ci_upper,
                    'difference': pos_mean - test_mean
                })
        
        return pd.DataFrame(results)
    
    def permutation_test(self, n_permutations=10000):
        """Perform permutation test for overall sequence differences."""
        print("=== Permutation Test ===")
        
        # Calculate observed difference in G+C content
        def gc_content(sequences):
            return [sum(1 for nuc in seq if nuc in 'GC') / len(seq) for seq in sequences]
        
        pos_gc = np.mean(gc_content(self.pos_sequences))
        test_gc = np.mean(gc_content(self.test_sequences))
        observed_diff = pos_gc - test_gc
        
        print(f"Observed GC content difference: {observed_diff:.4f}")
        
        # Permutation test
        all_sequences = self.pos_sequences + self.test_sequences
        all_labels = ['pos'] * len(self.pos_sequences) + ['test'] * len(self.test_sequences)
        
        permuted_diffs = []
        for _ in range(n_permutations):
            # Shuffle labels
            shuffled_labels = np.random.permutation(all_labels)
            
            # Calculate difference for this permutation
            perm_pos = [all_sequences[i] for i, label in enumerate(shuffled_labels) if label == 'pos']
            perm_test = [all_sequences[i] for i, label in enumerate(shuffled_labels) if label == 'test']
            
            perm_pos_gc = np.mean(gc_content(perm_pos))
            perm_test_gc = np.mean(gc_content(perm_test))
            permuted_diffs.append(perm_pos_gc - perm_test_gc)
        
        # Calculate p-value
        p_value = sum(1 for diff in permuted_diffs if abs(diff) >= abs(observed_diff)) / n_permutations
        
        print(f"Permutation test p-value: {p_value:.4f}")
        
        return {
            'observed_difference': observed_diff,
            'p_value': p_value,
            'permuted_differences': permuted_diffs
        }
    
    def motif_enrichment_analysis(self, k=3):
        """Analyze k-mer motif enrichment using exact binomial tests."""
        print(f"=== {k}-mer Motif Enrichment Analysis ===")
        
        def extract_kmers(sequences, k):
            kmers = []
            for seq in sequences:
                for i in range(len(seq) - k + 1):
                    kmers.append(seq[i:i+k])
            return kmers
        
        pos_kmers = extract_kmers(self.pos_sequences, k)
        test_kmers = extract_kmers(self.test_sequences, k)
        
        pos_counts = Counter(pos_kmers)
        test_counts = Counter(test_kmers)
        
        # Get all unique k-mers
        all_kmers = set(pos_kmers + test_kmers)
        
        results = []
        for kmer in all_kmers:
            pos_count = pos_counts.get(kmer, 0)
            test_count = test_counts.get(kmer, 0)
            
            pos_total = len(pos_kmers)
            test_total = len(test_kmers)
            
            pos_freq = pos_count / pos_total
            test_freq = test_count / test_total
            
            # Binomial test (is positive control frequency significantly different from test?)
            expected_prob = test_freq
            if expected_prob > 0:
                binom_result = binomtest(pos_count, pos_total, expected_prob)
                p_value = binom_result.pvalue
            else:
                p_value = 1.0 if pos_count == 0 else 0.0
            
            results.append({
                'kmer': kmer,
                'pos_count': pos_count,
                'test_count': test_count,
                'pos_freq': pos_freq,
                'test_freq': test_freq,
                'enrichment_ratio': pos_freq / test_freq if test_freq > 0 else float('inf'),
                'p_value': p_value
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('p_value')
        
        # Apply Bonferroni correction
        results_df['p_adjusted'] = results_df['p_value'] * len(results_df)
        results_df['significant'] = results_df['p_adjusted'] < 0.05
        
        print(f"Found {sum(results_df['significant'])} significantly enriched {k}-mers")
        print("Top 10 most significant:")
        print(results_df[['kmer', 'pos_freq', 'test_freq', 'enrichment_ratio', 'p_value', 'p_adjusted']].head(10))
        
        return results_df
    
    def plot_confidence_intervals(self, bootstrap_results):
        """Plot nucleotide frequencies with confidence intervals."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        nucleotides = ['A', 'C', 'G', 'U']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for i, nuc in enumerate(nucleotides):
            ax = axes[i]
            nuc_data = bootstrap_results[bootstrap_results['nucleotide'] == nuc]
            
            positions = nuc_data['position']
            
            # Plot positive controls
            ax.errorbar(positions, nuc_data['pos_mean'], 
                       yerr=[nuc_data['pos_mean'] - nuc_data['pos_ci_lower'],
                             nuc_data['pos_ci_upper'] - nuc_data['pos_mean']],
                       label='Positive Controls', marker='o', capsize=5, 
                       color=colors[i], alpha=0.8, linewidth=2, markersize=6)
            
            # Plot test sequences
            ax.errorbar(positions, nuc_data['test_mean'], 
                       yerr=[nuc_data['test_mean'] - nuc_data['test_ci_lower'],
                             nuc_data['test_ci_upper'] - nuc_data['test_mean']],
                       label='Test Sequences', marker='s', capsize=5, 
                       color='gray', alpha=0.8, linewidth=2, markersize=6)
            
            ax.set_title(f'Nucleotide {nuc} Frequency by Position\n(95% Confidence Intervals)', 
                        fontweight='bold', fontsize=14)
            ax.set_xlabel('Position', fontweight='bold', fontsize=12)
            ax.set_ylabel('Frequency', fontweight='bold', fontsize=12)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, 20))
            ax.set_ylim(0, 1)
        
        # Save the figure
        self.save_figure(fig, 'nucleotide_frequency_confidence_intervals')
        
        return fig
    
    def plot_effect_sizes(self, bootstrap_results):
        """Plot effect sizes (differences) between groups."""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Create heatmap data
        nucleotides = ['A', 'C', 'G', 'U']
        positions = range(1, 20)
        
        heatmap_data = []
        for pos in positions:
            row = []
            for nuc in nucleotides:
                diff = bootstrap_results[
                    (bootstrap_results['position'] == pos) & 
                    (bootstrap_results['nucleotide'] == nuc)
                ]['difference'].iloc[0]
                row.append(diff)
            heatmap_data.append(row)
        
        # Create heatmap
        heatmap_df = pd.DataFrame(heatmap_data, columns=nucleotides, index=positions)
        
        sns.heatmap(heatmap_df, annot=True, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Frequency Difference (Ctrl - Test)'}, 
                   fmt='.3f', ax=ax, linewidths=0.5)
        
        ax.set_title('Effect Sizes: Nucleotide Frequency Differences\n(Positive Controls - Test Sequences)', 
                    fontweight='bold', fontsize=16, pad=20)
        ax.set_xlabel('Nucleotide', fontweight='bold', fontsize=14)
        ax.set_ylabel('Position', fontweight='bold', fontsize=14)
        
        # Save the figure
        self.save_figure(fig, 'effect_sizes_heatmap')
        
        return fig
    
    def generate_statistical_summary(self, fisher_results, bootstrap_results, motif_results):
        """Generate a comprehensive statistical summary."""
        print("\n" + "="*60)
        print("STATISTICAL SUMMARY")
        print("="*60)
        
        # Power analysis warning
        print("⚠️  POWER ANALYSIS WARNING:")
        print(f"With only {len(self.pos_sequences)} positive controls, this study has LOW STATISTICAL POWER.")
        print("Results should be interpreted cautiously and validated with larger samples.\n")
        
        # Significant positions from Fisher's test
        significant_positions = []
        for _, row in fisher_results.iterrows():
            pos = row['position']
            sig_nucs = []
            for nuc in ['A', 'C', 'G', 'U']:
                if row[f'{nuc}_significant']:
                    sig_nucs.append(f"{nuc} (p={row[f'{nuc}_p_value']:.4f})")
            if sig_nucs:
                significant_positions.append(f"Position {pos}: {', '.join(sig_nucs)}")
        
        print("🔍 SIGNIFICANT NUCLEOTIDE DIFFERENCES (p < 0.05):")
        if significant_positions:
            for pos in significant_positions:
                print(f"  • {pos}")
        else:
            print("  • No statistically significant differences found")
        
        print(f"\n📊 EFFECT SIZES:")
        # Find largest effect sizes
        max_effects = []
        for _, row in bootstrap_results.iterrows():
            if abs(row['difference']) > 0.1:  # Meaningful difference threshold
                max_effects.append(f"Position {row['position']} {row['nucleotide']}: "
                                 f"{row['difference']:+.3f}")
        
        if max_effects:
            print("  Large effect sizes (|difference| > 0.1):")
            for effect in sorted(max_effects, key=lambda x: abs(float(x.split(': ')[1])), reverse=True)[:5]:
                print(f"  • {effect}")
        else:
            print("  • No large effect sizes detected")
        
        print(f"\n🧬 MOTIF ANALYSIS:")
        sig_motifs = motif_results[motif_results['significant']]
        if len(sig_motifs) > 0:
            print(f"  • {len(sig_motifs)} significantly enriched 3-mers found")
            print("  Top enriched motifs:")
            for _, row in sig_motifs.head(3).iterrows():
                print(f"    - {row['kmer']}: {row['enrichment_ratio']:.2f}x enriched (p={row['p_value']:.4f})")
        else:
            print("  • No significantly enriched motifs found")
        
        print(f"\n📈 RECOMMENDATIONS:")
        print("  1. Collect more positive control sequences (target: 50-100)")
        print("  2. Focus on positions/motifs with largest effect sizes")
        print("  3. Validate findings with independent dataset")
        print("  4. Consider biological relevance over statistical significance")
        
        return {
            'significant_positions': len(significant_positions),
            'large_effects': len(max_effects),
            'significant_motifs': len(sig_motifs)
        }

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_statistical_analysis(sequences_df, output_dir=None):
    """Run the complete statistical analysis pipeline with high-quality plots."""
    
    analyzer = StatisticalSaRNAAnalysis(sequences_df, output_dir)
    
    print("Starting comprehensive statistical analysis...")
    print("This will generate multiple high-quality plots and reports.\n")
    
    # 1. Fisher's exact tests
    print("1/5 Running Fisher's exact tests...")
    fisher_results = analyzer.fisher_exact_analysis()
    
    # 2. Bootstrap confidence intervals
    print("2/5 Calculating bootstrap confidence intervals...")
    bootstrap_results = analyzer.bootstrap_confidence_intervals()
    
    # 3. Permutation test
    print("3/5 Running permutation test...")
    perm_results = analyzer.permutation_test()
    
    # 4. Motif enrichment
    print("4/5 Analyzing motif enrichment...")
    motif_results = analyzer.motif_enrichment_analysis()
    
    # 5. Generate visualizations and summary
    print("5/5 Creating high-quality visualizations...")
    
    # Create plots
    fig1 = analyzer.plot_confidence_intervals(bootstrap_results)
    fig2 = analyzer.plot_effect_sizes(bootstrap_results)
    
    # Generate summary
    summary = analyzer.generate_statistical_summary(fisher_results, bootstrap_results, motif_results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"📁 All results saved to: {analyzer.output_dir}")
    print("🖼️  High-quality plots saved as TIFF and PDF files")
    
    return {
        'analyzer': analyzer,
        'fisher_results': fisher_results,
        'bootstrap_results': bootstrap_results,
        'permutation_results': perm_results,
        'motif_results': motif_results,
        'summary': summary,
        'output_directory': analyzer.output_dir
    }

# =============================================================================
# COMPLETE WORKFLOW
# =============================================================================

print("🧬 saRNA Statistical Analysis Pipeline")
print("="*50)

# Step 1: Check what Excel files are available
print("📁 Checking for Excel files...")
print(f"Current directory: {os.getcwd()}")
excel_files = glob.glob("*.xlsx") + glob.glob("*.xls")
print(f"Excel files found: {excel_files}")

# Step 2: Load your data (note the capital D in Designs.xlsx)
print("\n📊 Loading saRNA sequence data...")
sequences_df = load_sequences_from_excel("Designs.xlsx")  # Note: Capital D

# Step 3: Run analysis if data loaded successfully
if sequences_df is not None:
    print("✅ Data loaded successfully!")
    print(f"   - Total sequences: {len(sequences_df)}")
    print(f"   - Positive controls: {len(sequences_df[sequences_df['Type'] == 'POS_CTRL'])}")
    print(f"   - Test sequences: {len(sequences_df[sequences_df['Type'] == 'NON_CTRL'])}")
    
    # Run the complete statistical analysis
    print("\n🔬 Starting comprehensive statistical analysis...")
    results = run_statistical_analysis(sequences_df)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📁 Results saved in folder: {results['output_directory']}")
    
    # Show quick summary
    bootstrap_data = results['bootstrap_results']
    large_effects = bootstrap_data[abs(bootstrap_data['difference']) > 0.1]
    
    if len(large_effects) > 0:
        print("\n🎯 Key findings - positions with large effect sizes:")
        for _, row in large_effects.nlargest(5, 'difference', keep='all').iterrows():
            print(f"   Position {row['position']} {row['nucleotide']}: {row['difference']:+.3f}")

else:
    print("❌ Could not load data. Please check your file.")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)

# %%
# Extract the sequences dataframe
sequences_df = aligner.sequences_df
results = run_statistical_analysis(sequences_df)

# %%



