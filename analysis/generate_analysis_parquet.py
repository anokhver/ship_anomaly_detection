import os
import pandas as pd
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

# --- Settings ---
INPUT_PATH = '../data/fix_noise.parquet'
FIGURES_DIR = './figures_prepared'
REPORT_HTML = './profile_parquet.html'

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(file_path):
    """Load data from parquet file."""
    print(f"Loading data from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_parquet(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")

    return df


def clean_anomaly_column(df):
    """Remove anomaly column if it exists."""
    if df.columns[-1] == 'is_anomaly':
        df = df.drop(columns=['is_anomaly'])
        print("Removed 'is_anomaly' column")

    return df


def analyze_data_structure(df):
    """Print basic data analysis information."""
    print("\nData Structure Analysis:")
    print("=" * 40)

    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    print("\nColumn dtypes:")
    print(df.dtypes)

    print("\nMissing values per column:")
    missing_values = df.isna().sum()
    missing_percent = (missing_values / len(df) * 100).round(2)
    missing_summary = pd.DataFrame({
        'Missing Count': missing_values,
        'Missing %': missing_percent
    })
    print(missing_summary[missing_summary['Missing Count'] > 0])

    print(f"\nNumeric columns: {len(df.select_dtypes(include=['number']).columns)}")
    print(f"Categorical columns: {len(df.select_dtypes(include=['object', 'category']).columns)}")


def create_histograms(df, output_dir):
    """Create histogram plots for numeric columns."""
    numeric_cols = df.select_dtypes(include=['number']).columns

    print(f"\nCreating histograms for {len(numeric_cols)} numeric columns...")

    for col in numeric_cols:
        try:
            plt.figure(figsize=(8, 6))
            df[col].hist(bins=50, alpha=0.7, edgecolor='black')
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            output_path = os.path.join(output_dir, f"hist_{col}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error creating histogram for {col}: {e}")
            plt.close()


def create_boxplots(df, output_dir):
    """Create boxplot visualizations for numeric columns."""
    numeric_cols = df.select_dtypes(include=['number']).columns

    print(f"Creating boxplots for {len(numeric_cols)} numeric columns...")

    for col in numeric_cols:
        try:
            plt.figure(figsize=(8, 6))
            df.boxplot(column=col)
            plt.title(f"Boxplot of {col}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            output_path = os.path.join(output_dir, f"box_{col}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error creating boxplot for {col}: {e}")
            plt.close()


def create_visualizations(df, output_dir, include_histograms=False):
    """Create all visualizations for the dataset."""
    ensure_dir(output_dir)

    # Always create boxplots
    create_boxplots(df, output_dir)

    # Optionally create histograms
    if include_histograms:
        create_histograms(df, output_dir)

    print(f"Visualizations saved in {output_dir}")


def generate_profile_report(df, output_path, title="Data Profiling Report"):
    """Generate comprehensive HTML profiling report."""
    print("\nGenerating profiling report...")

    try:
        ensure_dir(os.path.dirname(output_path) or '.')

        profile = ProfileReport(
            df,
            title=title,
            explorative=True,
            minimal=False
        )

        profile.to_file(output_path)
        print(f"Profiling HTML report saved to {output_path}")

    except Exception as e:
        print(f"Error generating profile report: {e}")


def main():
    """Main analysis pipeline."""
    print("Starting data analysis pipeline...")
    print("=" * 50)

    try:
        # Load data
        df = load_data(INPUT_PATH)

        # Clean data
        df = clean_anomaly_column(df)

        # Analyze data structure
        analyze_data_structure(df)

        # Create visualizations (boxplots by default, histograms optional)
        create_visualizations(df, FIGURES_DIR, include_histograms=False)

        # Generate profiling report
        generate_profile_report(df, REPORT_HTML, "Parquet Data Analysis Report")

        print("\n" + "=" * 50)
        print("Analysis completed successfully!")

    except Exception as e:
        print(f"Error in analysis pipeline: {e}")
        raise


if __name__ == "__main__":
    main()