import os
import pandas as pd
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

# --- Settings ---
INPUT_PATH = '../data/prepared.parquet'  # Changed to parquet
FIGURES_DIR = './figures_prepered'
REPORT_HTML = './profile_prepered.html'
# --------------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Load data from Parquet
df = pd.read_parquet(INPUT_PATH)

# Analysis only - no processing
print("Data loaded successfully. Shape:", df.shape)
print("\nColumn dtypes:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isna().sum())

# Visualizations
ensure_dir(FIGURES_DIR)
numeric_cols = df.select_dtypes(include=['number']).columns
for col in numeric_cols:
    plt.figure()
    df[col].hist(bins=50)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"hist_{col}.png"))
    plt.close()

    plt.figure()
    df.boxplot(column=col)
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"box_{col}.png"))
    plt.close()
print(f"Visualizations saved in {FIGURES_DIR}")

# HTML Profiling Report
ensure_dir(os.path.dirname(REPORT_HTML) or '.')
profile = ProfileReport(
    df,  # Using original dataframe without processing
    title="Data Profiling Report",
    explorative=True
)
profile.to_file(REPORT_HTML)
print(f"Profiling HTML report saved to {REPORT_HTML}")