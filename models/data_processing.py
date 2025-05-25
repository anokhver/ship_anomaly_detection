import os
import pandas as pd
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport


# --- Settings ---
INPUT_PATH = './raw_data/merged.csv'
COLUMNS_TO_DROP = [
    'TripID', 'Name', 'Callsign', 'MMSI'
]
OUTPUT_CLEANED_CSV = './cleaned.csv'
FIGURES_DIR = './figures'
REPORT_HTML = './profile.html'
# --------------------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=['time'],
    engine='python',
    on_bad_lines='skip'
)

# Drop unwanted columns
df_clean = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')

# Save cleaned data
ensure_dir(os.path.dirname(OUTPUT_CLEANED_CSV) or '.')
df_clean.to_csv(OUTPUT_CLEANED_CSV, index=False)
print(f"Cleaned data saved to {OUTPUT_CLEANED_CSV}")

# Visualizations
ensure_dir(FIGURES_DIR)
numeric_cols = df_clean.select_dtypes(include=['number']).columns
for col in numeric_cols:
    plt.figure()
    df_clean[col].hist(bins=50)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"hist_{col}.png"))
    plt.close()

    plt.figure()
    df_clean.boxplot(column=col)
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"box_{col}.png"))
    plt.close()
print(f"Visualizations saved in {FIGURES_DIR}")

# HTML Profiling Report
ensure_dir(os.path.dirname(REPORT_HTML) or '.')
profile = ProfileReport(
    df_clean,
    title="Data Profiling Report",
    explorative=True
)
profile.to_file(REPORT_HTML)
print(f"Profiling HTML report saved to {REPORT_HTML}")