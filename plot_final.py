import matplotlib.pyplot as plt
import seaborn as sns
from snvlfdr.io import parse_bam_readcount_file
from snvlfdr.model import SNVLFDR
import pandas as pd
import numpy as np

# 1. Run the Model to get fresh results
print("Loading data...")
bam_input = "paper_input.csv"
bedfile = "targets_merged.bed"

df = parse_bam_readcount_file(bam_input, bedfile=bedfile)

print("Running Model...")
# Using the configuration that worked best
model = SNVLFDR(
    pi0_initial=0.95,
    bq_threshold=0, mq_threshold=0, af_threshold=0.0, dp_threshold=0, # No Filters
    lfdr_threshold=0.01, 
    method='empirical', 
    epsilon=0.01,
    fixed_error=0.005 # The fix for deep sequencing
)
results = model.fit(df)

# 2. Prepare Data for Plotting
# We separate "Called Mutants" from "Reference/Noise"
results['Classification'] = np.where(results['Mutant'] == 1, 'Variant Call', 'Background/Noise')

print(f"Total Sites: {len(results)}")
print(f"Variants Found: {results['Mutant'].sum()}")

# 3. Create the Plot (Replicating Figure 4)
plt.figure(figsize=(10, 6))


sns.scatterplot(
    data=results, 
    x='AF', 
    y='LFDR', 
    hue='Classification',
    style='Classification',
    palette={'Variant Call': 'red', 'Background/Noise': 'blue'},
    alpha=0.6,
    s=15
)

plt.axhline(0.01, color='green', linestyle='--', label='LFDR Threshold (0.01)')
plt.title(f"Replication of Paper Figure 4 (GM12877 - TST170)\nEstimated $\pi_0$: {results['estimated_pi0'].iloc[0]:.4f}")
plt.xlabel("Allele Frequency (AF)")
plt.ylabel("Local False Discovery Rate (LFDR)")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# Save
outfile = "replicated_figure_4.png"
plt.savefig(outfile, dpi=300)
print(f"Plot saved to {outfile}")

# 4. Save the Variant List
variants = results[results['Mutant'] == 1].sort_values('AF', ascending=False)
variants.to_csv("final_variants.csv", index=False)
print("Top 5 Variants found:")
print(variants[['CHR', 'POS', 'REF', 'ALT', 'AF', 'LFDR']].head())
