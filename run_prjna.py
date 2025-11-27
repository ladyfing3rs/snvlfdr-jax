# run_prjna.py
import time
from snvlfdr.io import parse_bam_readcount_file
from snvlfdr.model import SNVLFDR

bam_input = "paper_input.csv"
bedfile = "targets_merged.bed"

print("Running Python (JAX) Benchmark (Filters Disabled)...")

t0 = time.time()
df = parse_bam_readcount_file(bam_input, bedfile=bedfile)
t1 = time.time()
print(f"Parsing took: {t1-t0:.4f}s")
print(f"Sites: {len(df)}")

# CHANGE: Set thresholds to 0 to keep healthy sites in the calculation
model = SNVLFDR(
    pi0_initial=0.95,
    bq_threshold=0, 
    mq_threshold=0, 
    af_threshold=0.0,  # <--- CRITICAL
    dp_threshold=0, 
    lfdr_threshold=0.01, 
    method='empirical', epsilon=0.01,
    fixed_error=0.005
)

t_start = time.time()
results = model.fit(df)
t_end = time.time()

print("------------------------------------------------")
print(f"Python (JAX) Execution Time: {t_end - t_start:.4f}s")
if results is not None:
    print(f"Estimated pi0: {results['estimated_pi0'].iloc[0]:.4f}")
print("------------------------------------------------")
