import os
import time
from snvlfdr.io import parse_bam_readcount_file
from snvlfdr.model import SNVLFDR

# 1. Create dummy inputs (BAM txt and BED)
# We create a scenario with 1 obvious mutant and 1 reference site
bam_content = """chr1	11174415	G	4053	=:0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00	A:0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00	C:0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00	G:4053:36.98:31.36:36.98:2104:1949:0.00:0.00:0.00:0:0.00:0.00:0.00	T:0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00	N:0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00
chr1	11184544	T	3089	=:0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00	A:0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00	C:9:37.00:32.00:37.00:7:2:0.00:0.01:0.00:0:0.00:0.00:0.00	G:0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00	T:3079:37.00:30.97:37.00:1789:1290:0.00:0.00:0.00:0:0.00:0.00:0.00	N:0:0.00:0.00:0.00:0:0:0.00:0.00:0.00:0:0.00:0.00:0.00	-T:1:37.00:0.00:37.00:1:0:0.00:0.01:0.00:0:0.00:0.00:0.00"""

bed_content = "chr1\t11180000\t11190000" # Includes only the second site

with open("test_input.csv", "w") as f: f.write(bam_content)
with open("test_regions.bed", "w") as f: f.write(bed_content)

# 2. Run Pipeline
print("Running SNVLFDR on NixOS...")
t0 = time.time()

# Test BED filtering
df = parse_bam_readcount_file("test_input.csv", bedfile="test_regions.bed")
print(f"Sites after BED filtering (Should be 1): {len(df)}")

# Test Model with explicit R parameters
model = SNVLFDR(
    pi0_initial=0.95,
    bq_threshold=20,  # BQ.T
    mq_threshold=20,  # MQ.T
    af_threshold=0.001, # AF.T (Low to catch the test variant)
    dp_threshold=10,  # DP.T
    method='empirical'
)

results = model.fit(df)
t1 = time.time()

print("\nFinal Result:")
print(results[['POS', 'REF', 'ALT', 'AF', 'LFDR', 'Mutant']])
print(f"\nTotal execution time: {t1-t0:.4f} seconds")

# Cleanup
if os.path.exists("test_input.csv"): os.remove("test_input.csv")
if os.path.exists("test_regions.bed"): os.remove("test_regions.bed")
