import pandas as pd
from snvlfdr.io import parse_bam_readcount_file
from snvlfdr.model import SNVLFDR

# Paths to your files
bam_input = "data/bam_input.csv"
bedfile = "data/regions.bed"
calls_path = "data/calls.vcf"

print("==================================================")
print("WORKFLOW 1: Standard LFDR Variant Calling")
print("==================================================")

# 1. Parse and Filter (matches R's internal filtering)
# R: output=get_LFDRs(bam_input, bedfile, ...)
print(f"Reading {bam_input} with region filter {bedfile}...")
df_standard = parse_bam_readcount_file(bam_input, bedfile=bedfile)

# 2. Initialize Model with R defaults
model = SNVLFDR(
    pi0_initial=0.95,
    bq_threshold=20,    # BQ.T
    mq_threshold=20,    # MQ.T
    af_threshold=0.01,  # AF.T
    dp_threshold=10,    # DP.T
    lfdr_threshold=0.01, # LFDR.T
    method='empirical',
    epsilon=0.01
)

# 3. Fit
print("Running EM Algorithm...")
results_standard = model.fit(df_standard)

# 4. Outputs
if results_standard is not None:
    print(f"\nEstimated pi0: {results_standard['estimated_pi0'].iloc[0]:.4f}")
    print("\nTop 5 Estimated LFDRs:")
    print(results_standard[['CHR', 'POS', 'AF', 'LFDR', 'Mutant']].head())
    
    # Save filtered bam equivalent
    results_standard.to_csv("output_filtered_bam.csv", index=False)
    print("\nSaved 'output_filtered_bam.csv'")
else:
    print("No variants passed initial filters.")

print("\n\n==================================================")
print("WORKFLOW 2: Prioritize Variants from Caller")
print("==================================================")

# 1. Parse with VCF Filter (matches get_LFDRs_given_caller)
print(f"Reading {bam_input} filtered by {calls_path}...")
df_caller = parse_bam_readcount_file(bam_input, vcffile=calls_path)

if not df_caller.empty:
    # 2. Initialize Model (Parameters are simpler here)
    # Note: We need the size of the original BAM to calculate pi0 exactly like R
    # But for now, we pass None and it defaults to 0.95 or we can estimate.
    # Let's read the raw BAM size quickly for accuracy:
    raw_len = len(pd.read_csv(bam_input, sep='\t', usecols=[0])) 
    
    model_caller = SNVLFDR(lfdr_threshold=0.01)
    
    # 3. Run specific method
    print("Calculating LFDRs for provided candidates...")
    results_caller = model_caller.fit_given_caller(df_caller, original_bam_size=raw_len)
    
    # 4. Outputs
    print("\nTop 5 Updated LFDRs for VCF calls:")
    print(results_caller[['CHR', 'POS', 'AF', 'LFDR', 'Mutant']].head())
    
    results_caller.to_csv("output_updated_vcf.csv", index=False)
    print("\nSaved 'output_updated_vcf.csv'")
else:
    print("No overlap found between BAM and VCF.")
