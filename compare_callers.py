import pandas as pd
import pysam
import sys
import os

# --- CONFIGURATION ---
MY_TOOL_CSV = "final_variants.csv"

# Competitors (GATK removed)
VCFS = {
    "Bcftools": "calls_bcftools.vcf",
    "VarScan": "calls_varscan.vcf"
}

def load_vcf_variants(vcf_path):
    """Returns set of 'chr:pos' strings for variants in VCF."""
    variants = set()
    if not os.path.exists(vcf_path):
        print(f"Warning: File not found: {vcf_path}")
        return variants
        
    try:
        vcf = pysam.VariantFile(vcf_path)
        for record in vcf:
            # We create a unique ID based on Chromosome and Position
            # Note: VCF is 1-based, SNVLFDR CSV is 1-based. They match.
            vid = f"{record.chrom}:{record.pos}"
            variants.add(vid)
    except Exception as e:
        print(f"Error reading {vcf_path}: {e}")
    return variants

def main():
    print("Loading SNVLFDR results...")
    if not os.path.exists(MY_TOOL_CSV):
        print(f"Error: {MY_TOOL_CSV} not found. Did you run the python benchmark?")
        sys.exit(1)

    # Load your tool's results
    # Filter: We only count sites where Mutant == 1
    df = pd.read_csv(MY_TOOL_CSV)
    my_variants = set(df[df['Mutant'] == 1].apply(lambda x: f"{x['CHR']}:{x['POS']}", axis=1))
    print(f"SNVLFDR detected: {len(my_variants)} variants")

    # Storage for final stats
    stats_rows = []
    
    # Iterate through competitors
    for name, path in VCFS.items():
        print(f"\nLoading {name} ({path})...")
        comp_variants = load_vcf_variants(path)
        print(f"  -> Found {len(comp_variants)} variants")
        
        # Set Operations
        intersection = my_variants.intersection(comp_variants)
        only_mine = my_variants - comp_variants
        only_theirs = comp_variants - my_variants
        
        # Union for Jaccard Index
        union_set = my_variants.union(comp_variants)
        jaccard = len(intersection) / len(union_set) if len(union_set) > 0 else 0.0
        
        print(f"  Comparison vs {name}:")
        print(f"    Shared Variants:     {len(intersection)}")
        print(f"    Unique to SNVLFDR:   {len(only_mine)}")
        print(f"    Unique to {name}:    {len(only_theirs)}")
        print(f"    Agreement (Jaccard): {jaccard:.4f}")
        
        stats_rows.append({
            "Competitor": name,
            "SNVLFDR_Count": len(my_variants),
            "Competitor_Count": len(comp_variants),
            "Shared": len(intersection),
            "Unique_SNVLFDR": len(only_mine),
            "Unique_Competitor": len(only_theirs),
            "Jaccard_Index": round(jaccard, 4)
        })

    # Save Stats
    if stats_rows:
        out_df = pd.DataFrame(stats_rows)
        out_df.to_csv("benchmark_stats.csv", index=False)
        print("\nSummary saved to 'benchmark_stats.csv'")
    else:
        print("\nNo competitors found to compare.")

if __name__ == "__main__":
    main()
