import argparse
import sys
import time
import os
from snvlfdr.preprocess import run_preprocessing
from snvlfdr.io import parse_bam_readcount_file
from snvlfdr.model import SNVLFDR

def cmd_preprocess(args):
    run_preprocessing(
        bam_file=args.bam,
        ref_fasta=args.ref,
        output_file=args.output,
        bed_file=args.bed,
        region_str=args.region
    )

def cmd_call(args):
    print(f"Loading data from {args.input}...")
    t0 = time.time()
    
    # Use chunksize for memory safety
    df = parse_bam_readcount_file(args.input, bedfile=args.bed, chunksize=args.chunksize)
    print(f"Parsing took: {time.time()-t0:.2f}s. Sites: {len(df)}")
    
    if len(df) == 0:
        print("Error: No valid sites found.")
        sys.exit(1)

    print("Running SNVLFDR Model...")
    model = SNVLFDR(
        pi0_initial=args.pi0,
        fixed_error=args.fixed_error,
        method=args.method,
        # If user sets --strict, we use default clinical filters. 
        # If not (default), we set thresholds to 0 for discovery.
        bq_threshold=20 if args.strict else 0,
        mq_threshold=20 if args.strict else 0,
        af_threshold=0.01 if args.strict else 0.0,
        dp_threshold=10 if args.strict else 0
    )
    
    t_start = time.time()
    results = model.fit(df)
    print(f"Execution Time: {time.time() - t_start:.4f}s")
    print(f"Estimated pi0: {results['estimated_pi0'].iloc[0]:.4f}")
    
    # Save
    out_path = args.output
    results.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="SNVLFDR: Empirical Bayes Variant Calling (JAX-Accelerated)",
        prog="snvlfdr"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Subcommand: preprocess ---
    p_prep = subparsers.add_parser("preprocess", help="Convert BAM to Readcount CSV")
    p_prep.add_argument("-i", "--bam", required=True, help="Input BAM file")
    p_prep.add_argument("-r", "--ref", required=True, help="Reference FASTA")
    p_prep.add_argument("-o", "--output", required=True, help="Output CSV file")
    group = p_prep.add_mutually_exclusive_group(required=True)
    group.add_argument("-b", "--bed", help="Target BED file")
    group.add_argument("--region", help="Specific region (chr:start-end)")
    p_prep.set_defaults(func=cmd_preprocess)

    # --- Subcommand: call ---
    p_call = subparsers.add_parser("call", help="Run LFDR Model on CSV")
    p_call.add_argument("-i", "--input", required=True, help="Input Readcount CSV (from preprocess)")
    p_call.add_argument("-b", "--bed", help="Target BED file (Optional, for filtering)")
    p_call.add_argument("-o", "--output", required=True, help="Output Results CSV")
    p_call.add_argument("--pi0", type=float, default=0.95, help="Initial pi0 guess (default: 0.95)")
    p_call.add_argument("--fixed-error", type=float, default=0.005, help="Global error rate (default: 0.005)")
    p_call.add_argument("--method", default="empirical", choices=["empirical", "uniform"], help="Initialization method")
    p_call.add_argument("--strict", action="store_true", help="Enable strict clinical filters (BQ>20, AF>0.01)")
    p_call.add_argument("--chunksize", type=int, default=50000, help="Parser chunk size")
    p_call.set_defaults(func=cmd_call)

    # --- Subcommand: run (All in One) ---
    # In a full tool, this would chain the two above. 
    # For now, users can pipe them or run sequentially.
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
