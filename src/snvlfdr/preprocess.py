import pysam
import sys
import os
from tqdm import tqdm

def process_region_to_handle(samfile, fasta, chrom, start, end, handle):
    # Iterator for a specific region
    for pileupcolumn in samfile.pileup(chrom, start, end, truncate=True, min_base_quality=0):
        pos = pileupcolumn.pos
        
        if pileupcolumn.n_segments == 0: continue

        try:
            ref_base = fasta.fetch(chrom, pos, pos+1).upper()
        except IndexError:
            continue

        data = {b: {'count': 0, 'mq': [], 'bq': []} for b in ['A', 'C', 'G', 'T', 'N']}
        depth = 0
        
        for pileupread in pileupcolumn.pileups:
            if pileupread.is_del or pileupread.is_refskip: continue
            
            read = pileupread.alignment
            try:
                base = read.query_sequence[pileupread.query_position].upper()
                base_qual = read.query_qualities[pileupread.query_position]
                map_qual = read.mapping_quality
                
                if base in data:
                    data[base]['count'] += 1
                    data[base]['mq'].append(map_qual)
                    data[base]['bq'].append(base_qual)
                    depth += 1
            except (TypeError, IndexError): continue

        # Format: chr pos ref depth = A C G T N
        # Bam-readcount compat: '=' column is dummy
        row = [chrom, str(pos+1), ref_base, str(depth)]
        row.append("=:0:0:0:0:0:0:0:0:0:0:0:0:0") 
        
        for b in ['A', 'C', 'G', 'T', 'N']:
            d = data[b]
            c = d['count']
            avg_mq = sum(d['mq']) / c if c > 0 else 0.0
            avg_bq = sum(d['bq']) / c if c > 0 else 0.0
            row.append(f"{b}:{c}:{avg_mq:.2f}:{avg_bq:.2f}:0:0:0:0:0:0:0:0:0")
            
        handle.write("\t".join(row) + "\n")

def run_preprocessing(bam_file, ref_fasta, output_file, bed_file=None, region_str=None):
    """
    Main entry point for preprocessing.
    """
    if not bed_file and not region_str:
        raise ValueError("Must provide either --bed or --region")

    samfile = pysam.AlignmentFile(bam_file, "rb")
    fasta = pysam.FastaFile(ref_fasta)
    
    with open(output_file, 'w') as out_f:
        if region_str:
            print(f"Processing region: {region_str}")
            try:
                chrom, coords = region_str.split(':')
                start, end = map(int, coords.split('-'))
                process_region_to_handle(samfile, fasta, chrom, start, end, out_f)
            except ValueError:
                print("Error: Region must be in format chr:start-end")
                
        elif bed_file:
            print(f"Processing regions from BED: {bed_file}")
            # Count lines for progress bar
            with open(bed_file, 'r') as f:
                lines = [line for line in f if not line.startswith('#') and line.strip()]
            
            for line in tqdm(lines, desc="Generating CSV", unit="region"):
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                process_region_to_handle(samfile, fasta, chrom, start, end, out_f)

    samfile.close()
    fasta.close()
    print(f"CSV generation complete: {output_file}")
