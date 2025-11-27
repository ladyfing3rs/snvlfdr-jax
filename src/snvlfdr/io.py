import pandas as pd
import numpy as np
import sys

def parse_readcount_row(row, alleles_cols):
    ref_base = str(row['REF']).strip()
    
    bases = []
    counts = []
    mapqs = []
    baseqs = []
    
    # Q-Score Cap to prevent hyper-sensitivity at high depth
    MAX_Q = 40.0
    
    for col_idx in alleles_cols:
        if col_idx not in row or pd.isna(row[col_idx]): continue
        val = str(row[col_idx]).strip()
        parts = val.split(':')
        if len(parts) < 5: continue
        
        try:
            c = int(parts[1])
            if c == 0: continue
            b_name = parts[0].strip()
            
            bases.append(b_name)
            counts.append(c)
            
            # Apply Cap immediately
            mq = min(float(parts[2]), MAX_Q)
            bq = min(float(parts[3]), MAX_Q)
            
            mapqs.append(mq)
            baseqs.append(bq)
        except ValueError:
            continue
            
    # Match Ref vs Alt
    r_count = 0; r_mq = 0.0; r_bq = 0.0
    alts = []
    
    for b, c, mq, bq in zip(bases, counts, mapqs, baseqs):
        if b == ref_base:
            r_count = c
            r_mq = mq
            r_bq = bq
        else:
            alts.append({'base': b, 'count': c, 'mapq': mq, 'baseq': bq})
            
    alts.sort(key=lambda x: x['count'], reverse=True)
    
    padded_alts = []
    for i in range(3):
        if i < len(alts): padded_alts.append(alts[i])
        else: padded_alts.append({'count': 0, 'mapq': 0.0, 'baseq': 0.0, 'base': '.'})
            
    total_depth = r_count + sum(x['count'] for x in alts)
    
    if total_depth == 0:
        error = 0.01
        af = 0.0
    else:
        # Weighted Average with Capped Qualities
        numerator = (((r_mq + r_bq)/2.0) * r_count) + sum(((x['mapq'] + x['baseq'])/2.0) * x['count'] for x in alts)
        error = 10**(-(numerator / total_depth) / 10.0)
        
        # Standard AF
        dom_alt_count = padded_alts[0]['count']
        af = 0.0 if total_depth == 0 else dom_alt_count / total_depth

    return {
        'CHR': str(row['CHR']).strip(),
        'POS': row['POS'], 
        'REF': ref_base, 
        'ALT': padded_alts[0]['base'],
        'Counts': [r_count, padded_alts[0]['count'], padded_alts[1]['count'], padded_alts[2]['count']],
        'Error': float(error),
        'AF': float(af),
        'Depth': int(total_depth),
        'Ref_BQ': float(r_bq), 'Ref_MQ': float(r_mq), 
        'Alt_BQ': float(padded_alts[0]['baseq']), 'Alt_MQ': float(padded_alts[0]['mapq'])
    }

def parse_bam_readcount_file(filepath, bedfile=None, vcffile=None, chunksize=50000):
    bed_mapping = None
    vcf_df = None
    
    if bedfile:
        print(f"Indexing BED file {bedfile}...")
        bed_df = pd.read_csv(bedfile, sep='\t', header=None, names=['CHR', 'START', 'END'])
        bed_mapping = {}
        for chrom, group in bed_df.groupby('CHR'):
            try:
                bed_mapping[chrom] = pd.IntervalIndex.from_arrays(
                    group['START'], group['END'], closed='both'
                )
            except Exception: pass

    if vcffile:
        vcf_df = pd.read_csv(vcffile, sep='\t', comment='#', header=None)
        if vcf_df.shape[1] >= 2:
            vcf_df = vcf_df[[0, 1]].rename(columns={0: 'CHR', 1: 'POS'})

    processed_chunks = []
    print(f"Reading {filepath} in chunks...")
    
    with pd.read_csv(filepath, sep='\t', header=None, names=list(range(50)), chunksize=chunksize) as reader:
        for chunk in reader:
            chunk.dropna(axis=1, how='all', inplace=True)
            chunk.rename(columns={0: 'CHR', 1: 'POS', 2: 'REF', 3: 'RAW_DEPTH'}, inplace=True)
            
            if bed_mapping:
                valid_mask = np.zeros(len(chunk), dtype=bool)
                for chrom in chunk['CHR'].unique():
                    if chrom not in bed_mapping: continue
                    chr_mask = (chunk['CHR'] == chrom)
                    positions = chunk.loc[chr_mask, 'POS']
                    try:
                        indices = bed_mapping[chrom].get_indexer(positions)
                        valid_mask[chr_mask] = (indices != -1)
                    except: pass
                chunk = chunk[valid_mask]
            
            if vcf_df is not None:
                chunk = chunk.merge(vcf_df, on=['CHR', 'POS'], how='inner')
                
            if chunk.empty: continue
            
            allele_col_indices = [c for c in chunk.columns if isinstance(c, int) and c >= 4]
            chunk_results = []
            for idx, row in chunk.iterrows():
                chunk_results.append(parse_readcount_row(row, allele_col_indices))
            
            cdf = pd.DataFrame(chunk_results)
            cols_f = ['Error', 'AF', 'Ref_BQ', 'Ref_MQ', 'Alt_BQ', 'Alt_MQ']
            cdf[cols_f] = cdf[cols_f].astype('float32')
            processed_chunks.append(cdf)
            
            sys.stdout.write(f"\rCaptured {sum(len(x) for x in processed_chunks)} valid sites...")
            sys.stdout.flush()

    print("\nConcatenating results...")
    if not processed_chunks: return pd.DataFrame()
    return pd.concat(processed_chunks, ignore_index=True)
