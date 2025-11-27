import pandas as pd
import numpy as np

def parse_readcount_row(row, alleles_cols):
    ref_base = row['REF']
    allele_data = []
    
    # R COMPATIBILITY FIX: 
    # The R code strictly looks at specific columns for A, C, G, T, N.
    # It ignores column 4 ('=') and columns > 9 (Indels).
    # We must restrict our parser to indices 5, 6, 7, 8, 9.
    # (Assuming 0-based indexing: 0=CHR, 1=POS, 2=REF, 3=DEPTH, 4='=', 5=A, 6=C, 7=G, 8=T, 9=N)
    
    target_cols = [5, 6, 7, 8, 9]
    
    for col_idx in target_cols:
        # Safety check if file is truncated
        if col_idx not in row or pd.isna(row[col_idx]): continue
        
        val = row[col_idx]
        parts = str(val).split(':')
        if len(parts) < 5: continue
        
        base = parts[0]
        try: count = int(parts[1])
        except ValueError: continue
        if count == 0: continue
        
        try:
            avg_mapq = float(parts[2])
            avg_baseq = float(parts[3])
        except ValueError:
            avg_mapq, avg_baseq = 0.0, 0.0
        
        allele_data.append({'base': base, 'count': count, 'mapq': avg_mapq, 'baseq': avg_baseq})
    
    # Identify Reference
    ref_info = next((x for x in allele_data if x['base'] == ref_base), None)
    if ref_info:
        r_count, r_mq, r_bq = ref_info['count'], ref_info['mapq'], ref_info['baseq']
    else:
        r_count, r_mq, r_bq = 0, 0.0, 0.0
        
    # Identify Alts
    alts = [x for x in allele_data if x['base'] != ref_base]
    alts.sort(key=lambda x: x['count'], reverse=True)
    
    # Pad Alts to 3
    padded_alts = []
    for i in range(3):
        if i < len(alts):
            a_count = alts[i]['count']
            a_q = (alts[i]['mapq'] + alts[i]['baseq']) / 2.0
            padded_alts.append({'count': a_count, 'q': a_q, 'base': alts[i]['base'], 'mapq': alts[i]['mapq'], 'baseq': alts[i]['baseq']})
        else:
            padded_alts.append({'count': 0, 'q': 0.0, 'base': '.', 'mapq': 0.0, 'baseq': 0.0})
            
    # STANDARD: Error uses total depth of parsed alleles (A+C+G+T+N)
    total_depth = r_count + sum(x['count'] for x in padded_alts)
    if total_depth == 0:
        error = 0.01
    else:
        w_sum = (((r_mq + r_bq)/2.0) * r_count) + sum(x['q'] * x['count'] for x in padded_alts)
        avg_q = w_sum / total_depth
        error = 10**(-avg_q / 10.0)

    # STANDARD: AF matches R (Alt / Ref+Alt)
    dom_alt_count = padded_alts[0]['count']
    r_plus_alt = dom_alt_count + r_count
    if r_plus_alt == 0: af = 0.0
    else: af = dom_alt_count / r_plus_alt

    counts = [r_count, padded_alts[0]['count'], padded_alts[1]['count'], padded_alts[2]['count']]
    
    return {
        'CHR': row['CHR'], 'POS': row['POS'], 'REF': ref_base, 'ALT': padded_alts[0]['base'],
        'Counts': counts, 'Error': error, 'AF': af, 'Depth': total_depth,
        'Ref_BQ': r_bq, 'Ref_MQ': r_mq, 'Alt_BQ': padded_alts[0]['baseq'], 'Alt_MQ': padded_alts[0]['mapq']
    }

def parse_bam_readcount_file(filepath, bedfile=None, vcffile=None):
    # 1. Read Main File
    # Ensure we read enough columns to cover up to index 9 (A,C,G,T,N)
    df = pd.read_csv(filepath, sep='\t', header=None, names=list(range(100)))
    df.dropna(axis=1, how='all', inplace=True)
    df.rename(columns={0: 'CHR', 1: 'POS', 2: 'REF', 3: 'RAW_DEPTH'}, inplace=True)
    
    # 2. Filter by BED (Line-by-Line replication)
    if bedfile:
        bed = pd.read_csv(bedfile, sep='\t', header=None, names=['CHR', 'START', 'END'])
        
        # Add explicit unique ID to every BED line
        bed['BED_ID'] = bed.index
        
        merged = df.merge(bed, on='CHR', how='inner')
        mask = (merged['POS'] >= merged['START']) & (merged['POS'] <= merged['END'])
        merged = merged[mask]
        
        # Singleton Drop Logic
        counts = merged.groupby('BED_ID')['POS'].transform('count')
        merged = merged[counts > 1]
        
        df = merged[list(df.columns)].copy()
        df = df.drop_duplicates(subset=['CHR', 'POS'])

    # 3. Filter by VCF
    if vcffile:
        vcf = pd.read_csv(vcffile, sep='\t', comment='#', header=None)
        vcf = vcf[[0, 1]].rename(columns={0: 'CHR', 1: 'POS'})
        df = df.merge(vcf, on=['CHR', 'POS'], how='inner')
        df = df.drop_duplicates(subset=['CHR', 'POS'])

    parsed_rows = []
    # Note: We don't pass allele_col_indices anymore because parse_readcount_row hardcodes 5-9
    # but we pass a dummy list just to keep signature compatible if needed, or pass None.
    
    for idx, row in df.iterrows():
        parsed = parse_readcount_row(row, None) # None because we hardcoded cols inside
        parsed_rows.append(parsed)
        
    return pd.DataFrame(parsed_rows)
