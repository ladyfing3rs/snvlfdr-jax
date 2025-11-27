# SNVLFDR: JAX-Accelerated Empirical Bayes Variant Caller

A high-performance Python reimplementation of the SNVLFDR R package. This tool uses Empirical Bayes estimation to calculate Local False Discovery Rates (LFDR) for variant calling in next-generation sequencing data.

**Key Features:**
*   **Speed:** ~300x faster than the original R implementation.
*   **Scalability:** Supports whole-exome scale data via memory-efficient chunking.
*   **Performance:** JAX-accelerated math kernel.

## Installation

```bash
git clone https://github.com/your-repo/snvlfdr.git
cd snvlfdr
pip install .
```

## Usage

### Preprocessing (BAM -> CSV)
