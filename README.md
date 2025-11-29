# SNVLFDR: JAX-Accelerated Empirical Bayes Variant Caller

**IMPORTANT: Work in Progess...**

**Although one can (in theory) clone this repo and conduct analysis, I recommend to first get in touch with me. This project lacks documentation and the repo has some unnecessary files.** 

A high-performance Python reimplementation of the SNVLFDR R package. This tool uses Empirical Bayes estimation to calculate Local False Discovery Rates (LFDR) for variant calling in next-generation sequencing data.

**Key Features:**
*   **Speed:** ~300x faster than the original R implementation.
*   **Scalability:** Supports whole-exome scale data via memory-efficient chunking.
*   **Performance:** JAX-accelerated math kernel.

## Requirements

You will require Nix Package Manager [https://nixos.org/download/](https://nixos.org/download/) in order to _build_ the project. I chose Nix because that is what I knew. I have no experience with docker and I just needed a way to make sure that reproudcibility is maintained.  

## Installation

```bash
git clone https://github.com/ladyfing3rs/snvlfdr-jax.git
cd snvlfdr
nix develop
pip install -e .
```
---

Contact: [prabhatdotdubey@iitb.ac.in](mailto:prabhatdotdubey@iitb.ac.in)
