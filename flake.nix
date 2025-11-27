{
  description = "SNVLFDR Bioinfo Environment";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          # Python Setup
          python311
          python311Packages.pip
          python311Packages.venvShellHook
          python311Packages.cython # Often needed for pysam compilation
	  python311Packages.tqdm
	  python311Packages.matplotlib
	  python311Packages.seaborn
          
          # System Libs for compilation
          stdenv.cc.cc.lib
          zlib
          bzip2
          xz
          curl
          htslib # Header files for pysam
          
          # Bioinformatics Tools
          sratoolkit
          bwa
          samtools
          R
	  bedtools

	  # --- Competitors ---
	  bcftools
	  gatk
	  varscan
        ];
        
        # Help compiler find htslib and zlib
        LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.htslib}/lib";
        
        # Pysam needs these to find HTSLIB during 'pip install'
        HTSLIB_LIBRARY_DIR = "${pkgs.htslib}/lib";
        HTSLIB_INCLUDE_DIR = "${pkgs.htslib}/include";
        
        shellHook = ''
          echo "Entering SNVLFDR Bioinfo Shell"
          if [ ! -d ".venv" ]; then python -m venv .venv; fi
          source .venv/bin/activate
          
          pip install --upgrade pip
          
          # Install Pysam from source (ensures version match)
          # We tell it to use the system HTSLIB to speed up build
          pip install pysam --no-binary pysam
          
          pip install "jax[cpu]" pandas scipy numpy
          pip install -e .
        '';
      };
    };
}
