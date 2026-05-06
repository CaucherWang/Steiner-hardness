{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    efanna.url = "github:Cecca/efanna-flake";
  };

  outputs = {
    self,
    nixpkgs,
    efanna,
    ...
  }: let
    eachSystem = f:
      nixpkgs.lib.genAttrs nixpkgs.lib.systems.flakeExposed (system: f nixpkgs.legacyPackages.${system});
  in {
    packages = eachSystem (pkgs: {
      executables = pkgs.stdenv.mkDerivation {
        pname = "steiner-hardness";
        version = "1.0";
        src = ./src;
        buildInputs = [
          pkgs.gperftools
          pkgs.boost
          pkgs.coinmp
        ];
        buildPhase = ''
          g++ -o get_delta0_point get_delta0_point.cpp -I . -O3 -mavx2 -fopenmp -march=native
          g++ -o compute_gt compute_gt.cpp -I . -O3 -mavx2 -fopenmp -march=native
          g++ -o index_mrng index_mrng.cpp -I . -O3 -mavx2 -fopenmp -ffast-math -march=native
        '';
        installPhase = ''
          runHook preInstall
          install -Dm755 get_delta0_point     $out/bin/get_delta0_point
          install -Dm755 compute_gt           $out/bin/compute_gt
          install -Dm755 index_mrng           $out/bin/index_mrng
          runHook postInstall
        '';
      };
    });

    devShells = eachSystem (pkgs: {
      default = pkgs.mkShell {
        packages = [
          pkgs.gperftools
          pkgs.boost
          pkgs.coinmp
          efanna.packages.${pkgs.system}.efanna-graph
          self.packages.${pkgs.system}.executables
          (
            pkgs.python312.withPackages
            (ppkgs:
              with ppkgs; [
                numba
                faiss
                hnswlib
                matplotlib
                networkx
                numpy
                requests
                scikit-learn
                scipy
                tqdm
                h5py
              ])
          )
        ];
        shellHook = ''
          # export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:${pkgs.pythonManylinuxPackages.manylinux2014Package}/lib:$LD_LIBRARY_PATH";
          # test -d .venv || uv venv --python=${pkgs.python312.interpreter} .venv
          # source .venv/bin/activate
        '';
      };
    });
  };
}
