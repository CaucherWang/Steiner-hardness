"""
steiner_hardness(data, queries, k, work_dir, ...)

Computes Steiner hardness for each query against a dataset using the MRNG-based
pipeline. Intermediate artifacts (MRNG index, reversed graph, ground truth) are
cached in work_dir so repeated calls with the same data skip the expensive build.

Pipeline:
  1. Write data/queries to fvecs files (skipped if already present and matching).
  2. Build KGraph via Efanna (skipped if .efanna file exists).
  3. Build MRNG index from KGraph (skipped if .mrng file exists).
  4. Build reversed MRNG graph (skipped if _reversed_std file exists).
  5. Compute ground truth via compute_gt (skipped if .ivecs file exists).
  6. Run get_delta0_point purpose=0 (delta0 points).
  7. Run get_delta0_point purpose=1 (minimum effort / Steiner hardness).
  8. Return result as numpy int32 array of shape (n_queries,).
"""

import hashlib
import os
import struct
import subprocess
from pathlib import Path
from utils import *
import faiss
import numpy as np
import shutil

# ---------------------------------------------------------------------------
# Paths to compiled executables and source files.
# These are resolved relative to this file's location.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent
_SRC_DIR = _THIS_DIR.parent / "src"
_EFANNA_DIR = _THIS_DIR.parent / "efanna"


# ---------------------------------------------------------------------------
# Low-level I/O helpers (mirrors utils.py but without the print noise)
# ---------------------------------------------------------------------------


def _data_fingerprint(data: np.ndarray) -> str:
    """SHA-256 of the first 1 MB of data as a cheap identity check."""
    flat = data.ravel().view(np.uint8)
    sample = flat[: min(len(flat), 1 << 20)]
    h = hashlib.sha256(sample).hexdigest()[:16]
    return f"{data.shape[0]}x{data.shape[1]}_{h}"


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _compile(src_file: str, exe_name: str, extra_flags=None):
    src = str(_SRC_DIR / src_file)
    exe = str(_SRC_DIR / exe_name)
    cmd = [
        "g++", "-o", exe, src,
        "-I", str(_SRC_DIR),
        "-O3", "-mavx2", "-fopenmp",
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    subprocess.run(cmd, check=True)


def _run(exe_name: str, args):
    if shutil.which(exe_name) is None:
        exe = str(_SRC_DIR / exe_name)
    else:
        exe = exe_name
    cmd = [exe] + [str(a) for a in args]
    subprocess.run(cmd, check=True)


def _build_efanna(data_path: str, kgraph_path: str, kmrng: int):
    """Build KGraph using Efanna test_nndescent."""
    # efanna_tests = str(_EFANNA_DIR / "tests")
    exe = shutil.which("test_nndescent")
    if exe is None:
        exe = str(_EFANNA_DIR / "tests" / "test_nndescent")
        subprocess.run(["cmake", "."], cwd=str(_EFANNA_DIR), check=True)
        subprocess.run(["make", "-j"], cwd=str(_EFANNA_DIR), check=True)
    subprocess.run(
        [exe, data_path, kgraph_path, str(kmrng), "2400", "10", "15", "100"],
        check=True,
    )


def _build_mrng(data_path: str, mrng_path: str, kgraph_path: str, kmrng: int):
    exe = shutil.which("index_mrng")
    # if not os.path.exists(exe):
    if exe is None:
        exe = str(_SRC_DIR / "index_mrng")
        _compile("index_mrng.cpp", "index_mrng", ["-ffast-math", "-march=native"])
    subprocess.run(
        [exe, "-d", data_path, "-i", mrng_path, "-g", kgraph_path, "-k", str(kmrng), "-m", "0"],
        check=True,
    )


def _build_reversed_mrng(mrng_path: str, revg_path: str):
    """
    Build the reversed MRNG graph in the binary format expected by read_rev_graph().
    Format per node: sequence of uint32 neighbor ids terminated by 0xFFFFFFFF (-1 as uint32).
    """
    # Read MRNG (variable-degree format: each node is [degree, n0, n1, ...] as uint32)
    data = np.memmap(mrng_path, dtype=np.uint32, mode="r")
    n_nodes = 0
    cur = 0
    adjacency = []
    while cur < len(data):
        deg = int(data[cur])
        neighbors = [int(data[cur + 1 + j]) for j in range(deg)]
        adjacency.append(neighbors)
        cur += 1 + deg
        n_nodes += 1

    # Build reverse adjacency
    rev = [[] for _ in range(n_nodes)]
    for i, nbrs in enumerate(adjacency):
        for nb in nbrs:
            rev[nb].append(i)

    # Write in the format read_rev_graph() expects:
    # for each node: uint32 neighbors... then uint32(-1) as sentinel
    with open(revg_path, "wb") as f:
        sentinel = struct.pack("<I", 0xFFFFFFFF)
        for nbrs in rev:
            for nb in nbrs:
                f.write(struct.pack("<I", nb))
            f.write(sentinel)


def _compute_gt(data: np.ndarray, queries: np.ndarray, gt_dim: int, gt_path: str):
    """Compute exact k-NN ground truth using faiss and write as ivecs."""
    d = data.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(data.astype(np.float32))
    _, I = index.search(queries.astype(np.float32), gt_dim)
    write_ivecs(gt_path, I.astype(np.int32))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def steiner_hardness(
    data: np.ndarray,
    queries: np.ndarray,
    k: int,
    work_dir: str = './data/steiner_cache',
    recall: float = 0.98,
    prob: float = 0.98,
    kmrng: int = 2047,
    gt_dim: int = None,
    force_rebuild: bool = False,
) -> np.ndarray:
    """
    Compute Steiner hardness for each query.

    Parameters
    ----------
    data : np.ndarray, shape (n, d), float32
        The dataset vectors.
    queries : np.ndarray, shape (q, d), float32
        The query vectors.
    k : int
        Number of nearest neighbors for the hardness computation.
    work_dir : str
        Directory where intermediate artifacts are cached. Created if absent.
        Reuse the same directory for the same dataset to skip rebuilding.
    recall : float
        Recall target (Acc). Default 0.98.
    prob : float
        Probability lower bound. Default 0.98.
    kmrng : int
        MRNG build parameter (max degree). Default 2047.
    gt_dim : int or None
        Number of neighbors in the ground truth file. Defaults to max(k*10, 1000)
        but capped at n_data. Must be >= k.
    force_rebuild : bool
        If True, ignore cached artifacts and rebuild everything.

    Returns
    -------
    np.ndarray of int32, shape (n_queries,)
        Steiner hardness (minimum effort) for each query.
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    n, d = data.shape
    nq = queries.shape[0]

    if gt_dim is None:
        gt_dim = min(max(k * 10, 1000), n)
    if gt_dim < k:
        raise ValueError(f"gt_dim ({gt_dim}) must be >= k ({k})")

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # Use a fingerprint so callers can detect stale caches from different data.
    fp = _data_fingerprint(data)
    ds = f"dataset_{fp}"
    base = work / ds

    data_path   = str(base / f"{ds}_base.fvecs")
    query_path  = str(base / f"{ds}_query.fvecs")
    kgraph_path = str(base / f"{ds}.efanna")
    mrng_path   = str(base / f"{ds}_K{kmrng}.mrng")
    revg_path   = str(base / f"{ds}_K{kmrng}.mrng_reversed_std")
    # The C++ executable constructs the GT path as:
    #   {base_path}/{ds}/{ds}_groundtruth{GT_num_str}.ivecs
    # GT_num_str is set only for known dataset names (deep, sift, gist, etc.).
    # Our fingerprint name matches none of them, so GT_num_str = "" and the
    # expected file is simply {ds}_groundtruth.ivecs.
    gt_path     = str(base / f"{ds}_groundtruth.ivecs")

    recall_s = f"{recall:.2f}"
    prob_s   = f"{prob:.2f}"
    delta0_path = str(base / f"{ds}_delta0_forall_point_recall{recall_s}_prob{prob_s}_k{k}_K{kmrng}.ibin_mrng")
    result_path = str(base / f"{ds}_me_exhausted_forall_point_recall{recall_s}_prob{prob_s}_k{k}_K{kmrng}.ibin_mrng")

    base.mkdir(parents=True, exist_ok=True)

    def _missing(path):
        return force_rebuild or not os.path.exists(path)

    # Step 1: write data/queries
    if _missing(data_path):
        print(f"[steiner_hardness] Writing data ({n}x{d}) to {data_path}")
        write_fvecs(data_path, data)
    if _missing(query_path):
        print(f"[steiner_hardness] Writing queries ({nq}x{d}) to {query_path}")
        write_fvecs(query_path, queries)

    # Step 2: build KGraph
    if _missing(kgraph_path):
        print(f"[steiner_hardness] Building KGraph (Efanna) ...")
        _build_efanna(data_path, kgraph_path, kmrng)

    # Step 3: build MRNG
    if _missing(mrng_path):
        print(f"[steiner_hardness] Building MRNG index ...")
        _build_mrng(data_path, mrng_path, kgraph_path, kmrng)

    # Step 4: build reversed MRNG
    if _missing(revg_path):
        print(f"[steiner_hardness] Building reversed MRNG graph ...")
        _build_reversed_mrng(mrng_path, revg_path)

    # Step 5: compute ground truth
    if _missing(gt_path):
        print(f"[steiner_hardness] Computing ground truth (k={gt_dim}) ...")
        _compute_gt(data, queries, gt_dim, gt_path)

    # Compile get_delta0_point if needed
    exe_path = shutil.which("get_delta0_point")
    if exe_path is None:
    # if not os.path.exists(exe_path):
        print(f"[steiner_hardness] Compiling get_delta0_point ...")
        _compile("get_delta0_point.cpp", "get_delta0_point")
        exe_path = str(_SRC_DIR / "get_delta0_point")

    common_args = [
        "-d", ds,
        "-k", k,
        "-r", recall,
        "-p", prob,
        "-m", "13",
        "-b", str(work),
        "-l", kmrng,
    ]

    # Step 6: compute delta0 points (purpose=0)
    if _missing(delta0_path):
        print(f"[steiner_hardness] Computing delta0 points (purpose=0) ...")
        _run("get_delta0_point", common_args + ["-o", 0])

    # Step 7: compute minimum effort / Steiner hardness (purpose=1)
    if _missing(result_path):
        print(f"[steiner_hardness] Computing Steiner hardness (purpose=1) ...")
        _run("get_delta0_point", common_args + ["-o", 1])

    return read_ibin_simple(result_path)


if __name__ == "__main__":
    import h5py

    n = 10000

    with h5py.File("../glove-100-angular.hdf5") as hfp:
        data = hfp["/train"][:10000]
        queries = hfp["/test"][:]

    res = steiner_hardness(
        data, queries, 10
    )
    print(res.shape)
    print(res / n)
    
