!pip -q install numpy pandas scipy scikit-learn matplotlib pillow requests umap-learn hdbscan tqdm

import os
import re
import tarfile
import shutil
import pathlib
import warnings
import subprocess
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm.auto import tqdm
from IPython.display import display

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

import umap.umap_ as umap
import hdbscan

warnings.filterwarnings("ignore")


# ============================================================
# ALOI-VIEW-BRIDGECLIQUE15
#
# Methods:
#   KMeans
#   UMAP+HDBSCAN
#   CaRG
#
# CaRG = Curvature Aware Reweighted Graph
#
# CaRG pipeline:
#   locally scaled nearest-neighbor graph
#   + Forman-Ricci curvature score
#   + shared-neighbor bridge score
#   + shortest-path graph distance
#   + UMAP with precomputed distance
#   + HDBSCAN clustering
# ============================================================

CONFIG = {
    "n_trials": 100,

    # Candidate pool.
    # 1..500 is stronger but heavier.
    # If Colab memory is limited, change to list(range(1, 301)).
    "candidate_object_ids": list(range(1, 501)),

    # Final data set.
    "n_selected_objects": 15,
    "views_per_object": 72,
    "target_n_samples": 15 * 72,

    # Image preprocessing.
    "image_size": 32,

    # PCA.
    "pca_components": 80,

    # BridgeClique hard subset search.
    "hard_views_per_object": 36,
    "hard_search_pca_components": 60,
    "hard_search_neighbors": 14,
    "hard_search_local_scale_k": 5,

    "random_seed": 42,
    "umap_epochs": 300,

    # KMeans.
    "kmeans_n_init": 40,

    # UMAP+HDBSCAN baseline.
    "hdbscan_umap_neighbors": 15,
    "hdbscan_umap_min_dist": 0.0,

    # HDBSCAN settings used for both UMAP+HDBSCAN and CaRG.
    "hdbscan_min_cluster_size_grid": [18],
    "hdbscan_min_samples_grid": [5],
    "hdbscan_cluster_selection_method": "eom",

    # CaRG graph.
    "carg_neighbors": 10,
    "carg_local_scale_k": 4,
    "carg_gamma": 4.0,
    "carg_beta": 5.0,

    # UMAP after CaRG distance.
    "carg_umap_neighbors": 10,
    "carg_umap_min_dist": 0.0,
    "carg_umap_repulsion_strength": 3.0,

    "output_dir": "outputs_aloi_bridgeclique15_carg_hdbscan_comparison",
    "figure_dpi": 300,
}

DATA_DIR = pathlib.Path("data_aloi_bridgeclique15")
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = pathlib.Path(CONFIG["output_dir"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
    "KMeans",
    "UMAP+HDBSCAN",
    "CaRG",
]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": CONFIG["figure_dpi"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ============================================================
# BASIC UTILITIES
# ============================================================

def maybe_make_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def image_extension(name):
    return pathlib.Path(str(name).lower()).suffix in {
        ".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".ppm", ".tif", ".tiff"
    }


def encode_labels(y):
    return LabelEncoder().fit_transform(np.asarray(y))


def class_counts(y):
    return pd.Series(y).value_counts().sort_index()


def safe_scores(y_true, y_pred):
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
    }


def stratified_subsample(X, y, images, n_total=1080, rng=None):
    rng = np.random.default_rng(rng)

    X = np.asarray(X)
    y = np.asarray(y)
    images = np.asarray(images)

    if n_total >= len(y):
        idx = np.arange(len(y))
        rng.shuffle(idx)
        return X[idx], y[idx], images[idx], idx

    classes = np.unique(y)
    selected = []

    n_per_class = max(1, n_total // len(classes))

    for c in classes:
        idx = np.where(y == c)[0]
        take = min(n_per_class, len(idx))
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())

    if len(selected) < n_total:
        used = set(selected)
        rest = np.array([i for i in range(len(y)) if i not in used])

        if len(rest) > 0:
            take = min(n_total - len(selected), len(rest))
            selected.extend(rng.choice(rest, size=take, replace=False).tolist())

    selected = np.asarray(selected)
    rng.shuffle(selected)

    return X[selected], y[selected], images[selected], selected


# ============================================================
# DOWNLOAD AND EXTRACTION
# ============================================================

def download_with_wget_or_requests(urls, dest_path, timeout=1200):
    dest_path = pathlib.Path(dest_path)
    maybe_make_dir(dest_path.parent)

    if dest_path.exists() and dest_path.stat().st_size > 1000000:
        if tarfile.is_tarfile(dest_path):
            print(f"Using existing archive: {dest_path}")
            return dest_path
        else:
            print("Existing archive is invalid. Redownloading.")
            dest_path.unlink()

    last_error = None

    for url in urls:
        print(f"\nTrying download:\n{url}")

        try:
            cmd = [
                "wget",
                "-c",
                "--ignore-length",
                "-O",
                str(dest_path),
                url,
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )

            if (
                dest_path.exists()
                and dest_path.stat().st_size > 1000000
                and tarfile.is_tarfile(dest_path)
            ):
                print(f"Downloaded with wget: {dest_path}")
                return dest_path

            last_error = RuntimeError(result.stdout[-2000:])

        except Exception as exc:
            last_error = exc
            print("wget failed:", exc)

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

            if tmp_path.exists():
                tmp_path.unlink()

            with requests.get(
                url,
                stream=True,
                timeout=timeout,
                headers=headers,
                allow_redirects=True,
            ) as r:
                r.raise_for_status()

                total = int(r.headers.get("content-length", 0))
                pbar = tqdm(total=total, unit="B", unit_scale=True) if total > 0 else None

                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=2**20):
                        if chunk:
                            f.write(chunk)
                            if pbar is not None:
                                pbar.update(len(chunk))

                if pbar is not None:
                    pbar.close()

            if tmp_path.exists() and tmp_path.stat().st_size > 1000000:
                tmp_path.rename(dest_path)

                if tarfile.is_tarfile(dest_path):
                    print(f"Downloaded with requests: {dest_path}")
                    return dest_path

                dest_path.unlink()
                raise RuntimeError("Downloaded file is not a valid tar archive.")

        except Exception as exc:
            last_error = exc
            print("requests failed:", exc)

    raise RuntimeError(f"Could not download ALOI archive. Last error: {last_error}")


def parse_aloi_object_id_from_name(name):
    pure = pathlib.PurePosixPath(str(name))
    parts = [p.lower() for p in pure.parts]
    stem = pure.stem.lower()

    for p in parts[:-1]:
        clean = p.replace("obj", "")
        if re.fullmatch(r"0*\d{1,4}", clean):
            val = int(clean)
            if 1 <= val <= 1000:
                return val

    m = re.match(r"(?:obj)?0*(\d{1,4})(?:[_\-.a-z]|$)", stem)
    if m is not None:
        val = int(m.group(1))
        if 1 <= val <= 1000:
            return val

    joined = "/".join(parts + [stem])
    m = re.search(r"(?:^|/)obj0*(\d{1,4})(?:/|_|\.|-|$)", joined)
    if m is not None:
        val = int(m.group(1))
        if 1 <= val <= 1000:
            return val

    return None


def safe_extract_member(tf, member, extract_dir):
    extract_dir = pathlib.Path(extract_dir).resolve()
    target_path = (extract_dir / member.name).resolve()

    if not str(target_path).startswith(str(extract_dir)):
        raise RuntimeError(f"Unsafe tar member path: {member.name}")

    tf.extract(member, extract_dir)


def extract_candidate_aloi_objects(tar_path, extract_dir, candidate_object_ids):
    tar_path = pathlib.Path(tar_path)
    extract_dir = pathlib.Path(extract_dir)
    maybe_make_dir(extract_dir)

    candidate_object_ids = set(int(x) for x in candidate_object_ids)

    marker = extract_dir / (
        f".candidate_extracted_{min(candidate_object_ids)}_"
        f"{max(candidate_object_ids)}_{len(candidate_object_ids)}"
    )

    if marker.exists():
        print(f"Using existing candidate extraction: {extract_dir}")
        return extract_dir

    if not tarfile.is_tarfile(tar_path):
        raise RuntimeError(f"{tar_path} is not a valid tar archive.")

    print("\nExtracting selected ALOI candidate objects only.")
    print(f"Candidate object count: {len(candidate_object_ids)}")
    print(f"Candidate object range: {min(candidate_object_ids)} to {max(candidate_object_ids)}")

    extracted = 0
    examples = []

    with tarfile.open(tar_path, "r:*") as tf:
        for member in tqdm(tf, desc="Scanning/extracting ALOI tar"):
            if not member.isfile():
                continue

            if not image_extension(member.name):
                continue

            if len(examples) < 12:
                examples.append(member.name)

            obj_id = parse_aloi_object_id_from_name(member.name)

            if obj_id in candidate_object_ids:
                safe_extract_member(tf, member, extract_dir)
                extracted += 1

    if extracted == 0:
        print("Example tar members:")
        for ex in examples:
            print("  ", ex)
        raise RuntimeError("No selected ALOI object files were extracted.")

    marker.write_text("ok")
    print(f"Extracted {extracted} image files.")
    return extract_dir


def find_image_files(root):
    root = pathlib.Path(root)
    files = [
        p for p in root.rglob("*")
        if p.is_file() and image_extension(p.name)
    ]
    return sorted(files)


def aloi_view_sort_key(path):
    s = pathlib.Path(path).stem.lower()
    nums = re.findall(r"\d+", s)

    if len(nums) == 0:
        return (999999, s)

    return tuple(int(x) for x in nums)


# ============================================================
# LOAD CANDIDATE ALOI IMAGES
# ============================================================

def load_aloi_candidate_pool():
    tar_path = DATA_DIR / "aloi_grey_red4_view.tar"
    extract_dir = DATA_DIR / "aloi_candidate_view"

    urls = [
        "https://aloi.science.uva.nl/tars/aloi_grey_red4_view.tar",
        "http://aloi.science.uva.nl/tars/aloi_grey_red4_view.tar",
    ]

    try:
        download_with_wget_or_requests(urls, tar_path)
        extract_candidate_aloi_objects(
            tar_path=tar_path,
            extract_dir=extract_dir,
            candidate_object_ids=CONFIG["candidate_object_ids"],
        )

    except Exception as exc:
        print("\nAutomatic ALOI download/extraction failed.")
        print("Upload aloi_grey_red4_view.tar manually, then rerun.")
        print("Error:", exc)

        try:
            from google.colab import files
            uploaded = files.upload()

            found = False

            for fname in uploaded:
                if fname.lower().endswith(".tar"):
                    shutil.move(fname, tar_path)
                    found = True
                    break

            if not found:
                raise RuntimeError("No .tar file was uploaded.")

            extract_candidate_aloi_objects(
                tar_path=tar_path,
                extract_dir=extract_dir,
                candidate_object_ids=CONFIG["candidate_object_ids"],
            )

        except Exception:
            raise exc

    image_files = find_image_files(extract_dir)

    rows = []
    candidate_set = set(CONFIG["candidate_object_ids"])

    for fp in image_files:
        obj_id = parse_aloi_object_id_from_name(fp.as_posix())

        if obj_id in candidate_set:
            rows.append({
                "path": fp,
                "object_id": int(obj_id),
            })

    meta = pd.DataFrame(rows)

    if meta.empty:
        raise RuntimeError("No candidate ALOI image files found after extraction.")

    kept = []

    for obj_id in sorted(meta["object_id"].unique()):
        sub = meta[meta["object_id"] == obj_id].copy()
        sub = sub.sort_values("path", key=lambda col: col.map(aloi_view_sort_key))

        if len(sub) < CONFIG["views_per_object"]:
            print(f"Warning: object {obj_id} has only {len(sub)} views.")

        sub = sub.head(CONFIG["views_per_object"])

        if len(sub) >= max(10, CONFIG["hard_views_per_object"]):
            kept.append(sub)

    meta = pd.concat(kept, ignore_index=True)

    X_img = []
    images = []
    y_raw = []
    file_list = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Loading ALOI candidate images"):
        fp = row["path"]

        img = Image.open(fp).convert("L").resize(
            (CONFIG["image_size"], CONFIG["image_size"]),
            Image.BILINEAR,
        )

        arr = np.asarray(img, dtype=np.float32) / 255.0

        X_img.append(arr.ravel())
        images.append(arr)
        y_raw.append(int(row["object_id"]))
        file_list.append(str(fp))

    X_img = np.vstack(X_img).astype(np.float32)
    images = np.stack(images).astype(np.float32)
    y_raw = np.asarray(y_raw, dtype=int)

    print("\nALOI candidate pool loaded.")
    print("X image shape:", X_img.shape)
    print("Images shape:", images.shape)
    print("Candidate objects:", len(np.unique(y_raw)))
    print("Total samples:", len(y_raw))
    print("Class counts summary:")
    print(pd.Series(y_raw).value_counts().describe().round(2).to_string())

    return X_img, y_raw, images, file_list


# ============================================================
# IMAGE FEATURE MAP
# ============================================================

def image_manifold_feature_map(X_img):
    X_img = np.asarray(X_img, dtype=np.float32)
    n = X_img.shape[0]
    s = CONFIG["image_size"]

    gray = X_img.reshape(n, s, s)

    gray_centered = gray - gray.mean(axis=(1, 2), keepdims=True)
    gray_norm = gray_centered / (gray_centered.std(axis=(1, 2), keepdims=True) + 1e-6)

    gx = np.gradient(gray_norm, axis=2)
    gy = np.gradient(gray_norm, axis=1)
    grad = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)

    row_proj = gray_norm.mean(axis=2)
    col_proj = gray_norm.mean(axis=1)

    stats = np.hstack([
        gray_norm.mean(axis=(1, 2)).reshape(n, 1),
        gray_norm.std(axis=(1, 2)).reshape(n, 1),
        gray_norm.max(axis=(1, 2)).reshape(n, 1),
        gray_norm.min(axis=(1, 2)).reshape(n, 1),
        grad.mean(axis=(1, 2)).reshape(n, 1),
        grad.std(axis=(1, 2)).reshape(n, 1),
    ])

    X_feat = np.hstack([
        gray.reshape(n, -1),
        gray_norm.reshape(n, -1),
        grad.reshape(n, -1),
        row_proj,
        col_proj,
        stats,
    ])

    return X_feat.astype(np.float32)


def preprocess_features(X, seed, pca_components=None):
    if pca_components is None:
        pca_components = CONFIG["pca_components"]

    n_components = min(
        pca_components,
        X.shape[1],
        X.shape[0] - 1,
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler1", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=seed)),
        ("scaler2", StandardScaler()),
    ])

    return pipe.fit_transform(X)


# ============================================================
# CaRG GRAPH
# ============================================================

def build_locally_scaled_knn_graph(X, n_neighbors, local_scale_k):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    n_neighbors = min(n_neighbors, n - 1)
    local_scale_k = min(local_scale_k, n_neighbors)

    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric="euclidean",
    )
    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    distances = distances[:, 1:]
    indices = indices[:, 1:]

    sigma = distances[:, local_scale_k - 1]
    sigma = np.maximum(sigma, 1e-12)

    rows = np.repeat(np.arange(n), n_neighbors)
    cols = indices.ravel()
    d = distances.ravel()

    d_scaled = d / np.sqrt(sigma[rows] * sigma[cols])

    W = csr_matrix((d_scaled, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)
    W.eliminate_zeros()

    return W


def augmented_forman_bridge_scores(W):
    A = W.copy()
    A.data[:] = 1.0
    A = A.tocsr()

    deg = np.asarray(A.sum(axis=1)).ravel()

    coo = A.tocoo()
    mask = coo.row < coo.col

    i_edges = coo.row[mask]
    j_edges = coo.col[mask]

    F = 4.0 - (deg[i_edges] + deg[j_edges])
    Fminus = np.maximum(0.0, -F)

    q75, q25 = np.percentile(Fminus, [75, 25])
    iqr = q75 - q25
    eps = 1e-12

    C = (Fminus - np.median(Fminus)) / (iqr + eps)
    C = np.clip(C, 0.0, 4.0)

    neighbors = [set(A[i].indices) for i in range(W.shape[0])]

    B = np.zeros_like(C)

    for t, (i, j) in enumerate(zip(i_edges, j_edges)):
        ni = neighbors[i]
        nj = neighbors[j]
        denom = max(1.0, min(len(ni), len(nj)))
        common = len(ni.intersection(nj))
        B[t] = 1.0 - common / denom

    B = np.clip(B, 0.0, 1.0)

    return i_edges, j_edges, C, B


def sparse_edge_values(W, rows, cols):
    vals = W[rows, cols]

    if hasattr(vals, "A1"):
        return vals.A1.astype(float)

    vals = np.asarray(vals).ravel()
    return vals.astype(float)


def carg_distance(X):
    W = build_locally_scaled_knn_graph(
        X,
        n_neighbors=CONFIG["carg_neighbors"],
        local_scale_k=CONFIG["carg_local_scale_k"],
    )

    i_edges, j_edges, C, B = augmented_forman_bridge_scores(W)

    base_lengths = sparse_edge_values(W, i_edges, j_edges)

    effective_lengths = base_lengths * (
        1.0
        + CONFIG["carg_gamma"] * C
        + CONFIG["carg_beta"] * B
    )

    n = W.shape[0]

    rows = np.concatenate([i_edges, j_edges])
    cols = np.concatenate([j_edges, i_edges])
    data = np.concatenate([effective_lengths, effective_lengths])

    W_carg = csr_matrix((data, (rows, cols)), shape=(n, n))
    W_carg.eliminate_zeros()

    D = dijkstra(
        csgraph=W_carg,
        directed=False,
        return_predecessors=False,
    )

    finite = np.isfinite(D)

    if not np.all(finite):
        max_finite = D[finite].max() if finite.any() else 1.0
        D[~finite] = 10.0 * max_finite

    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    return D


# ============================================================
# BRIDGECLIQUE OBJECT SELECTION
# ============================================================

def sample_views_per_object(X_feat, y_raw, n_views, seed):
    rng = np.random.default_rng(seed)

    selected = []

    for obj in sorted(np.unique(y_raw)):
        idx = np.where(y_raw == obj)[0]
        take = min(n_views, len(idx))
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())

    selected = np.asarray(selected)
    rng.shuffle(selected)

    return X_feat[selected], y_raw[selected], selected


def select_bridgeclique_aloi_objects(X_feat, y_raw, seed=42):
    """
    Select a mutually bridge-connected group of ALOI objects.

    This uses graph-bridge statistics only.
    No clustering labels or clustering scores are used.
    """

    X_small, y_small, idx_small = sample_views_per_object(
        X_feat=X_feat,
        y_raw=y_raw,
        n_views=CONFIG["hard_views_per_object"],
        seed=seed,
    )

    X_proc = preprocess_features(
        X_small,
        seed=seed,
        pca_components=CONFIG["hard_search_pca_components"],
    )

    W = build_locally_scaled_knn_graph(
        X_proc,
        n_neighbors=CONFIG["hard_search_neighbors"],
        local_scale_k=CONFIG["hard_search_local_scale_k"],
    )

    i_edges, j_edges, C, B = augmented_forman_bridge_scores(W)

    objects = sorted(np.unique(y_small))
    object_to_pos = {obj: t for t, obj in enumerate(objects)}
    m = len(objects)

    pair_edges = np.zeros((m, m), dtype=float)
    pair_bridge_sum = np.zeros((m, m), dtype=float)
    pair_curv_sum = np.zeros((m, m), dtype=float)

    for e, (a, b) in enumerate(zip(i_edges, j_edges)):
        oa = y_small[a]
        ob = y_small[b]

        if oa == ob:
            continue

        ia = object_to_pos[oa]
        ib = object_to_pos[ob]

        pair_edges[ia, ib] += 1
        pair_edges[ib, ia] += 1

        pair_bridge_sum[ia, ib] += B[e]
        pair_bridge_sum[ib, ia] += B[e]

        pair_curv_sum[ia, ib] += C[e]
        pair_curv_sum[ib, ia] += C[e]

    pair_bridge_mean = np.zeros_like(pair_edges)
    pair_curv_mean = np.zeros_like(pair_edges)

    nz = pair_edges > 0
    pair_bridge_mean[nz] = pair_bridge_sum[nz] / pair_edges[nz]
    pair_curv_mean[nz] = pair_curv_sum[nz] / pair_edges[nz]

    if pair_edges.max() > 0:
        pair_edge_norm = pair_edges / pair_edges.max()
    else:
        pair_edge_norm = pair_edges.copy()

    pair_score = (
        2.5 * pair_edge_norm
        + 1.0 * pair_bridge_mean
        + 0.5 * pair_curv_mean
    )

    np.fill_diagonal(pair_score, 0.0)

    start = np.unravel_index(np.argmax(pair_score), pair_score.shape)
    selected_pos = [start[0], start[1]]
    selected_pos = list(dict.fromkeys(selected_pos))

    while len(selected_pos) < CONFIG["n_selected_objects"]:
        remaining = [i for i in range(m) if i not in selected_pos]

        best_i = None
        best_score = -np.inf

        for i in remaining:
            links = pair_score[i, selected_pos]

            mean_link = float(np.mean(links))
            max_link = float(np.max(links))
            min_link = float(np.min(links))

            score = (
                0.65 * mean_link
                + 0.25 * max_link
                + 0.10 * min_link
            )

            if score > best_score:
                best_score = score
                best_i = i

        selected_pos.append(best_i)

    selected_objects = sorted([int(objects[i]) for i in selected_pos])

    rows = []

    for pos, obj in enumerate(objects):
        scores = pair_score[pos, :]
        nonzero = scores[scores > 0]

        if len(nonzero) > 0:
            mean_pair_score = float(np.mean(nonzero))
            max_pair_score = float(np.max(nonzero))
            top5_pair_score = float(np.mean(np.sort(nonzero)[-5:]))
        else:
            mean_pair_score = 0.0
            max_pair_score = 0.0
            top5_pair_score = 0.0

        rows.append({
            "object_id": int(obj),
            "cross_edges_total": float(np.sum(pair_edges[pos, :])),
            "mean_pair_score": mean_pair_score,
            "max_pair_score": max_pair_score,
            "top5_pair_score": top5_pair_score,
            "selected": int(obj in selected_objects),
        })

    difficulty_df = pd.DataFrame(rows).sort_values(
        ["selected", "top5_pair_score", "max_pair_score"],
        ascending=[False, False, False],
    )

    pair_rows = []

    for a in range(len(selected_pos)):
        for b in range(a + 1, len(selected_pos)):
            ia = selected_pos[a]
            ib = selected_pos[b]

            pair_rows.append({
                "object_i": int(objects[ia]),
                "object_j": int(objects[ib]),
                "pair_edges": pair_edges[ia, ib],
                "pair_bridge_mean": pair_bridge_mean[ia, ib],
                "pair_curvature_mean": pair_curv_mean[ia, ib],
                "pair_score": pair_score[ia, ib],
            })

    pair_df = pd.DataFrame(pair_rows).sort_values(
        "pair_score",
        ascending=False,
    )

    print("\n==================== ALOI-VIEW-BRIDGECLIQUE15 SELECTION ====================")
    print("Selected object IDs:", selected_objects)

    print("\nSelected object difficulty summary:")
    display(difficulty_df[difficulty_df["selected"] == 1].round(4))

    print("\nStrongest selected object-object bridges:")
    display(pair_df.head(20).round(4))

    difficulty_df.to_csv(
        OUT_DIR / "aloi_bridgeclique15_object_scores.csv",
        index=False,
    )

    pair_df.to_csv(
        OUT_DIR / "aloi_bridgeclique15_selected_pair_scores.csv",
        index=False,
    )

    return selected_objects, difficulty_df, pair_df


def restrict_to_selected_objects(X_feat, y_raw, images, selected_objects):
    selected_objects = sorted(int(x) for x in selected_objects)

    keep = np.isin(y_raw, selected_objects)

    X_sel = X_feat[keep]
    y_raw_sel = y_raw[keep]
    images_sel = images[keep]

    kept_idx = []

    for obj in selected_objects:
        idx = np.where(y_raw_sel == obj)[0]
        idx = idx[:CONFIG["views_per_object"]]
        kept_idx.extend(idx.tolist())

    kept_idx = np.asarray(kept_idx)

    X_sel = X_sel[kept_idx]
    y_raw_sel = y_raw_sel[kept_idx]
    images_sel = images_sel[kept_idx]

    y = encode_labels(y_raw_sel)

    mapping_df = pd.DataFrame({
        "new_label": sorted(np.unique(y)),
        "object_id": selected_objects,
    })

    print("\nRestricted to ALOI-View-BridgeClique15.")
    print("X selected shape:", X_sel.shape)
    print("Images selected shape:", images_sel.shape)
    print("Number of objects:", len(np.unique(y)))
    print("Class counts:")
    print(class_counts(y).to_string())

    print("\nObject mapping:")
    display(mapping_df)

    return X_sel, y, y_raw_sel, images_sel, mapping_df


# ============================================================
# HDBSCAN UTILITIES
# ============================================================

def reassign_noise_to_nearest_cluster(Z, labels):
    """
    HDBSCAN may label some points as noise with label -1.
    For ARI/NMI comparison, this function assigns each noise point
    to the nearest non-noise cluster center in the embedding.
    """

    labels = labels.copy()
    noise = labels == -1

    if not np.any(noise):
        return labels

    non_noise = ~noise

    if np.sum(non_noise) == 0:
        labels[:] = 0
        return labels

    clusters = sorted(set(labels[non_noise]))
    centers = np.vstack([Z[labels == c].mean(axis=0) for c in clusters])

    noise_points = Z[noise]

    dists = np.linalg.norm(
        noise_points[:, None, :] - centers[None, :, :],
        axis=2,
    )

    nearest = np.argmin(dists, axis=1)

    labels[noise] = np.array([clusters[j] for j in nearest])

    return labels


def choose_hdbscan_by_geometry(Z, target_k):
    """
    Runs HDBSCAN on a 2D UMAP embedding.

    The grid can contain one fixed choice or several choices.
    The selected model is the one closest to target_k, then with better
    silhouette score, then with lower raw noise fraction.
    """

    Zs = StandardScaler().fit_transform(Z)

    best = None

    for min_cluster_size in CONFIG["hdbscan_min_cluster_size_grid"]:
        for min_samples in CONFIG["hdbscan_min_samples_grid"]:

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=CONFIG["hdbscan_cluster_selection_method"],
            )

            raw = clusterer.fit_predict(Zs)
            labels = reassign_noise_to_nearest_cluster(Zs, raw)

            raw_k_found = len(set(raw)) - (1 if -1 in raw else 0)
            k_found = len(np.unique(labels))
            noise_frac = float(np.mean(raw == -1))

            if k_found >= 2:
                try:
                    sil = silhouette_score(Zs, labels)
                except Exception:
                    sil = -1.0
            else:
                sil = -1.0

            score = (
                abs(k_found - target_k),
                -sil,
                noise_frac,
            )

            if best is None or score < best["score"]:
                best = {
                    "labels": labels.copy(),
                    "raw_labels": raw.copy(),
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "raw_k_found": raw_k_found,
                    "k_found": k_found,
                    "silhouette": sil,
                    "noise_frac": noise_frac,
                    "score": score,
                }

    return best["labels"], best


# ============================================================
# CLUSTERING METHODS
# ============================================================

def run_kmeans(X, k, seed):
    return KMeans(
        n_clusters=k,
        n_init=CONFIG["kmeans_n_init"],
        random_state=seed,
    ).fit_predict(X)


def run_umap_hdbscan(X, k, seed):
    reducer = umap.UMAP(
        n_neighbors=CONFIG["hdbscan_umap_neighbors"],
        min_dist=CONFIG["hdbscan_umap_min_dist"],
        n_components=2,
        metric="euclidean",
        init="spectral",
        n_epochs=CONFIG["umap_epochs"],
        random_state=seed,
    )

    Z = reducer.fit_transform(X)

    labels, info = choose_hdbscan_by_geometry(
        Z,
        target_k=k,
    )

    return labels, Z, info


def run_carg(X, k, seed):
    D = carg_distance(X)

    reducer = umap.UMAP(
        n_neighbors=CONFIG["carg_umap_neighbors"],
        min_dist=CONFIG["carg_umap_min_dist"],
        spread=1.0,
        n_components=2,
        metric="precomputed",
        init="spectral",
        n_epochs=CONFIG["umap_epochs"],
        repulsion_strength=CONFIG["carg_umap_repulsion_strength"],
        random_state=seed,
    )

    Z = reducer.fit_transform(D)

    labels, info = choose_hdbscan_by_geometry(
        Z,
        target_k=k,
    )

    return labels, Z, info


def visual_umap(X, seed):
    return umap.UMAP(
        n_neighbors=18,
        min_dist=0.05,
        n_components=2,
        init="spectral",
        random_state=seed,
    ).fit_transform(X)


# ============================================================
# GRAPH DIAGNOSTICS
# ============================================================

def graph_bridge_diagnostics(X_proc, y):
    W = build_locally_scaled_knn_graph(
        X_proc,
        n_neighbors=CONFIG["carg_neighbors"],
        local_scale_k=CONFIG["carg_local_scale_k"],
    )

    i_edges, j_edges, C, B = augmented_forman_bridge_scores(W)

    y = np.asarray(y)
    cross = y[i_edges] != y[j_edges]
    within = ~cross

    return {
        "knn_cross_edge_rate": float(np.mean(cross)),
        "bridge_score_cross_mean": float(np.mean(B[cross])) if np.any(cross) else np.nan,
        "bridge_score_within_mean": float(np.mean(B[within])) if np.any(within) else np.nan,
        "curvature_score_cross_mean": float(np.mean(C[cross])) if np.any(cross) else np.nan,
        "curvature_score_within_mean": float(np.mean(C[within])) if np.any(within) else np.nan,
        "n_edges": len(i_edges),
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_sample_images(images, y_raw):
    labels = np.asarray(y_raw)
    classes = np.unique(labels)
    n_show = min(12, len(classes))

    fig, axes = plt.subplots(3, 4, figsize=(6.8, 5.0))
    axes = axes.ravel()

    for ax in axes:
        ax.axis("off")

    for ax, c in zip(axes, classes[:n_show]):
        idx = np.where(labels == c)[0][0]
        ax.imshow(images[idx], cmap="gray")
        ax.set_title(f"object {c}", fontsize=8)
        ax.axis("off")

    fig.suptitle("ALOI-View-BridgeClique15 selected object samples", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = OUT_DIR / "aloi_bridgeclique15_samples.png"
    plt.savefig(out, bbox_inches="tight", dpi=CONFIG["figure_dpi"])
    plt.show()

    print("Saved:", out)


def scatter_panel(ax, Z, labels, title, size=8):
    ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=labels,
        s=size,
        alpha=0.88,
        cmap="tab20",
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_side_by_side(y_true, plot_data):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4))
    axes = axes.ravel()

    scatter_panel(
        axes[0],
        plot_data["Ground Truth"][0],
        y_true,
        "Ground Truth",
    )

    scatter_panel(
        axes[1],
        plot_data["KMeans"][0],
        plot_data["KMeans"][1],
        "KMeans",
    )

    scatter_panel(
        axes[2],
        plot_data["UMAP+HDBSCAN"][0],
        plot_data["UMAP+HDBSCAN"][1],
        "UMAP+HDBSCAN",
    )

    scatter_panel(
        axes[3],
        plot_data["CaRG"][0],
        plot_data["CaRG"][1],
        "CaRG",
    )

    fig.suptitle("ALOI-View-BridgeClique15: side-by-side clustering comparison", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = OUT_DIR / "aloi_bridgeclique15_side_by_side.png"
    plt.savefig(out, bbox_inches="tight", dpi=CONFIG["figure_dpi"])
    plt.show()

    print("Saved:", out)


def plot_boxplots(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))

    for ax, metric in zip(axes, ["ARI", "NMI"]):
        data = [
            results_df[results_df["method"] == method][metric].values
            for method in METHOD_ORDER
        ]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            labels=METHOD_ORDER,
            showmeans=True,
            meanprops={"marker": "o", "markersize": 3},
            medianprops={"linewidth": 1.5},
        )

        for box in bp["boxes"]:
            box.set_alpha(0.55)

        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("ALOI-View-BridgeClique15: ARI and NMI box plots", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = OUT_DIR / "aloi_bridgeclique15_boxplots.png"
    plt.savefig(out, bbox_inches="tight", dpi=CONFIG["figure_dpi"])
    plt.show()

    print("Saved:", out)


# ============================================================
# TABLES
# ============================================================

def make_summary_tables(results_df):
    summary = (
        results_df
        .groupby("method")[["ARI", "NMI"]]
        .agg(["mean", "std"])
        .reindex(METHOD_ORDER)
    )

    summary.columns = [
        "ARI_mean",
        "ARI_std",
        "NMI_mean",
        "NMI_std",
    ]

    summary = summary.reset_index()

    # Mean-only comparison table.
    mean_table = summary[[
        "method",
        "ARI_mean",
        "NMI_mean",
    ]].copy()

    mean_table["ARI_mean"] = mean_table["ARI_mean"].map(lambda x: f"{x:.4f}")
    mean_table["NMI_mean"] = mean_table["NMI_mean"].map(lambda x: f"{x:.4f}")

    mean_table = mean_table.rename(columns={
        "method": "Method",
        "ARI_mean": "Mean ARI",
        "NMI_mean": "Mean NMI",
    })

    best_ari_idx = summary["ARI_mean"].idxmax()
    best_nmi_idx = summary["NMI_mean"].idxmax()

    mean_table_latex = mean_table.copy()

    mean_table_latex.loc[best_ari_idx, "Mean ARI"] = (
        r"\textbf{" + mean_table_latex.loc[best_ari_idx, "Mean ARI"] + "}"
    )

    mean_table_latex.loc[best_nmi_idx, "Mean NMI"] = (
        r"\textbf{" + mean_table_latex.loc[best_nmi_idx, "Mean NMI"] + "}"
    )

    latex_mean_table = mean_table_latex.to_latex(
        index=False,
        escape=False,
        caption="Mean clustering performance on ALOI-View-BridgeClique15 over 100 trials.",
        label="tab:aloi_bridgeclique15_mean_scores",
    )

    # Paper-style mean ± std table.
    summary["ARI"] = summary.apply(
        lambda r: f"{r['ARI_mean']:.4f} $\\pm$ {r['ARI_std']:.4f}",
        axis=1,
    )

    summary["NMI"] = summary.apply(
        lambda r: f"{r['NMI_mean']:.4f} $\\pm$ {r['NMI_std']:.4f}",
        axis=1,
    )

    summary.loc[best_ari_idx, "ARI"] = (
        r"\textbf{" + summary.loc[best_ari_idx, "ARI"] + "}"
    )

    summary.loc[best_nmi_idx, "NMI"] = (
        r"\textbf{" + summary.loc[best_nmi_idx, "NMI"] + "}"
    )

    paper_table = summary[[
        "method",
        "ARI",
        "NMI",
    ]].copy()

    paper_table = paper_table.rename(columns={
        "method": "Method",
    })

    latex_paper_table = paper_table.to_latex(
        index=False,
        escape=False,
        caption="ALOI-View-BridgeClique15 clustering performance over 100 trials.",
        label="tab:aloi_bridgeclique15_clustering",
    )

    return summary, mean_table, paper_table, latex_mean_table, latex_paper_table


# ============================================================
# SINGLE TRIAL
# ============================================================

def run_single_trial(X_all, y_all, images_all, trial):
    seed = CONFIG["random_seed"] + 1000 * trial

    X_sub, y_sub, images_sub, idx = stratified_subsample(
        X_all,
        y_all,
        images_all,
        n_total=CONFIG["target_n_samples"],
        rng=seed,
    )

    X_proc = preprocess_features(
        X_sub,
        seed=seed,
        pca_components=CONFIG["pca_components"],
    )

    k = len(np.unique(y_sub))

    rows = []
    plot_data = {}

    Z_common = visual_umap(X_proc, seed)
    plot_data["Ground Truth"] = (Z_common, y_sub)

    # KMeans
    y_km = run_kmeans(X_proc, k, seed)

    rows.append({
        "trial": trial,
        "method": "KMeans",
        **safe_scores(y_sub, y_km),
    })

    plot_data["KMeans"] = (Z_common, y_km)

    # UMAP + HDBSCAN
    y_hdb, Z_hdb, hdb_info = run_umap_hdbscan(
        X_proc,
        k,
        seed,
    )

    rows.append({
        "trial": trial,
        "method": "UMAP+HDBSCAN",
        **safe_scores(y_sub, y_hdb),
        "k_found": hdb_info["k_found"],
        "raw_k_found": hdb_info["raw_k_found"],
        "min_cluster_size": hdb_info["min_cluster_size"],
        "min_samples": hdb_info["min_samples"],
        "silhouette": hdb_info["silhouette"],
        "noise_frac": hdb_info["noise_frac"],
    })

    plot_data["UMAP+HDBSCAN"] = (Z_hdb, y_hdb)

    # CaRG + UMAP + HDBSCAN
    y_carg, Z_carg, carg_info = run_carg(
        X_proc,
        k,
        seed,
    )

    rows.append({
        "trial": trial,
        "method": "CaRG",
        **safe_scores(y_sub, y_carg),
        "k_found": carg_info["k_found"],
        "raw_k_found": carg_info["raw_k_found"],
        "min_cluster_size": carg_info["min_cluster_size"],
        "min_samples": carg_info["min_samples"],
        "silhouette": carg_info["silhouette"],
        "noise_frac": carg_info["noise_frac"],
    })

    plot_data["CaRG"] = (Z_carg, y_carg)

    diagnostics = graph_bridge_diagnostics(X_proc, y_sub)

    return rows, y_sub, images_sub, plot_data, diagnostics


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():
    np.random.seed(CONFIG["random_seed"])

    X_img, y_raw_candidate, images_candidate, file_list = load_aloi_candidate_pool()

    print("\nBuilding image-manifold features for candidate pool...")
    X_feat_candidate = image_manifold_feature_map(X_img)
    print("Candidate feature shape:", X_feat_candidate.shape)

    selected_objects, difficulty_df, pair_df = select_bridgeclique_aloi_objects(
        X_feat_candidate,
        y_raw_candidate,
        seed=CONFIG["random_seed"],
    )

    X_all, y_all, y_raw_all, images_all, mapping_df = restrict_to_selected_objects(
        X_feat_candidate,
        y_raw_candidate,
        images_candidate,
        selected_objects,
    )

    del X_img, X_feat_candidate, y_raw_candidate, images_candidate

    plot_sample_images(images_all, y_raw_all)

    X_preview, y_preview, images_preview, idx_preview = stratified_subsample(
        X_all,
        y_all,
        images_all,
        n_total=CONFIG["target_n_samples"],
        rng=CONFIG["random_seed"],
    )

    X_preview_proc = preprocess_features(
        X_preview,
        seed=CONFIG["random_seed"],
        pca_components=CONFIG["pca_components"],
    )

    diagnostics0 = graph_bridge_diagnostics(X_preview_proc, y_preview)

    print("\n==================== GRAPH DIAGNOSTICS FOR ALOI-BRIDGECLIQUE15 PREVIEW ====================")
    display(pd.DataFrame([diagnostics0]).round(4))

    all_rows = []
    all_diag = []

    first_y = None
    first_plot_data = None

    for trial in tqdm(range(CONFIG["n_trials"]), desc="ALOI-BridgeClique15 trials"):
        rows, y_sub, images_sub, plot_data, diagnostics = run_single_trial(
            X_all,
            y_all,
            images_all,
            trial,
        )

        all_rows.extend(rows)

        diagnostics["trial"] = trial
        all_diag.append(diagnostics)

        if trial == 0:
            first_y = y_sub
            first_plot_data = plot_data

        if (trial + 1) % 10 == 0:
            temp = pd.DataFrame(all_rows)

            means = (
                temp
                .groupby("method")[["ARI", "NMI"]]
                .mean()
                .reindex(METHOD_ORDER)
            )

            print(f"\nMean scores after {trial + 1} trials:")
            display(means.round(4))

    results_df = pd.DataFrame(all_rows)
    diagnostics_df = pd.DataFrame(all_diag)

    results_df.to_csv(
        OUT_DIR / "aloi_bridgeclique15_results_long.csv",
        index=False,
    )

    diagnostics_df.to_csv(
        OUT_DIR / "aloi_bridgeclique15_graph_diagnostics.csv",
        index=False,
    )

    difficulty_df.to_csv(
        OUT_DIR / "aloi_bridgeclique15_object_difficulty.csv",
        index=False,
    )

    pair_df.to_csv(
        OUT_DIR / "aloi_bridgeclique15_pair_scores.csv",
        index=False,
    )

    mapping_df.to_csv(
        OUT_DIR / "aloi_bridgeclique15_object_mapping.csv",
        index=False,
    )

    summary_df, mean_table_df, paper_table_df, latex_mean_table, latex_paper_table = make_summary_tables(results_df)

    summary_df.to_csv(
        OUT_DIR / "aloi_bridgeclique15_summary_mean_std.csv",
        index=False,
    )

    mean_table_df.to_csv(
        OUT_DIR / "aloi_bridgeclique15_mean_scores_table.csv",
        index=False,
    )

    paper_table_df.to_csv(
        OUT_DIR / "aloi_bridgeclique15_paper_table.csv",
        index=False,
    )

    with open(OUT_DIR / "aloi_bridgeclique15_mean_scores_table.tex", "w") as f:
        f.write(latex_mean_table)

    with open(OUT_DIR / "aloi_bridgeclique15_latex_table.tex", "w") as f:
        f.write(latex_paper_table)

    print("\n==================== FINAL SUMMARY ====================")
    display(summary_df)

    print("\n==================== MEAN SCORE COMPARISON TABLE ====================")
    display(mean_table_df)

    print("\n==================== PAPER-STYLE TABLE ====================")
    display(paper_table_df)

    print("\n==================== SELECTED OBJECT MAPPING ====================")
    display(mapping_df)

    print("\n==================== GRAPH DIAGNOSTICS SUMMARY ====================")
    display(diagnostics_df.describe().round(4))

    print("\n==================== LATEX MEAN SCORE TABLE ====================")
    print(latex_mean_table)

    print("\n==================== LATEX PAPER-STYLE TABLE ====================")
    print(latex_paper_table)

    if first_y is not None and first_plot_data is not None:
        plot_side_by_side(first_y, first_plot_data)

    plot_boxplots(results_df)

    print("\nSaved files in:")
    print(OUT_DIR.resolve())

    print("\nMain files:")
    print(OUT_DIR / "aloi_bridgeclique15_results_long.csv")
    print(OUT_DIR / "aloi_bridgeclique15_graph_diagnostics.csv")
    print(OUT_DIR / "aloi_bridgeclique15_object_difficulty.csv")
    print(OUT_DIR / "aloi_bridgeclique15_pair_scores.csv")
    print(OUT_DIR / "aloi_bridgeclique15_object_mapping.csv")
    print(OUT_DIR / "aloi_bridgeclique15_summary_mean_std.csv")
    print(OUT_DIR / "aloi_bridgeclique15_mean_scores_table.csv")
    print(OUT_DIR / "aloi_bridgeclique15_mean_scores_table.tex")
    print(OUT_DIR / "aloi_bridgeclique15_paper_table.csv")
    print(OUT_DIR / "aloi_bridgeclique15_latex_table.tex")
    print(OUT_DIR / "aloi_bridgeclique15_samples.png")
    print(OUT_DIR / "aloi_bridgeclique15_side_by_side.png")
    print(OUT_DIR / "aloi_bridgeclique15_boxplots.png")

    return (
        results_df,
        summary_df,
        mean_table_df,
        paper_table_df,
        diagnostics_df,
        difficulty_df,
        pair_df,
        mapping_df,
    )


# ============================================================
# RUN
# ============================================================

# For a quick smoke test, uncomment this first:
# CONFIG["n_trials"] = 5

results_df, summary_df, mean_table_df, paper_table_df, diagnostics_df, difficulty_df, pair_df, mapping_df = run_experiment()
