!pip -q install numpy pandas scipy scikit-learn matplotlib pillow requests umap-learn hdbscan tqdm

import re
import zipfile
import shutil
import pathlib
import warnings
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

import umap.umap_ as umap
import hdbscan

warnings.filterwarnings("ignore")


# ============================================================
# COIL-20 ONLY
# KMeans vs UMAP+HDBSCAN vs CaRG
#
# CaRG = Curvature Aware Reweighted Graph
#
# CaRG pipeline:
# locally scaled nearest-neighbor graph
# + Forman-Ricci curvature score
# + shared-neighbor bridge score
# + shortest-path graph distance
# + UMAP with precomputed distance
# + HDBSCAN clustering
#
# 100 trials, images, side-by-side plots, box plots, tables
# ============================================================

CONFIG = {
    "n_trials": 100,
    "target_n_samples": 1000,

    # 15 objects x 72 views = 1080 images.
    # COIL-20 has 20 objects. Use 15 objects for same scale as COIL-100.
    "coil20_object_ids": list(range(1, 16)),
    "coil20_resize": 32,

    # COIL-20 processed images are grayscale.
    "coil20_grayscale": True,

    "pca_components": 80,
    "random_seed": 42,
    "umap_epochs": 300,

    # KMeans baseline
    "kmeans_n_init": 30,

    # UMAP+HDBSCAN baseline
    "hdbscan_umap_neighbors": 15,
    "hdbscan_umap_min_dist": 0.0,
    "hdbscan_min_cluster_size": 18,
    "hdbscan_min_samples": 5,
    "hdbscan_cluster_selection_method": "eom",

    # CaRG: Curvature Aware Reweighted Graph
    "carg_neighbors": 12,
    "carg_local_scale_k": 5,
    "carg_gamma": 2.2,
    "carg_beta": 2.6,

    # UMAP after CaRG distance construction
    "carg_umap_neighbors": 12,
    "carg_umap_min_dist": 0.0,
    "carg_umap_repulsion_strength": 2.2,

    # HDBSCAN after CaRG UMAP embedding
    "carg_hdbscan_min_cluster_size": 18,
    "carg_hdbscan_min_samples": 5,
    "carg_hdbscan_cluster_selection_method": "eom",

    "output_dir": "outputs_coil20_carg_hdbscan_comparison",
    "figure_dpi": 300,
}

DATA_DIR = pathlib.Path("data_coil20")
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
# UTILITIES
# ============================================================

def maybe_make_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def download_binary(urls, dest_path, chunk_size=2**20, timeout=180):
    dest_path = pathlib.Path(dest_path)
    maybe_make_dir(dest_path.parent)

    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return dest_path

    headers = {"User-Agent": "Mozilla/5.0"}
    last_error = None

    for url in urls:
        try:
            print(f"Downloading: {url}")

            with requests.get(
                url,
                stream=True,
                timeout=timeout,
                headers=headers,
                allow_redirects=True,
            ) as r:
                r.raise_for_status()

                total = int(r.headers.get("content-length", 0))
                tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

                with open(tmp_path, "wb") as f:
                    pbar = tqdm(total=total, unit="B", unit_scale=True) if total > 0 else None

                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            if pbar is not None:
                                pbar.update(len(chunk))

                    if pbar is not None:
                        pbar.close()

                if tmp_path.stat().st_size < 1000:
                    raise RuntimeError("Downloaded file is too small.")

                tmp_path.rename(dest_path)
                return dest_path

        except Exception as exc:
            last_error = exc
            print("  failed:", exc)

    raise RuntimeError(f"Could not download {dest_path.name}. Last error: {last_error}")


def extract_zip(zip_path, extract_dir):
    zip_path = pathlib.Path(zip_path)
    extract_dir = pathlib.Path(extract_dir)
    maybe_make_dir(extract_dir)

    marker = extract_dir / ".extracted"

    if marker.exists():
        return extract_dir

    if not zipfile.is_zipfile(zip_path):
        try:
            zip_path.unlink()
        except Exception:
            pass
        raise RuntimeError(f"{zip_path} is not a valid zip file.")

    print(f"Extracting {zip_path} -> {extract_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    marker.write_text("ok")
    return extract_dir


def find_image_files(root):
    root = pathlib.Path(root)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".ppm"}

    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]

    return sorted(files)


def encode_labels(y):
    return LabelEncoder().fit_transform(np.asarray(y))


def class_counts(y):
    return pd.Series(y).value_counts().sort_index()


def safe_scores(y_true, y_pred):
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
    }


def stratified_subsample(X, y, n_total=1000, rng=None):
    rng = np.random.default_rng(rng)

    X = np.asarray(X)
    y = np.asarray(y)

    if n_total >= len(y):
        idx = np.arange(len(y))
        rng.shuffle(idx)
        return X[idx], y[idx], idx

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

    selected = np.array(selected)
    rng.shuffle(selected)

    return X[selected], y[selected], selected


# ============================================================
# LOAD COIL-20
# ============================================================

def load_coil20():
    coil_dir = DATA_DIR / "COIL20"
    zip_path = DATA_DIR / "coil-20-proc.zip"

    urls = [
        "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip",
        "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip",
    ]

    try:
        download_binary(urls, zip_path)
        extract_zip(zip_path, coil_dir)

    except Exception as exc:
        print("\nAutomatic COIL-20 download failed.")
        print("Please upload coil-20-proc.zip manually in Colab.")

        try:
            from google.colab import files
            uploaded = files.upload()

            for fname in uploaded:
                if fname.lower().endswith(".zip"):
                    shutil.move(fname, zip_path)
                    break

            marker = coil_dir / ".extracted"
            if marker.exists():
                marker.unlink()

            extract_zip(zip_path, coil_dir)

        except Exception:
            raise exc

    image_files = find_image_files(coil_dir)

    if not image_files:
        raise RuntimeError("No COIL-20 image files found.")

    object_ids_set = set(CONFIG["coil20_object_ids"])

    X_list = []
    y_list = []
    image_list = []
    file_list = []

    for fp in image_files:
        name = fp.name.lower()

        m = re.search(r"obj0*(\d+)", name)

        if m is None:
            continue

        obj_id = int(m.group(1))

        if obj_id not in object_ids_set:
            continue

        img = Image.open(fp).convert("L").resize(
            (CONFIG["coil20_resize"], CONFIG["coil20_resize"]),
            Image.BILINEAR,
        )

        arr = np.asarray(img, dtype=np.float32) / 255.0
        feat = arr.ravel()

        X_list.append(feat)
        y_list.append(obj_id)
        image_list.append(arr)
        file_list.append(str(fp))

    if len(X_list) == 0:
        raise RuntimeError("COIL-20 loaded, but selected object IDs were not found.")

    X = np.vstack(X_list)
    y_raw = np.asarray(y_list)
    y = encode_labels(y_raw)
    images = np.stack(image_list)

    print("\nCOIL-20 loaded.")
    print("X shape:", X.shape)
    print("Number of classes:", len(np.unique(y)))
    print("Class counts:")
    print(class_counts(y).to_string())

    return X, y, y_raw, images, file_list


# ============================================================
# FEATURE MAP
# ============================================================

def image_manifold_feature_map(X):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    s = CONFIG["coil20_resize"]

    gray = X.reshape(n, s, s)
    raw_flat = X

    gray = gray - gray.mean(axis=(1, 2), keepdims=True)
    gray = gray / (gray.std(axis=(1, 2), keepdims=True) + 1e-12)

    gx = np.gradient(gray, axis=2)
    gy = np.gradient(gray, axis=1)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    row_proj = gray.mean(axis=2)
    col_proj = gray.mean(axis=1)

    stats = np.hstack([
        gray.mean(axis=(1, 2)).reshape(n, 1),
        gray.std(axis=(1, 2)).reshape(n, 1),
        gray.max(axis=(1, 2)).reshape(n, 1),
        gray.min(axis=(1, 2)).reshape(n, 1),
    ])

    return np.hstack([
        raw_flat,
        gray.reshape(n, -1),
        grad.reshape(n, -1),
        row_proj,
        col_proj,
        stats,
    ])


def preprocess_features(X, seed):
    n_components = min(
        CONFIG["pca_components"],
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
# HDBSCAN UTILITIES
# ============================================================

def reassign_noise_to_nearest_cluster(Z, labels):
    """
    HDBSCAN may label some points as noise with label -1.
    For ARI/NMI comparison, this assigns each noise point to the nearest
    non-noise cluster center in the embedding.
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


def run_hdbscan_on_embedding(Z, min_cluster_size, min_samples, cluster_selection_method):
    Zs = StandardScaler().fit_transform(Z)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
    )

    raw_labels = clusterer.fit_predict(Zs)
    labels = reassign_noise_to_nearest_cluster(Zs, raw_labels)

    info = {
        "k_found": len(np.unique(labels)),
        "raw_k_found": len(set(raw_labels)) - (1 if -1 in raw_labels else 0),
        "noise_frac": float(np.mean(raw_labels == -1)),
    }

    return labels, info


# ============================================================
# CLUSTERING METHODS
# ============================================================

def run_kmeans(X, k, seed):
    return KMeans(
        n_clusters=k,
        n_init=CONFIG["kmeans_n_init"],
        random_state=seed,
    ).fit_predict(X)


def run_umap_hdbscan(X, seed):
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

    labels, info = run_hdbscan_on_embedding(
        Z,
        min_cluster_size=CONFIG["hdbscan_min_cluster_size"],
        min_samples=CONFIG["hdbscan_min_samples"],
        cluster_selection_method=CONFIG["hdbscan_cluster_selection_method"],
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

    labels, info = run_hdbscan_on_embedding(
        Z,
        min_cluster_size=CONFIG["carg_hdbscan_min_cluster_size"],
        min_samples=CONFIG["carg_hdbscan_min_samples"],
        cluster_selection_method=CONFIG["carg_hdbscan_cluster_selection_method"],
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
# PLOTTING
# ============================================================

def plot_sample_images(images, y):
    classes = np.unique(y)
    n_show = min(12, len(classes))

    fig, axes = plt.subplots(3, 4, figsize=(6.8, 5.0))
    axes = axes.ravel()

    for ax in axes:
        ax.axis("off")

    for ax, c in zip(axes, classes[:n_show]):
        idx = np.where(y == c)[0][0]
        ax.imshow(images[idx], cmap="gray")
        ax.set_title(f"class {c}")
        ax.axis("off")

    fig.suptitle("COIL-20 selected object samples", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = OUT_DIR / "coil20_samples.png"
    plt.savefig(out, bbox_inches="tight", dpi=CONFIG["figure_dpi"])
    plt.show()

    print("Saved:", out)


def scatter_panel(ax, Z, labels, title):
    ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=labels,
        s=8,
        alpha=0.85,
        cmap="tab20",
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_side_by_side(y_true, plot_data):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1))
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

    fig.suptitle("COIL-20: side-by-side clustering comparison", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = OUT_DIR / "coil20_side_by_side.png"
    plt.savefig(out, bbox_inches="tight", dpi=CONFIG["figure_dpi"])
    plt.show()

    print("Saved:", out)


def plot_boxplots(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

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

    fig.suptitle("COIL-20: ARI and NMI box plots", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = OUT_DIR / "coil20_boxplots.png"
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

    # Mean-only table
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
        caption="Mean clustering performance on COIL-20 over 100 trials.",
        label="tab:coil20_mean_scores",
    )

    # Paper-style mean ± std table
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
        caption="COIL-20 clustering performance over 100 trials.",
        label="tab:coil20_clustering",
    )

    return summary, mean_table, paper_table, latex_mean_table, latex_paper_table


# ============================================================
# SINGLE TRIAL
# ============================================================

def run_single_trial(X_feat_all, y_all, trial):
    seed = CONFIG["random_seed"] + 1000 * trial

    X_sub, y_sub, idx = stratified_subsample(
        X_feat_all,
        y_all,
        n_total=CONFIG["target_n_samples"],
        rng=seed,
    )

    X_proc = preprocess_features(
        X_sub,
        seed=seed,
    )

    k = len(np.unique(y_sub))

    rows = []
    plot_data = {}

    Z_common = visual_umap(X_proc, seed)
    plot_data["Ground Truth"] = (Z_common, y_sub)

    # --------------------------------------------------------
    # Baseline 1: KMeans
    # --------------------------------------------------------
    y_km = run_kmeans(X_proc, k, seed)

    rows.append({
        "trial": trial,
        "method": "KMeans",
        **safe_scores(y_sub, y_km),
    })

    plot_data["KMeans"] = (Z_common, y_km)

    # --------------------------------------------------------
    # Baseline 2: UMAP+HDBSCAN
    # --------------------------------------------------------
    y_hdb, Z_hdb, hdb_info = run_umap_hdbscan(X_proc, seed)

    rows.append({
        "trial": trial,
        "method": "UMAP+HDBSCAN",
        **safe_scores(y_sub, y_hdb),
        "k_found": hdb_info["k_found"],
        "raw_k_found": hdb_info["raw_k_found"],
        "noise_frac": hdb_info["noise_frac"],
    })

    plot_data["UMAP+HDBSCAN"] = (Z_hdb, y_hdb)

    # --------------------------------------------------------
    # Proposed method: CaRG
    # CaRG = CaRG distance + UMAP + HDBSCAN
    # --------------------------------------------------------
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
        "noise_frac": carg_info["noise_frac"],
    })

    plot_data["CaRG"] = (Z_carg, y_carg)

    return rows, y_sub, plot_data


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():
    np.random.seed(CONFIG["random_seed"])

    X_all, y_all, y_raw, images, file_list = load_coil20()

    plot_sample_images(images, y_all)

    print("\nBuilding COIL-20 image features once...")
    X_feat_all = image_manifold_feature_map(X_all)
    print("Feature shape:", X_feat_all.shape)

    all_rows = []
    first_y = None
    first_plot_data = None

    for trial in tqdm(range(CONFIG["n_trials"]), desc="COIL-20 trials"):
        rows, y_sub, plot_data = run_single_trial(
            X_feat_all,
            y_all,
            trial,
        )

        all_rows.extend(rows)

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

    results_df.to_csv(
        OUT_DIR / "coil20_results_long.csv",
        index=False,
    )

    (
        summary_df,
        mean_table_df,
        paper_table_df,
        latex_mean_table,
        latex_paper_table,
    ) = make_summary_tables(results_df)

    summary_df.to_csv(
        OUT_DIR / "coil20_summary_mean_std.csv",
        index=False,
    )

    mean_table_df.to_csv(
        OUT_DIR / "coil20_mean_scores_table.csv",
        index=False,
    )

    paper_table_df.to_csv(
        OUT_DIR / "coil20_paper_table.csv",
        index=False,
    )

    with open(OUT_DIR / "coil20_mean_scores_table.tex", "w") as f:
        f.write(latex_mean_table)

    with open(OUT_DIR / "coil20_latex_table.tex", "w") as f:
        f.write(latex_paper_table)

    print("\n==================== FINAL SUMMARY ====================")
    display(summary_df)

    print("\n==================== MEAN SCORE COMPARISON TABLE ====================")
    display(mean_table_df)

    print("\n==================== PAPER-STYLE TABLE ====================")
    display(paper_table_df)

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
    print(OUT_DIR / "coil20_results_long.csv")
    print(OUT_DIR / "coil20_summary_mean_std.csv")
    print(OUT_DIR / "coil20_mean_scores_table.csv")
    print(OUT_DIR / "coil20_mean_scores_table.tex")
    print(OUT_DIR / "coil20_paper_table.csv")
    print(OUT_DIR / "coil20_latex_table.tex")
    print(OUT_DIR / "coil20_samples.png")
    print(OUT_DIR / "coil20_side_by_side.png")
    print(OUT_DIR / "coil20_boxplots.png")

    return results_df, summary_df, mean_table_df, paper_table_df


# ============================================================
# RUN
# ============================================================

# For quick smoke test, uncomment this first:
# CONFIG["n_trials"] = 5

results_df, summary_df, mean_table_df, paper_table_df = run_experiment()
