# CaRG: Curvature-Aware Manifold Clustering

This repository contains Python code for the project:

**Curvature-Aware Manifold Clustering: Robust Graph Reweighting via Forman-Ricci Flow**

The project compares three clustering methods:

1. KMeans
2. UMAP + HDBSCAN
3. CaRG + UMAP + HDBSCAN

The experiments are run on three datasets:

- COIL-20
- COIL-100
- ALOI-View-BridgeClique15

Each Python file runs one dataset experiment and produces comparison images, ARI/NMI tables, LaTeX tables, and box plots.

## Files


run_coil20.py
run_coil100.py
run_aloi_bridgeclique15.py

## Main Results

Dataset	Method	ARI	NMI
COIL-20	KMeans	0.1566 ± 0.0326	0.5009 ± 0.0331
COIL-20	UMAP+HDBSCAN	0.5829 ± 0.0952	0.7934 ± 0.0211
COIL-20	CaRG	0.7426 ± 0.0321	0.8757 ± 0.0148
COIL-100	KMeans	0.1704 ± 0.0444	0.5298 ± 0.0421
COIL-100	UMAP+HDBSCAN	0.5386 ± 0.1571	0.8147 ± 0.0347
COIL-100	CaRG	0.7088 ± 0.0607	0.8737 ± 0.0229
ALOI-View-BridgeClique15	KMeans	0.2049 ± 0.0562	0.4704 ± 0.0517
ALOI-View-BridgeClique15	UMAP+HDBSCAN	0.4662 ± 0.1263	0.7905 ± 0.0446
ALOI-View-BridgeClique15	CaRG	0.7560 ± 0.0424	0.8899 ± 0.0171

| ALOI-View-BridgeClique15 | UMAP+HDBSCAN | 0.4662 ± 0.1263 | 0.7905 ± 0.0446 |
| ALOI-View-BridgeClique15 | CaRG | **0.7560 ± 0.0424** | **0.8899 ± 0.0171** |
