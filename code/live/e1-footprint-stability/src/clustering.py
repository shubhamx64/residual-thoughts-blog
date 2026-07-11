"""Self-clustering (the ontology check): do per-sequence footprints carve the
data the way the human labels do -- or some other way?

k-means (k=3 core, k=5 all) + HDBSCAN (no k) on PCA-50 of per-sequence
footprint vectors, ARI against true labels. Saves 2-D coords for plotting.
"""
import argparse
import json

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score

from common import CLASSES, CORE_CLASSES, result_dir
from metrics import load_records


def seq_matrix(recs, n_layers, dim):
    rows, cols, vals = [], [], []
    for i, r in enumerate(recs):
        for l, (idx, cnt) in enumerate(r["layers"]):
            rows.extend([i] * len(idx))
            cols.extend(l * dim + idx)
            vals.extend(cnt / r["n_tokens"])
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(recs), n_layers * dim))
    return X


def run(recs, classes, tag, out):
    sub = [r for r in recs if r["class"] in classes]
    n_layers = len(sub[0]["layers"])
    dim = int(max(idx.max() if len(idx) else 0 for r in sub for idx, _ in r["layers"])) + 1
    X = seq_matrix(sub, n_layers, dim)
    labels = np.array([classes.index(r["class"]) for r in sub])

    svd = TruncatedSVD(n_components=50, random_state=0)
    Z = svd.fit_transform(X)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    km = KMeans(n_clusters=len(classes), n_init=10, random_state=0).fit(Z)
    ari_km = adjusted_rand_score(labels, km.labels_)

    hdb = HDBSCAN(min_cluster_size=15).fit(Z)
    n_found = len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)
    mask = hdb.labels_ >= 0
    ari_hdb = adjusted_rand_score(labels[mask], hdb.labels_[mask]) if mask.sum() > 10 else 0.0

    # cluster composition table: what does each discovered cluster contain?
    comp = {}
    for cl in sorted(set(hdb.labels_)):
        idxs = np.where(hdb.labels_ == cl)[0]
        comp[str(cl)] = {classes[k]: int((labels[idxs] == k).sum()) for k in range(len(classes))}

    out[tag] = {
        "classes": classes,
        "kmeans_ari": float(ari_km),
        "hdbscan_ari_clustered_only": float(ari_hdb),
        "hdbscan_n_clusters": int(n_found),
        "hdbscan_noise_frac": float((~mask).mean()),
        "hdbscan_composition": comp,
        "svd_explained": float(svd.explained_variance_ratio_.sum()),
    }
    np.savez(result_dir(out["model"]) / f"cluster_coords_{tag}.npz",
             coords=Z[:, :2], labels=labels,
             ids=np.array([r["id"] for r in sub]))
    print(f"{tag}: kmeans ARI {ari_km:.3f} | HDBSCAN found {n_found} clusters, "
          f"ARI(clustered) {ari_hdb:.3f}, noise {(~mask).mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantile", type=float, default=99.0)
    args = ap.parse_args()

    recs = load_records(args.model, args.quantile)
    out = {"model": args.model, "quantile": args.quantile}
    run(recs, CORE_CLASSES, "core3", out)
    run(recs, CLASSES, "all5", out)

    path = result_dir(args.model) / f"clustering_q{args.quantile}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
