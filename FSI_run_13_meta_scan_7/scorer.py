import numpy as np
from math import atan2, pi
from scipy.spatial import KDTree, distance
from scipy.spatial import Delaunay

def normalize_points(X):
    X = np.asarray(X, float)
    X -= X.mean(axis=0)
    tree = KDTree(X)
    dists, _ = tree.query(X, k=2)
    scale = dists[:,1].mean()
    return X / scale

def psi_n(X, n=4):
    """Bond-orientational order parameter (Delaunay-based)."""
    N = len(X)
    tri = Delaunay(X)
    neighs = [set() for _ in range(N)]
    for s in tri.simplices:
        for i in range(3):
            a,b = s[i], s[(i+1)%3]
            neighs[a].add(b); neighs[b].add(a)
    psi = 0+0j
    for i in range(N):
        if not neighs[i]: continue
        local = 0+0j
        for j in neighs[i]:
            dx, dy = X[j] - X[i]
            theta = atan2(dy, dx)
            local += np.exp(1j * n * theta)
        psi += local / len(neighs[i])
    psi /= N
    return abs(psi)

def pairwise_distance_regular(X, bins=40):
    """Regularity via distance histogram entropy (normalized)."""
    d = distance.pdist(X)
    hist, _ = np.histogram(d, bins=bins, density=True)
    hist = hist[hist > 0]
    p = hist / hist.sum()
    H = -np.sum(p * np.log(p))
    Hmax = np.log(bins)
    R = 1 - H/Hmax
    return float(np.clip(R, 0, 1))

def order_score(X, n=4):
    """Final combined order score (0–1)."""
    Xn = normalize_points(X)
    N = len(Xn)
    psi = psi_n(Xn, n=n)
    R = pairwise_distance_regular(Xn)
    w_psi = min(0.5, 0.4 + 0.015*N)  
    S = w_psi * psi + (1 - w_psi) * R
    return dict(score=S, psi=psi, regularity=R, weight=w_psi, N=N)
