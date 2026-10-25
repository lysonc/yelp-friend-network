import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LogNorm

from . import graph

BLUE, ORANGE, RED = "#4C78A8", "#F58518", "#E45756"


def degree_distribution(deg, path, kmin, name="Yelp"):
    """Log-binned P(k) and CCDF on log-log axes, with a Poisson baseline and the power-law MLE fit."""
    deg = deg[deg > 0]
    gamma = graph.powerlaw_gamma(deg, kmin)
    k, p = graph.ccdf(deg)
    kp, pp = graph.ccdf(np.random.default_rng(0).poisson(deg.mean(), deg.size))
    fit = k[k >= kmin]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

    ax[0].plot(*graph.log_binned_pk(deg), "o", ms=5, color=BLUE)
    ax[0].set(title="P(k), logarithmic binning", ylabel="P(degree = k)")

    ax[1].plot(k[p > 0], p[p > 0], "o", ms=3, color=BLUE, label=f"{name} friendship network")
    ax[1].plot(kp[pp > 0], pp[pp > 0], "o", ms=3, color=ORANGE, label=f"Poisson, same mean ({deg.mean():.1f})")
    ax[1].plot(fit, (deg >= kmin).mean() * (fit / kmin) ** (1 - gamma), "--", color=RED,
               label=f"power law, MLE $\\gamma$ = {gamma:.2f} (k $\\geq$ {kmin})")
    ax[1].set(title="CCDF", ylabel="P(degree > k)", ylim=(0.5 / deg.size, 1.5))
    ax[1].legend()

    for a in ax:
        a.set(xscale="log", yscale="log", xlabel="degree k")
        a.grid(alpha=0.3)
    fig.suptitle(f"{name} friendship degree distribution ({deg.size:,} users with $\\geq$ 1 friend)")
    _save(fig, path)


def network(G, path, name="Yelp"):
    deg = np.array([d for _, d in G.degree()])
    pos = nx.spring_layout(G, seed=0)
    fig, ax = plt.subplots(figsize=(11, 11))
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.2, edge_color="gray", alpha=0.15)
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=5 + 3 * np.sqrt(deg), node_color=deg,
                                   cmap="viridis", linewidths=0)
    nodes.set_norm(LogNorm())
    fig.colorbar(nodes, ax=ax, shrink=0.5, label="degree within sample")
    ax.set_title(f"BFS sample of the {name} friendship network ({G.number_of_nodes():,} users, "
                 f"{G.number_of_edges():,} edges)")
    ax.axis("off")
    _save(fig, path)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
