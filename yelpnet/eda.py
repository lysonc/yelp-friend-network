"""Step 0: EDA of the whole friendship network -> preliminary_results/0_eda/."""
import argparse
import json

from . import data, graph, plot
from .paths import EDA


def run(a, out, name="Yelp", kmin=20, sample=2000, seed=0):
    """Write stats.json, degree_distribution.png and network_sample.png for adjacency matrix a."""
    stats = graph.summary(a, kmin)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    plot.degree_distribution(graph.degrees(a), out / "degree_distribution.png", kmin, name)
    plot.network(graph.snowball(a, sample, seed), out / "network_sample.png", name)
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kmin", type=int, default=20, help="lower cutoff of the power-law fit")
    p.add_argument("--sample", type=int, default=2000, help="number of users in the drawn subgraph")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print(json.dumps(run(data.load_adjacency(), EDA, **vars(args)), indent=2))


if __name__ == "__main__":
    main()
