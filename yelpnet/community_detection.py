"""Step 2: Louvain communities of the chosen city and the restaurant categories they review
-> preliminary_results/2_communities/."""
import argparse
import json

import numpy as np
import pandas as pd

from . import city, community, data, graph, plot
from .paths import COMMUNITIES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", default=city.CHOSEN)
    p.add_argument("--top", type=int, default=10, help="number of largest communities to report")
    p.add_argument("--categories", type=int, default=20, help="number of most reviewed categories in the heatmap")
    p.add_argument("--shuffles", type=int, default=20, help="label shuffles for the random baseline")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    k, name = args.top, args.city.split(",")[0]

    a, reviews, restaurants = data.load_adjacency(), data.load_reviews(), data.load_restaurants()
    users = city.users(reviews, restaurants, args.city)
    s = a[users][:, users]
    lcc = graph.largest_component(s)
    s, users = s[lcc][:, lcc], users[lcc]
    labels, modularity = community.louvain(s, args.seed)
    sizes = np.bincount(labels)

    uc = community.user_categories(reviews, restaurants[restaurants.city == args.city])
    uc = uc[uc.user.isin(users)].assign(community=lambda d: d.user.map(pd.Series(labels, index=users)))
    lift = community.lift(uc, k, args.categories)
    table = community.divergence(uc, k, args.shuffles, args.seed).assign(
        users=sizes[:k], top_category=lift.idxmax(axis=1), top_lift=lift.max(axis=1))

    stats = {
        "nodes": len(users),
        "edges": int(s.nnz // 2),
        "communities": len(sizes),
        "modularity": modularity,
        f"top{k}_coverage": float(sizes[:k].sum() / len(users)),
        "mean_jsd": float(table.jsd.mean()),
        "mean_null_jsd": float(table.null_jsd.mean()),
    }
    print(json.dumps(stats, indent=2))
    COMMUNITIES.mkdir(parents=True, exist_ok=True)
    (COMMUNITIES / "stats.json").write_text(json.dumps(stats, indent=2))
    table = table.rename_axis("community")
    print(table)
    table.to_csv(COMMUNITIES / "communities.csv")
    lift.to_csv(COMMUNITIES / "category_lift.csv")
    plot.community_graph(community.quotient(s, labels, k), sizes[:k],
                         {c: f"C{c}\n{t}" for c, t in table.top_category.items()},
                         COMMUNITIES / "community_graph.png", name)
    plot.lift_heatmap(lift, sizes[:k], COMMUNITIES / "category_lift.png", name)


if __name__ == "__main__":
    main()
