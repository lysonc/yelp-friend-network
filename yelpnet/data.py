"""Load the Yelp friendship network, restaurants and restaurant reviews. Slow parses are cached in data/processed/."""
import json
from array import array

import numpy as np
import pandas as pd

from . import graph
from .paths import PROCESSED, RAW

EDGES, IDS, REVIEWS = PROCESSED / "edges.npy", PROCESSED / "user_ids.txt", PROCESSED / "restaurant_reviews.pkl"


def _records(name):
    with open(RAW / f"yelp_academic_dataset_{name}.json", encoding="utf-8") as f:
        yield from map(json.loads, f)


def parse_edges():
    """Return (edges, ids): unique (u, v) pairs with u < v among users with a record, and ids[i] = user_id of node i."""
    index, users, src, dst = {}, array("i"), array("i"), array("i")
    for rec in _records("user"):
        u = index.setdefault(rec["user_id"], len(index))
        users.append(u)
        for v in rec["friends"].split(", "):
            if v not in ("", "None"):
                src.append(u)
                dst.append(index.setdefault(v, len(index)))
    node = np.full(len(index), -1)
    node[users] = np.arange(len(users))
    e = node[np.column_stack([src, dst])]
    e = np.sort(e[(e >= 0).all(axis=1) & (e[:, 0] != e[:, 1])], axis=1)  # drop friends without a user record
    n = len(users)
    keys = np.unique(e[:, 0] * n + e[:, 1])
    ids = list(index)
    return np.column_stack([keys // n, keys % n]).astype(np.int32), [ids[i] for i in users]


def load_edges():
    if not EDGES.exists():
        edges, ids = parse_edges()
        PROCESSED.mkdir(parents=True, exist_ok=True)
        np.save(EDGES, edges)
        IDS.write_text("\n".join(ids), encoding="utf-8")
    return np.load(EDGES)


def load_ids():
    load_edges()
    return IDS.read_text(encoding="utf-8").split("\n")


def load_adjacency():
    """Sparse adjacency matrix over all users, including those without friends."""
    return graph.adjacency(load_edges(), len(load_ids()))


def load_restaurants():
    """Businesses tagged Restaurants; city is "City, ST" so same-named cities in different states stay apart."""
    cols = ["business_id", "name", "city", "state", "categories", "stars", "review_count"]
    b = pd.DataFrame(list(_records("business")), columns=cols)
    b = b[b.categories.str.contains("Restaurants", na=False)]
    return b.assign(city=b.city.str.strip().str.title() + ", " + b.state)


def load_reviews():
    """Restaurant reviews with user as the node index of the friendship network."""
    if not REVIEWS.exists():
        restaurants = set(load_restaurants().business_id)
        node = {u: i for i, u in enumerate(load_ids())}
        rows = [(node.get(r["user_id"], -1), r["business_id"], r["stars"], r["date"])
                for r in _records("review") if r["business_id"] in restaurants]
        df = pd.DataFrame(rows, columns=["user", "business_id", "stars", "date"])
        df[df.user >= 0].astype({"business_id": "category"}).to_pickle(REVIEWS)
    return pd.read_pickle(REVIEWS)
