"""Louvain communities of a friendship network and the restaurant categories their members review."""
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jensenshannon

GENERIC = ["Restaurants", "Food"]  # tags on nearly every restaurant, not a preference


def louvain(a, seed=0):
    """Community label per node (0 = largest) and the modularity of the partition."""
    G = nx.from_scipy_sparse_array(a)
    comms = sorted(nx.community.louvain_communities(G, seed=seed), key=len, reverse=True)
    labels = np.empty(a.shape[0], int)
    for i, c in enumerate(comms):
        labels[list(c)] = i
    return labels, nx.community.modularity(G, comms)


def quotient(a, labels, k):
    """k x k friendship counts between the k largest communities (diagonal: within, counted twice)."""
    keep = np.flatnonzero(labels < k)
    m = csr_matrix((np.ones(keep.size), (keep, labels[keep])), shape=(a.shape[0], k))
    return (m.T @ a @ m).toarray()


def user_categories(reviews, restaurants):
    """One row per (review, category) for reviews of the given restaurants."""
    cats = restaurants.set_index("business_id").categories.str.split(", ").explode().rename("category")
    cats = cats[~cats.isin(GENERIC)]
    uc = reviews[["user", "business_id"]].merge(cats, left_on="business_id", right_index=True)
    return uc[["user", "category"]].reset_index(drop=True)


def lift(uc, k, n_categories):
    """Share of each category in the k largest communities divided by its share overall, for the top categories."""
    overall = uc.category.value_counts(normalize=True)
    top = overall.index[:n_categories]
    share = pd.crosstab(uc.community, uc.category, normalize="index").loc[: k - 1, top]
    return share / overall[top]


def divergence(uc, k, shuffles, seed=0):
    """JS divergence of each community's category mix from the overall mix, observed and averaged over
    shuffles of community labels across users (same community sizes, no friendship structure)."""
    overall = uc.category.value_counts(normalize=True)
    label = uc.drop_duplicates("user").set_index("user").community

    def jsd(lab):
        share = pd.crosstab(uc.user.map(lab), uc.category, normalize="index")
        share = share.loc[: k - 1].reindex(columns=overall.index, fill_value=0)
        return pd.Series([jensenshannon(r, overall.values) for r in share.values], index=share.index)

    rng = np.random.default_rng(seed)
    null = [jsd(pd.Series(rng.permutation(label.values), index=label.index)) for _ in range(shuffles)]
    return pd.DataFrame({"jsd": jsd(label), "null_jsd": pd.concat(null, axis=1).mean(axis=1)})
