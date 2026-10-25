"""City friendship networks: the subgraph induced by users who reviewed a restaurant in the city."""
import numpy as np
import pandas as pd

from . import graph

CHOSEN = "Philadelphia, PA"  # see README, City selection


def users(reviews, restaurants, city):
    """Node indices of users who reviewed at least one restaurant in city."""
    ids = restaurants.business_id[restaurants.city == city]
    return np.unique(reviews.user[reviews.business_id.isin(ids)])


def subgraph(a, reviews, restaurants, city):
    u = users(reviews, restaurants, city)
    return a[u][:, u]


def compare(a, reviews, restaurants, kmin, top=15):
    """Data volume and network structure of the top cities by number of restaurant reviews."""
    city_of = restaurants.set_index("business_id").city
    counts = reviews.business_id.map(city_of).value_counts()
    rows = []
    for city in counts.index[:top]:
        r = restaurants[restaurants.city == city]
        u = users(reviews, restaurants, city)
        rows.append({
            "city": city,
            "restaurants": len(r),
            "categories": r.categories.str.split(", ").explode().nunique(),
            "reviews": int(counts[city]),
            "reviewers": len(u),
            **graph.summary(a[u][:, u], kmin),
        })
    df = pd.DataFrame(rows).rename(columns={"nodes": "reviewers_with_friend"})
    return df.assign(social_frac=df.reviewers_with_friend / df.reviewers)
