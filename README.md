# yelp-friend-network

18-755 project: social connections and consumer behavior on Yelp ([proposal](doc/milestone0_proposal.pdf)).

## Setup

```bash
pip install -r requirements.txt
```

Put `yelp_academic_dataset_{user,business,review}.json` from the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) in `data/raw/`. Parsed data is cached in `data/processed/`.

## Run

Each step is one module and writes to its own folder in `preliminary_results/`:

```bash
python -m yelpnet.eda             # 0_eda/: whole-network stats and figures
python -m yelpnet.city_selection  # 1_city_selection/: city comparison, then EDA on the chosen city
```

`yelpnet/` also holds the shared modules: `data` (loading and caching), `graph` (network stats), `plot` (figures) and `city` (city subgraphs).

## EDA

Nodes are users, edges are friendships between users who both have a record (most friend IDs in the data do not). 906K of 1.99M users have at least one friend, connected by 7.3M friendships. The degree distribution is heavy tailed (MLE γ ≈ 2.30 for k ≥ 20, against a Poisson baseline), bending down above about 2,000 friends. 98.5% of users with friends are in one component; the next component has 7 users.

Stats and degree plots use every user. `network_sample.png` is a breadth-first sample: 2,000 users reached from one random user in the largest component (`--seed`). A uniform random sample would have almost no edges at this density. The sample favors hubs and shows only part of the network.

## City selection

A city network is the friendship subgraph among users who reviewed at least one restaurant in that city. Top 5 cities by restaurant reviews:

| City | Restaurants | Reviews | Reviewers with a friend | Edges | Largest component |
|---|---|---|---|---|---|
| **Philadelphia, PA** | **5,855** | **688K** | **95.9K (46%)** | **625K** | **97.1%** |
| New Orleans, LA | 2,261 | 477K | 82.4K (41%) | 458K | 93.5% |
| Nashville, TN | 2,505 | 326K | 53.0K (38%) | 234K | 93.8% |
| Tampa, FL | 2,967 | 304K | 45.7K (40%) | 262K | 95.4% |
| Indianapolis, IN | 2,863 | 251K | 33.3K (43%) | 176K | 96.8% |

We use **Philadelphia**. It ranks first on every column: the most restaurants and cuisine categories (382), the most reviews, the largest share of reviewers with friends, and the most dominant component. Its network keeps the whole network's shape (γ ≈ 2.31), so results there should carry over.
