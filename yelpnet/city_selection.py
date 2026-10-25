"""Step 1: compare cities, then run the EDA on the chosen one -> preliminary_results/1_city_selection/."""
import argparse

import pandas as pd

from . import city, data, eda
from .paths import CITY_SELECTION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", default=city.CHOSEN, help="city to run the EDA on")
    p.add_argument("--top", type=int, default=15, help="number of cities, ranked by restaurant reviews")
    p.add_argument("--kmin", type=int, default=20, help="lower cutoff of the power-law fit")
    args = p.parse_args()

    a, reviews, restaurants = data.load_adjacency(), data.load_reviews(), data.load_restaurants()
    df = city.compare(a, reviews, restaurants, args.kmin, args.top)
    CITY_SELECTION.mkdir(parents=True, exist_ok=True)
    df.to_csv(CITY_SELECTION / "city_comparison.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 3):
        print(df)

    name = args.city.split(",")[0]
    eda.run(city.subgraph(a, reviews, restaurants, args.city), CITY_SELECTION / name.lower().replace(" ", "_"),
            name, args.kmin)


if __name__ == "__main__":
    main()
