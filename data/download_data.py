"""Download the bike-sharing dataset once and save it as a CSV in this folder.

Students do not need to run this: the CSV is already in the repo.
Source: OpenML "Bike_Sharing_Demand" (Capital Bikeshare, Washington DC, 2011-2012),
the same data scikit-learn uses in its time-related feature engineering example.
"""

from pathlib import Path

from sklearn.datasets import fetch_openml

OUT = Path(__file__).parent / "bike_sharing_hourly.csv"


def main() -> None:
    bikes = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True).frame
    bikes.to_csv(OUT, index=False)
    print(f"saved {len(bikes)} rows to {OUT}")
    print(bikes.head())
    print(bikes.dtypes)


if __name__ == "__main__":
    main()
