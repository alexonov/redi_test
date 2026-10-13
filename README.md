# Bike demand: from laptop model to a model a business can use

ReDi School Hamburg, ML course, session 9: Databricks, MLflow and the ML workflow.

**The task.** A bike-rental company needs to know how many bikes will be rented each hour,
so it can move bikes to the right stations. You build that forecast, keep a record of
everything you try, pick a winner, and update it when new data arrives.

**The data.** Hourly bike rentals from Capital Bikeshare (Washington DC, 2011-2012), with
weather, hour, weekday and season. 17,379 rows. Already in `data/bike_sharing_hourly.csv`.
Original dataset: Fanaee-T and Gama (2013), UCI Machine Learning Repository, CC BY 4.0; downloaded via OpenML.

## Start here

**Follow [GUIDE.md](GUIDE.md)** step by step: setup, the exercises, an MLflow cheat sheet, and fixes for common problems.

## The files

| File | What it does |
|---|---|
| `1_train.py` | Trains one model and logs it to MLflow as one run. **You fill in the TODOs.** |
| `2_register.py` | Finds the run with the lowest error, registers its model, labels it `champion`. |
| `3_predict.py` | Loads "the champion" by name and forecasts a June working day. |
| `4_new_month.py` | July data arrives. Retrains, compares with the champion, moves the label if the new one is better. |
| `5_from_scratch.py` | Write a full MLflow run yourself, tuning a decision tree. **You fill in the TODOs.** |
| `bikes.py` | Shared helpers: data, features, time split. Optional: your own features under `"mine"`. |
| `solution/` | Finished versions of the files with TODOs. |

## Read next

Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022):
- Chapter 1: why ML in a business is different from ML in a course
- Chapter 6: model development, offline evaluation, and experiment tracking
- Chapters 8-9: why models go stale and how teams retrain them
