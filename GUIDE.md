# Session 9 guide: from a model on your laptop to a model a business can use

Follow this page step by step. Each part ends with a **Check**: what you should see.
If your screen looks different, raise your hand.

**What you build today.** A model that predicts how many bikes people rent each hour.
You record every training attempt in MLflow, pick the best model, and replace it when new data arrives.
Everything runs in Databricks, in your browser. You install nothing.

**Contents**

| Part | What you do | Time |
|---|---|---|
| 0 | Before class: accounts | at home |
| 1 | Your own copy of the project on GitHub | 5 min |
| 2 | Open it in Databricks | 5 min |
| 3 | First run | 5 min |
| 4 | Make the training script record itself | 10 min |
| 5 | Experiment and compare | 15 min |
| 6 | Write a run yourself, from scratch | 15 min |
| 7 | Pick a champion and use it | 5 min |
| 8 | A new month of data arrives | 5 min |
| 9 | Save your work to GitHub | 5 min |
| | MLflow cheat sheet, problems and fixes, words, what to read | |

---

## Part 0 — Before class

1. **GitHub.** Make sure you can log in at [github.com](https://github.com).
2. **Databricks.** Search for **Databricks Free Edition** and sign up. Use your normal email.
   Free Edition is free, for learning, and needs no credit card.

**Check:** you can log in to both websites.

## Part 1 — Your own copy of the project

The course project is a **template**: a starting point that you copy. Teams start new projects like this.

1. Open **github.com/alexonov/redi-bike-demand**.
2. Click the green button **Use this template** → **Create a new repository**.
3. Owner: you. Repository name: for example `bike-demand`. Keep it **Public** (so Databricks can read it).
4. Click **Create repository**.
5. On your new repo page, click the green **Code** button and copy the **HTTPS** URL.
   It looks like `https://github.com/YOUR-NAME/bike-demand.git`.

**Check:** the page shows **your** name in the top left, not `alexonov`.

## Part 2 — Open it in Databricks

A **Git folder** is a copy of your GitHub repo inside Databricks.

1. Log in to Databricks.
2. In the left menu click **Workspace**. Open **Users** → your email.
3. Top right: **Create** → **Git folder**.
4. Paste your repo URL. The provider (GitHub) and name fill in by themselves.
5. Click **Create Git folder**.

**Check:** you see a folder with `1_train.py`, `2_register.py`, `bikes.py`, `data`, and more.

## Part 3 — First run

1. Click `1_train.py` to open it.
2. Click **Run** at the top. The first time, Databricks needs about a minute to start a computer for you.

**Check:** below the code you see a line like:
```
run 6b38e4...: linear + raw -> validation MAE 145.1 bikes/hour, RMSE 179.5
```
**MAE** is the average miss: on average this model is wrong by 145 bikes per hour. Bad! We fix it soon.

## Part 4 — Make the training script record itself

Right now the script trains a model, prints a score, and forgets everything.
You add three lines so MLflow keeps a record of each run.

1. In `1_train.py`, find **TODO 1**, **TODO 2** and **TODO 3**.
2. Write the code below each TODO. The hints tell you which MLflow command to use; see also the cheat sheet at the end.
3. Run the file again.
4. In the left menu click **Experiments** → `bike-demand`. Click your newest run.

**Check:** the run page shows **Parameters** (`model`, `features`), **Metrics** (`val_mae` = 145.1)
and, under **Artifacts**, a folder `model`.

Stuck? Open `solution/1_train.py` and copy the lines. That's fine: the point is to know where each line goes.

## Part 5 — Experiment and compare

Change the settings at the top of `1_train.py`, run, and repeat. **Change one thing at a time.**
Before you start: write down which run you think will win.

| Run | `MODEL` | `FEATURES` | `MAX_DEPTH` | Question |
|---|---|---|---|---|
| 1 | `"linear"` | `"raw"` | — | baseline (you did this already) |
| 2 | `"linear"` | `"hour_onehot"` | — | does a better feature help? |
| 3 | `"forest"` | `"raw"` | `5` | does a different model help? |
| 4 | `"forest"` | `"raw"` | `15` | does one setting matter? |
| 5 | `"forest"` | `"hour_onehot"` | `15` | both together |
| 6+ | your choice | `"mine"` | your choice | your own idea (see below) |

- **forest** (random forest) = many decision trees, each a bit different; the prediction is the average of their answers.
- **hour_onehot** = hour becomes many yes/no columns, one per hour, separately for work days and days off. Rush hour only happens on work days.
- **mine** = your own features. Open `bikes.py`, search for `"mine"`, add columns. There is an example to start from.

**Compare:** Experiments → `bike-demand` → click the `val_mae` column to sort.
Tick 2 to 5 runs → **Compare**. You see settings and scores side by side.

**Check:** you have at least 5 runs, and you can say which one has the lowest `val_mae`.

## Part 6 — Write a run yourself, from scratch

`1_train.py` used helpers. Now you write all the MLflow code yourself.

1. Open `5_from_scratch.py`.
2. Change `EMAIL` to the email you log in to Databricks with.
3. Fill in TODO 1 to 5. Use the cheat sheet below.
4. Run it. It trains a decision tree 4 times, with `max_depth` 3, 6, 10 and 15.
5. Open **Experiments** → `bike-scratch`. Compare the 4 runs.

**Check:** 4 runs, each with `max_depth` and `val_mae`. Which depth is best?

**Bonus 1:** draw actual vs predicted with matplotlib, and save the picture in the run with `mlflow.log_figure`.
**Bonus 2:** delete your logging lines, add `mlflow.sklearn.autolog()` at the top, and run again. What does MLflow record by itself?

Stuck? `solution/5_from_scratch.py`.

## Part 7 — Pick a champion and use it

1. Run `2_register.py`. It finds the run with the lowest `val_mae` in `bike-demand`,
   registers its model (version 1), and puts the label **champion** on it.
2. In the left menu click **Catalog** → `workspace` → `default` → **Models** → `bike_demand`.
3. Run `3_predict.py`. It loads "the champion" by name and predicts an average June work day, hour by hour.

**Check:** in Catalog you see version 1 with the label `champion`. `3_predict.py` prints a table of hours with actual and predicted bikes.

Why a label? Whoever uses the forecast asks for "the champion". They don't need to know which run won.
When a better model comes, we move the label. Their code stays the same.

## Part 8 — A new month of data arrives

It is now July 2012. Is our champion still the best we can do?

1. Run `4_new_month.py`. It:
   - checks the champion on July,
   - trains the same recipe again on all data up to June (a "challenger"),
   - registers the challenger as a new version,
   - moves the `champion` label to it, **only if** it is better on July.
2. Look in **Catalog** → `bike_demand` again.

**Check:** you see version 2. Which version has the `champion` label now?

People rented many more bikes in 2012 than in 2011. A model trained on old data gets worse over time.
This is called **data drift**, and it is why real models are retrained.

## Part 9 — Save your work to GitHub

1. In your Git folder, click the branch name (`main`) at the top.
2. You see your changed files. Write a short message, for example `fill in MLflow logging`.
3. Click **Commit & Push**.
4. If Databricks asks you to connect GitHub, follow the steps it shows.

**Check:** on github.com, your repo shows your new commit.

---

## MLflow cheat sheet

```python
import mlflow

# where runs go: one experiment = one folder of runs
mlflow.set_experiment("/Users/you@example.com/my-experiment")

# one run = everything inside "with"
with mlflow.start_run(run_name="my first run"):
    mlflow.log_param("max_depth", 5)                    # one setting
    mlflow.log_params({"model": "tree", "depth": 5})    # many settings
    mlflow.log_metric("val_mae", 87.3)                  # one score
    mlflow.log_metrics({"val_mae": 87.3, "val_rmse": 120.1})
    mlflow.set_tag("written_by", "me")                  # a note
    mlflow.log_figure(fig, "plot.png")                  # a matplotlib picture
    mlflow.sklearn.log_model(model, artifact_path="model",
                             serialization_format="cloudpickle")   # the trained model

# or: let MLflow record settings, training scores and the model by itself
mlflow.sklearn.autolog()

# find runs, best first (returns a pandas table)
runs = mlflow.search_runs(order_by=["metrics.val_mae ASC"])

# keep a model: register it, put a label on it
version = mlflow.register_model(f"runs:/{run_id}/model", "workspace.default.my_model")
mlflow.MlflowClient().set_registered_model_alias("workspace.default.my_model", "champion", version.version)

# use it anywhere, by label
model = mlflow.sklearn.load_model("models:/workspace.default.my_model@champion")
```

Note: we write `serialization_format="cloudpickle"` so the same code also works with newer MLflow on a laptop.

## Problems and fixes

| Problem | Fix |
|---|---|
| No **Use this template** button | You are not logged in to GitHub |
| Git folder: "repository not found" | You pasted the template URL, not your own. Or your repo is private |
| `ModuleNotFoundError: No module named 'bikes'` | You ran a file outside your Git folder, or moved files |
| `FileNotFoundError: data/bike_sharing_hourly.csv` | Same: run the file from inside your Git folder |
| `5_from_scratch.py`: an error about the experiment or a path | Check `EMAIL`: it must be exactly the email you log in to Databricks with |
| `2_register.py`: no runs or `val_mae` missing | Part 4 not finished: your runs have no `val_mae` yet |
| Run takes very long | The first start is slow. Wait 2–3 minutes, then refresh the page |
| A run you don't want | Experiments → tick it → **Delete** |

## Words

| Word | Meaning |
|---|---|
| Databricks | A website where a team keeps code, data and computers in one place |
| Serverless compute | The computer that runs your code. It starts when you press Run |
| Git folder | A copy of your GitHub repo inside Databricks |
| MLflow | A logbook for model training |
| Experiment | A folder of runs for one problem |
| Run | One training attempt |
| Parameter | A setting you chose (model type, depth) |
| Metric | A score (validation MAE) |
| Registered model | A model we keep, with versions 1, 2, 3… |
| Alias / label | A name tag on one version. `champion` = the one in use |
| MAE | Mean absolute error: the average miss, here in bikes per hour |
| Data drift | The world changes after training, so the model gets worse |

## What to read next

Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022). Very practical. Start with:
- **Chapter 1** — ML in a business vs ML in a course
- **Chapter 2** — ML work as a loop you go round many times
- **Chapter 6** — model development and offline evaluation; the section on experiment tracking is today's topic
- **Chapters 8–9** — why models get worse over time (data drift) and how teams retrain them

**Try at home:** record the runs of your regression challenge in MLflow, with the same commands.

**Want to work from your own laptop later?** Databricks has a command-line tool (Databricks CLI) and a VS Code extension.
You don't need them for this course.

---
Data: Capital Bikeshare hourly rentals, Washington DC, 2011–2012 (Fanaee-T and Gama, UCI Machine Learning Repository, CC BY 4.0). The bike-rental company in the story is invented.
