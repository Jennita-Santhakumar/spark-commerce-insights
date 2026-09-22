
# E-Commerce Analysis: Statistical + Big Data Frameworks

## Honest history of this repo

This repo used to contain a single Jupyter notebook (`BDF_review.ipynb`) that
applied **Benford's Law**, a forensic first-digit statistical test, to
e-commerce transaction data using plain **pandas/NumPy/scipy**. Despite the
repo's name referencing "bigdata frameworks," there was no real distributed
computing anywhere — the notebook briefly imported PySpark to read a CSV, but
immediately converted the result to a pandas DataFrame with `.toPandas()`,
so no actual distributed computation happened. That was a misleading repo
name for a single-machine pandas notebook.

This repo now contains two things:

1. **`statistical_analysis/`** — the original notebook, unchanged in
   substance, relocated here and clearly labeled as the single-machine
   pandas/NumPy/scipy statistical analysis it always was.
2. **`spark/`** — a genuine **PySpark** pipeline that rebuilds and extends the
   same analysis using real distributed DataFrame operations: `groupBy`
   aggregations, joins across `campaigns.csv` / `holidays.csv` / `product.csv`,
   and a from-scratch re-implementation of the Benford's Law leading-digit
   test using Spark DataFrame/SQL functions (not pandas) for the digit
   extraction and counting step.

## Why "synthetically scaled" data

The CSVs checked into this repo are small samples (a few thousand rows). A
truly massive multi-GB dataset isn't available locally, so `spark/data_prep.py`
**synthetically scales up** `campaigns.csv` by cross-joining it against a
replication-factor range and perturbing the numeric (`total_count`) and
timestamp (`started_at`) columns per replica with random jitter. This is
explicitly a demonstration technique to give the distributed operations a
larger, more realistic row count to operate on — it is not fabricated
"real" transaction data, and every place that does this says so in code
comments. With the default `replication_factor=200` this turns ~1,900
campaign rows into ~380,000 rows.

## What the Spark pipeline does (`spark/pipeline.py`)

1. Loads `campaigns.csv`, `holidays.csv`, `product.csv` as Spark DataFrames.
2. Synthetically scales `campaigns.csv` (see above).
3. Assigns each scaled row a synthetic `product_id` and joins it against
   `product.csv` (an inner join), and joins the scaled campaigns against
   `holidays.csv` on event date (a left join) to flag holiday campaigns.
4. Runs distributed `groupBy` aggregations:
   - reach by `channel` x `campaign_type`
   - average reach on holiday vs. non-holiday campaign days
   - average reach by linked product quality score
5. Re-implements the Benford's Law leading-digit distribution using Spark
   DataFrame operations (`spark/benford.py`): first-digit extraction via
   `regexp_replace`/`substring` on a formatted numeric string, then a
   distributed `groupBy("leading_digit").count()`. Only the final 9-row
   observed/expected table is collected to the driver for the scipy
   chi-square goodness-of-fit test — the ~380k raw rows never leave Spark.
6. Writes each result table to `spark/output/<name>/` as CSV and prints a
   summary to stdout.

## How to run it

### Requirements

- **Java 17** (or any JDK 8/11/17 that your PySpark version supports) — Spark
  runs on the JVM. Check with `java -version`.
- Python 3.9+ with the packages in `spark/requirements.txt`:

  ```bash
  pip install -r spark/requirements.txt
  ```

### Run the pipeline

```bash
python spark/pipeline.py
```

or with `spark-submit` if you have a full Spark distribution installed:

```bash
spark-submit spark/pipeline.py
```

### Run the tests

```bash
cd spark
python -m pytest tests/ -v
```

Tests spin up a real local PySpark session (`local[*]`, no cluster needed)
and exercise the digit-extraction, Benford aggregation, join, and
groupBy-aggregation functions against small in-memory DataFrames.

### Windows-specific gotchas (hit and fixed while building this)

- **`Python was not found; run without arguments to install from the
  Microsoft Store`** — Spark's executor tries to spawn a Python worker
  process using the bare `python` command, which on Windows can resolve to
  the Microsoft Store app-execution-alias stub instead of a real
  interpreter, causing every task to fail with a socket timeout. Fix: set
  `PYSPARK_PYTHON` (and `PYSPARK_DRIVER_PYTHON`) to the **full path** of a
  real `python.exe` before running Spark, e.g. (PowerShell):

  ```powershell
  $env:PYSPARK_PYTHON = "C:\Python313\python.exe"
  $env:PYSPARK_DRIVER_PYTHON = "C:\Python313\python.exe"
  ```

- **`Did not find winutils.exe: ... HADOOP_HOME and hadoop.home.dir are
  unset`** — this is a well-known Windows/Hadoop-client warning that Spark
  logs on startup because it looks for the Hadoop native/`winutils.exe`
  bits used for local filesystem permission checks. It is a **warning, not
  a fatal error** for local-mode CSV read/write and DataFrame operations —
  the pipeline in this repo runs fine without it. If you hit an actual
  failure related to it (typically only when writing to certain output
  formats or using `saveAsHadoopFile`), download a Windows `winutils.exe`
  build matching your Hadoop version, put it in `C:\hadoop\bin\winutils.exe`,
  and set `HADOOP_HOME=C:\hadoop`.

- Mixed numeric literal types (e.g. `int` and `float` in the same column)
  passed to `spark.createDataFrame` without an explicit schema can raise
  `PySparkTypeError: [CANNOT_MERGE_TYPE]` during type inference — pass an
  explicit schema or keep literal types consistent per column (see
  `spark/tests/test_benford.py` for examples).

## Verified run

This pipeline was actually executed locally (Java 17 Temurin, PySpark 4.2.0,
Python 3.13, Windows 11) against the real CSVs in this repo with the default
`replication_factor=200` (1,907 campaigns scaled to 381,400 rows). All 7
tests in `spark/tests/` pass under a local PySpark session. Real output from
an actual run:

```
[pipeline] Loaded raw campaigns rows: 1907
[pipeline] Loaded product rows: 1599
[pipeline] Loaded holiday rows: 48
[pipeline] Synthetically scaled campaigns to 381400 rows (replication_factor=200)

[pipeline] Reach on holidays vs non-holidays
+----------+-----------+-----------------+
|is_holiday|n_campaigns|avg_reach        |
+----------+-----------+-----------------+
|false     |365448     |373863.8440380911|
|true      |15952      |379371.2948219659|
+----------+-----------+-----------------+

[pipeline] Benford's Law leading-digit distribution (Spark-native)
+-----+-------------------+--------------+--------------------+
|digit|expected_pct       |observed_count|observed_pct        |
+-----+-------------------+--------------+--------------------+
|1    |0.3010299956639812 |145020        |0.398693565148541   |
|2    |0.17609125905568124|53944         |0.14830454887858843 |
...
|9    |0.04575749056067514|19347         |0.05318938356729294 |
+-----+-------------------+--------------+--------------------+

[pipeline] Chi-square=22092.1720 p-value=0.000000 n=363738
```

(This synthetic, jitter-scaled dataset deviates noticeably from Benford's
law — a large chi-square/tiny p-value — which is expected, since it's a
multiplicatively-perturbed replica of a small real sample rather than an
organically-generated one; the point of this run is to demonstrate the
Spark-native distributed computation, not to claim a fraud finding.)

One Windows-only wrinkle found while getting this to run end to end:
Spark's native `DataFrame.write.csv(...)` goes through Hadoop's file
committer, which throws `FileNotFoundException: HADOOP_HOME and
hadoop.home.dir are unset` without `winutils.exe` installed (see the
gotcha list below). All the actual computation - the synthetic scale-up,
every join, every groupBy aggregation, and the Benford digit
extraction/counting - runs correctly and fully distributed under Spark
regardless; only the final tiny (a few dozen rows at most) result tables
are collected once via `toPandas()` to write plain CSV files, sidestepping
that Windows packaging gap rather than avoiding real Spark computation.
`spark/pipeline.py` documents this decision inline.

## Repo layout

```
statistical_analysis/
  BDF_review.ipynb        # original pandas/NumPy/scipy Benford's Law notebook
spark/
  data_prep.py             # Spark session, CSV loading, synthetic scale-up
  benford.py                # Benford's Law re-implemented with Spark DataFrame ops
  transformations.py        # groupBy aggregations + joins across campaigns/holidays/product
  pipeline.py                # main entry point, ties it all together
  requirements.txt
  tests/                      # pytest suite using a local PySpark session
  output/                      # CSV results written by a pipeline run (gitignored contents)
Dataset.csv, campaigns.csv, holidays.csv, product.csv   # shared source data
```

## Use cases

* Fraud detection in accounting, healthcare billing, tax records
* Audit analytics at a scale beyond what fits comfortably in pandas
* Academic demonstration of forensic analytics and distributed data
  processing with Spark

## Sustainable Development Goals (SDG)

This work aligns with **SDG 16**: *Peace, Justice, and Strong Institutions* —
specifically in enhancing transparency and combating corruption through data
science, forensic analytics, and scalable data processing.

## Author

* **Kabilesh Rajaselvan**
  Vellore Institute of Technology, Chennai Campus
  [LinkedIn](https://www.linkedin.com/in/kabilesh-rajaselvan-3344a7287/)
