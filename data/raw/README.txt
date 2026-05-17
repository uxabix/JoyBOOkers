Source: Kaggle — goodreads-book-datasets-10m
https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m

Expected layout (full dataset, split into shards):
  - book1-100k.csv, book100k-200k.csv, … book4000k-5000k.csv
  - user_rating_0_to_1000.csv, user_rating_1000_to_2000.csv, …

Legacy single files are also supported when present:
  - book.csv
  - user-rating.csv

Ratings CSV layout (either):
  1) Matrix: user_id, book_id, rating [, timestamp]
  2) Kaggle text export: ID (user), Name (book title), Rating ("it was amazing", …)
     — titles are matched to the book catalog after strip().

Stage 1 loads every matching shard, concatenates in memory (no merged copy on disk),
then cleans and writes outputs under data/processed/.

From the repository root:
  pip install -r requirements.txt
  python scripts/stage1_prepare.py

Optional flags:
  --no-plots          skip EDA PNG export
  --books PATH        single book CSV instead of all shards
  --ratings PATH      single ratings CSV instead of all shards

Outputs: data/processed/ (tables, stage1_summary.json, eda/*.png)
