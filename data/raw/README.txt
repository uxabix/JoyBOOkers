Источник: Kaggle — goodreads-book-datasets-10m
https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m

Положите в эту папку (имена по умолчанию для скрипта):
  - book.csv
  - user-rating.csv

Файл user-rating.csv поддерживается в двух вариантах:
  1) Матрица: user_id, book_id, rating [, timestamp]
  2) Экспорт Kaggle с текстовыми оценками: ID (пользователь), Name (название книги),
     Rating («it was amazing», «really liked it», …) — сопоставление с book.csv по точному
     совпадению названия после strip().

Если файл рейтингов называется иначе (например user_rating.csv), укажите путь:
  python scripts/stage1_prepare.py --ratings data/raw/user_rating.csv

Запуск этапа 1 из корня репозитория:
  pip install -r requirements.txt
  python scripts/stage1_prepare.py

Результаты: data/processed/ (таблицы, stage1_summary.json, папка eda/*.png)
