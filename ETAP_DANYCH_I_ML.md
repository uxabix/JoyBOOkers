# JoyBookers — etap przygotowania danych i uczenia modeli

Dokument opisuje **bieżący etap** projektu uniwersyteckiego *Book Recommendation System*: skąd pobrać dane, gdzie je umieścić, jakie polecenia uruchomić oraz **szczegółowe wyniki** ostatniego przebiegu pipeline’u.

> Raporty JSON i wykresy EDA są wersjonowane w katalogu [`reports/`](reports/). Duże pliki binarne (`.pkl`, `.parquet`, `.npz`) pozostają w `data/processed/` i **nie trafiają do git**.

**Źródła metryk:** sekcja **§9** (pełny katalog plików analitycznych). Główne agregaty: `reports/ml/evaluation/evaluate_all.json`, `reports/features/features_summary.json`, `reports/data_pipeline/preprocess_summary.json`.

**Version control:** JSON reports live in [`reports/`](reports/) (git-tracked). Binary models stay in `data/processed/` (gitignored). Refresh and commit with `python scripts/export_reports.py` then `git add reports/`.

---

## 1. Cel etapu i status końcowy

| Moduł | Zbiór danych | Algorytm | Status |
|-------|----------------|----------|--------|
| Filtrowanie kolaboracyjne | DS1 (Goodreads 2M) | Surprise SVD | **Gotowe** |
| Segmentacja użytkowników | DS1 | K-Means | **Gotowe** |
| Analiza sentymentu | DS4 (Amazon Reviews) | TF-IDF + regresja logistyczna | **Gotowe** |
| Rekomendacje treściowe | DS2 + DS3 | TF-IDF + podobieństwo cosinus | **Gotowe** |

**Wniosek:** wszystkie cztery moduły ML są wytrenowane. API (`uvicorn main:app`) może korzystać z modeli w `data/processed/models/`.

---

## 2. Wymagania środowiska

| Wymaganie | Szczegóły |
|-----------|-----------|
| Python | 3.12+ |
| RAM | 8–16 GB (DS4: plik recenzji ~2,9 GB) |
| Dysk | kilka GB na surowe CSV + modele lokalnie |
| Zależności | `pip install -r requirements.txt` |

```powershell
cd ścieżka\do\JoyBookers
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 3. Pobieranie zbiorów danych (Kaggle)

| ID | Zbiór Kaggle | Rola |
|----|----------------|------|
| **DS1** | *Goodreads Book Datasets With User Rating 2M* | Oceny → SVD, K-Means |
| **DS2** | *Goodreads 100k Books* | Gatunki, metadane → TF-IDF |
| **DS3** | *Goodreads Best Books* / *Best Books Ever* | Postacie, tagi (wzbogacenie) |
| **DS4** | *Amazon Books Reviews* | Recenzje → sentyment (bez łączenia z Goodreads) |

Dużych CSV **nie commituj** — tylko raporty w `reports/`.

---

## 4. Gdzie umieścić pliki

```
data/raw/
├── book*.csv                 # DS1 — katalog (legacy: root data/raw/)
├── user_rating_*.csv         # DS1 — oceny
├── ds2_goodreads_100k/
│   └── GoodReads_100k_books.csv
├── ds3_goodreads_best/
│   └── books_1.Best_Books_Ever.csv
└── ds4_amazon_reviews/
    ├── Books_rating.csv      # wymagane (kolumna review/text)
    └── books_data.csv        # opcjonalne; pipeline ignoruje przy NLP
```

**Ważne:**
- DS2/DS3 muszą być w **podfolderach** — inaczej `preprocess` je pominie.
- DS4: używany jest wyłącznie plik z tekstem recenzji (`Books_rating.csv`).
- Przy małej RAM: `python scripts/run_data_pipeline.py --stages preprocess --ds4-sample 100000`

---

## 5. Uruchomienie pipeline’u

### Pełny przebieg (pierwszy raz)

```powershell
python scripts/run_data_pipeline.py --stages all
python scripts/ml/run_ml_pipeline.py --stages all
python scripts/export_reports.py
```

### Tylko wybrany moduł (bez ponownego czekania na całość)

| Cel | Polecenia |
|-----|-----------|
| Tylko cechy treściowe (DS2+DS3) | `python scripts/run_data_pipeline.py --stages features` |
| Tylko uczenie content | `python scripts/ml/run_ml_pipeline.py --module content --stages train evaluate` |
| Tylko ewaluacja wszystkich modeli | `python scripts/ml/evaluate_all.py` |
| Aktualizacja raportów w git | `python scripts/export_reports.py` |

### API

```powershell
uvicorn main:app --reload
```

- WWW: http://127.0.0.1:8000  
- OpenAPI: http://127.0.0.1:8000/docs  

---

## 6. Szczegółowe wyniki — przygotowanie danych

### 6.1 DS1 (Goodreads 2M)

| Metryka | Wartość | Komentarz |
|---------|---------|-----------|
| Książki surowe → czyste | 1 850 310 → 1 548 594 | Usunięto duplikaty i błędne ID |
| Oceny surowe → czyste | 362 596 → **235 484** | ~35% strat |
| Odrzucone (brak dopasowania tytułu) | **121 329** | Eksport Kaggle: oceny po tytule, nie po ID |
| Użytkownicy | 3 980 | |
| Książki z ≥1 oceną | 48 920 | |
| Gęstość macierzy user–item | **0,12%** | Typowa rzadkość dla CF |
| Średnia ocena | **3,80** / 5 | |
| Książki z tylko 1 oceną | 28 592 | Cold start |
| Użytkownicy z 1 oceną | 454 | Pominięci w rankingu Precision@K |

**Wykresy EDA** (w `reports/eda/`): rozkład ocen, top książki/użytkownicy, języki, wydawcy, lata publikacji.

### 6.2 DS2 (Goodreads 100k)

| Metryka | Wartość |
|---------|---------|
| Wiersze surowe | 100 000 |
| Wiersze po czyszczeniu | 99 999 |
| Wiersze z gatunkami | 89 532 |
| Unikalni autorzy (znormalizowani) | 68 513 |

### 6.3 DS3 (Best Books Ever)

| Metryka | Wartość |
|---------|---------|
| Wiersze po czyszczeniu | 52 424 |
| Wiersze z postaciami | 13 758 |
| Wiersze z tagami | **0** | W tym pliku CSV kolumna tagów jest pusta |

### 6.4 DS4 (Amazon Reviews)

| Etap | Wiersze |
|------|---------|
| Surowe (`Books_rating.csv`) | 3 000 000 |
| Po filtracji długości i usunięciu neutralnych 3★ | 96 460* |
| W pipeline NLP / splittach (pełny przebieg) | **1 887 091** |

\*Różne etapy pipeline’u; model sentymentu trenowany na podziałach z ~1,51 mln wierszy train.

**Etykiety sentymentu:** ocena ≤2★ → negatywny (0), ≥4★ → pozytywny (1); oceny 3★ odrzucone.

### 6.5 Katalog treściowy DS2 + DS3 (cechy)

| Metryka | Wartość |
|---------|---------|
| Łącznie książek w katalogu | **149 342** |
| Z DS2 | 99 999 |
| Tylko z DS3 (bez match_key w DS2) | 49 343 |
| Wzbogacone DS3→DS2 (`match_key`) | **3 095** | Postacie / metadane DS3 dopisane do wierszy DS2 |
| Macierz BoW (features) | 5 000 słów, 5,5 mln nnz |
| Gatunki (multi-hot) | 200 cech |
| Tagi w DS3 CSV | **0** | Kolumna tagów pusta w `books_1.Best_Books_Ever.csv` |

---

## 7. Szczegółowe wyniki — modele ML

### 7.1 Filtrowanie kolaboracyjne (Surprise SVD)

**Dane treningowe:** 189 404 interakcje (train split), 3 980 użytkowników, 42 953 książek.  
**Test:** 46 080 interakcji (hold-out 20% ocen per user).

**Hiperparametry:**

| Parametr | Wartość |
|----------|---------|
| `n_factors` | 100 |
| `n_epochs` | 20 |
| `lr_all` | 0.005 |
| `reg_all` | 0.02 |
| `random_state` | 42 |

**Metryki regresji (przewidywanie oceny 1–5):**

| Zbiór | RMSE | MAE |
|-------|------|-----|
| Walidacja (10% wewnętrzny split) | **0,882** | **0,700** |
| Test | **0,890** | **0,706** |

**Interpretacja:**
- Walidacja ≈ test → **brak przeuczenia**.
- RMSE ~0,89 przy średniej oceny 3,8 to wyraźnie lepiej niż naiwny baseline (stała średnia → RMSE ~1,0+).
- Model nadaje się do endpointu `/api/v1/recommendations/for-user`.

**Metryki rankingu (trudniejsze):**

| Metryka | Wartość |
|---------|---------|
| Precision@10 | 0,56% |
| Recall@10 | 0,66% |

Niskie wartości wynikają z ~43k książek, cold start i rzadkiej macierzy — w pracy opisz RMSE/MAE jako główne metryki SVD.

**Artefakt:** `data/processed/models/collaborative/svd_model.pkl` (~41 MB)

---

### 7.2 Segmentacja użytkowników (K-Means, k = 3)

**Wejście:** 3 311 użytkowników z ≥3 ocenami (669 odrzuconych).  
**Cechy:** liczba ocen, średnia, odchylenie std, rozstęp ocen, poziom aktywności (low/medium/high).

| Metryka | Wartość |
|---------|---------|
| Silhouette | **0,47** |
| Inertia | 10 552,5 |
| Wybrane k | 3 (najlepsze silhouette w zakresie 3–12) |

**Profile klastrów:**

| ID | Użytkownicy | Śr. liczba ocen | Śr. ocena | Śr. std | Opis |
|----|-------------|-----------------|-----------|---------|------|
| **0** | 1 378 (42%) | 148,6 | 3,82 | 0,92 | „Power users” — bardzo aktywni, krytyczniejsi |
| **1** | 611 (18%) | 6,0 | 4,14 | 0,68 | „Łagodni rzadcy” — mało ocen, wysokie średnie |
| **2** | 1 322 (40%) | 19,8 | 4,02 | 0,84 | „Średnia aktywność” |

**Artefakty:** `kmeans_model.joblib`, `user_cluster_assignments.parquet`

---

### 7.3 Rekomendacje treściowe (TF-IDF + cosinus)

**Katalog:** 149 342 książek (DS2 + unikalne wiersze DS3 + 3 095 wzbogaconych match_key). **DS1 wykluczony** (1,5M+ książek).

**Macierze rzadkie (CSR), L2-normalizacja wierszy:**

| Blok cech | Liczba cech | Uwagi |
|-----------|-------------|-------|
| Autorzy | 3 000 | TF-IDF na polu `authors` |
| Gatunki | 500 | TF-IDF na liście gatunków |
| Tagi | **0** | Brak użytecznych tagów w DS3 CSV |
| Treść (tytuł+opis+gatunki+postacie) | 5 000 | TF-IDF na `content_text` |
| **Łącznie (combined)** | **8 500** | `tfidf_combined.npz`, 6,7 mln nnz |

**Ewaluacja ( próbka 500 losowych książek, Top-10 sąsiadów):**

| Metryka | Wartość | Znaczenie |
|---------|---------|-----------|
| Średnie pokrycie gatunków @10 | **71,4%** | Sąsiedzi często dzielą gatunki z książką zapytania |
| Średnie podobieństwo cosinus | **0,633** | Umiarkowanie wysoka spójność wektorów |

**Interpretacja:**
- Model **działa** — podobne książki są semantycznie bliskie (gatunki się zgadzają).
- Brak cech tagów to ograniczenie danych DS3, nie błąd pipeline’u.
- Endpointy: `/api/v1/books/{id}/similar`, strona `/books/{id}`.

**Artefakty:** `data/processed/models/content/tfidf_combined.npz`, `tfidf_vectorizers.joblib`

---

### 7.4 Analiza sentymentu (TF-IDF + regresja logistyczna)

**Dane:** niezależny korpus Amazon (bez łączenia z Goodreads).

**Podział stratified:**

| Zbiór | Wiersze | Udział |
|-------|---------|--------|
| Train | 1 509 675 | ~80% |
| Validation | 188 708 | ~10% |
| Test | 188 708 | ~10% |

**Hiperparametry:** `max_features=20000`, `ngram_range=(1,2)`, `C=1.0`, `class_weight=balanced`.

**Metryki:**

| Metryka | Validation | Test |
|---------|------------|------|
| Accuracy | 91,80% | 91,78% |
| F1 macro | 0,850 | 0,850 |
| Precision macro | 0,812 | 0,811 |
| Recall macro | 0,914 | 0,915 |

**Raport per klasa (test):**

| Klasa | Precision | Recall | F1 | Próbki |
|-------|-----------|--------|-----|--------|
| 0 negatywny | 0,64 | 0,91 | 0,75 | 25 550 |
| 1 pozytywny | 0,99 | 0,92 | 0,95 | 163 158 |

**Interpretacja:**
- **Nierównowaga klas** (~87% pozytywnych) — accuracy 92% nie oznacza idealnego modelu.
- Negatywy: niższa precyzja (0,64) — więcej fałszywych alarmów „negatywny”.
- Pozytywy: bardzo wysoka precyzja (0,99).
- Val ≈ test → model **stabilny** i gotowy do produkcji akademickiej.

**Artefakt:** `data/processed/models/sentiment/sentiment_pipeline.joblib`

---

## 8. Ograniczenia i uwagi do pracy dyplomowej

1. **DS1 — dopasowanie tytułów:** 33% ocen utraconych przez brak mapowania tytuł→ID katalogu.
2. **DS3 — brak tagów** w pliku `books_1.Best_Books_Ever.csv`; wzbogacenie `match_key` działa częściowo (**3 095** wierszy DS2), reszta DS3 to osobne wiersze katalogu (**49 343**).
3. **CF Precision@K** — niska z definicji zadania; skup się na RMSE/MAE.
4. **Sentyment** — wysoka accuracy częściowo dzięki dominacji klasy pozytywnej; podawaj F1 macro i macierz per klasa.
5. **Modele binarne** nie są w git — na innym komputerze trzeba ponownie uruchomić ML lub skopiować `data/processed/models/`.

---

## 9. Analysis source files (catalog)

This section lists **every JSON/PNG report** used to write sections §6–§8 of this document. Each file exists in two places:

| Location | Git | Role |
|----------|-----|------|
| **`reports/`** | Yes — commit to share results | Human-readable snapshots for thesis / review |
| **`data/processed/`** | No — local only | Live pipeline output; same JSON before export |

Refresh `reports/` from disk:

```powershell
python scripts/export_reports.py
```

`export_reports` also rebuilds **`evaluate_all.json`** and **`train_all.json`** from per-module `evaluate_report.json` / `train_report.json` files.

---

### 9.1 Top-level index

| File in `reports/` | Mirror in `data/processed/` | What it contains | Used in doc |
|--------------------|-----------------------------|------------------|-------------|
| `manifest.json` | — (written by export only) | List of all 39 exported files; `aggregated_refreshed` shows rebuilt summaries | Export audit |
| `data_pipeline/pipeline_summary.json` | `analysis/pipeline_summary.json` | Stages run, nested copies of analyze/features/splits summaries | Pipeline status |
| `features/features_summary.json` | `features/features_summary.json` | **Rollup:** project scope, interactions, clustering, content catalog, NLP corpus counts | §1, §6.1, §6.5 |
| `ml/evaluation/evaluate_all.json` | `models/evaluation/evaluate_all.json` | **Single-file metrics for all 4 ML modules** (RMSE, silhouette, genre overlap, sentiment F1) | §7, §10 |
| `ml/evaluation/train_all.json` | `models/evaluation/train_all.json` | Merged training reports (hyperparameters, validation metrics) | §7 |
| `ml/evaluation/ml_pipeline_summary.json` | `models/evaluation/ml_pipeline_summary.json` | Last `run_ml_pipeline` call (may be one module only) | Debugging partial runs |

---

### 9.2 Data pipeline — cleaning & EDA

| File in `reports/` | Mirror in `data/processed/` | What it contains | Used in doc |
|--------------------|-----------------------------|------------------|-------------|
| `data_pipeline/preprocess_summary.json` | `analysis/preprocess_summary.json` | Per-dataset cleaning: DS1 book/rating row counts, title-match losses (121 329), DS2/DS3 row counts, DS4 skip/sample flags | §6.1–§6.4 |
| `data_pipeline/analyze_all.json` | `analysis/analyze_all.json` | Project scope, raw schemas, column mappings, preprocessing notes per dataset | §3–§4, design |
| `eda/01_rating_distribution.png` | `eda/01_rating_distribution.png` | Histogram of user ratings (DS1) | §6.1 |
| `eda/02_top_books.png` | `eda/02_top_books.png` | Most-rated books | §6.1 |
| `eda/03_top_users.png` | `eda/03_top_users.png` | Most active users | §6.1 |
| `eda/04_languages.png` | `eda/04_languages.png` | Language distribution in catalog | §6.1 |
| `eda/05_books_per_year.png` | `eda/05_books_per_year.png` | Publication year distribution | §6.1 |
| `eda/06_top_publishers.png` | `eda/06_top_publishers.png` | Top publishers | §6.1 |
| `eda/07_avg_rating_books_hist.png` | `eda/07_avg_rating_books_hist.png` | Catalog average-rating histogram | §6.1 |

> **Note:** If `preprocess_summary.json` shows `"ds2": {"skipped": "no raw files"}`, DS2 stats in §6.2 come from `features_summary.json` / `content_features_report.json` after a successful features run.

---

### 9.3 Feature engineering reports

| File in `reports/` | Mirror in `data/processed/` | What it contains | Used in doc |
|--------------------|-----------------------------|------------------|-------------|
| `features/interactions/interactions_features_report.json` | `features/interactions/interactions_features_report.json` | CF matrix: 235 484 interactions, 3 980 users, 48 920 books, density 0.12%, mean rating 3.80; paths to parquet indexes | §6.1, §7.1 |
| `features/clustering/clustering_features_report.json` | `features/clustering/clustering_features_report.json` | 3 311 users (669 dropped), feature column names, activity quantiles | §7.2 |
| `features/content/content_features_report.json` | `features/content/content_features_report.json` | Content catalog merge: **149 342** books, ds2/ds3_enriched/ds3_only counts, BoW 5 000 vocab, 200 genres | §6.5, §7.3 |
| `features/content/vocabulary.json` | `features/content/vocabulary.json` | BoW token → index map (5 000 terms) | Feature audit |
| `features/content/genre_labels.json` | `features/content/genre_labels.json` | Genre label → index map (200 genres) | Feature audit |
| `features/nlp/nlp_corpus_report.json` | `features/nlp/nlp_corpus_report.json` | DS4 NLP corpus: 96 460 reviews after filtering, class balance, word-count stats | §6.4 |
| `features/nlp/nlp_vocabulary.json` | `features/nlp/nlp_vocabulary.json` | NLP corpus vocabulary (15 000 terms) | Feature audit |

---

### 9.4 Train / test splits

| File in `reports/` | Mirror in `data/processed/` | What it contains | Used in doc |
|--------------------|-----------------------------|------------------|-------------|
| `splits/splits_report.json` | `splits/splits_report.json` | CF: 189 404 train / 46 080 test rows; NLP: 1 509 675 / 188 708 / 188 708 stratified splits; parquet paths | §7.1, §7.4 |

> **Known inconsistency:** `nlp_corpus_report.json` may show ~96 k filtered reviews while `splits_report.json` NLP splits can still reflect an older ~1.9 M run. Check both before retraining sentiment.

---

### 9.5 ML training reports (per module)

| File in `reports/` | Mirror in `data/processed/` | What it contains | Used in doc |
|--------------------|-----------------------------|------------------|-------------|
| `ml/collaborative/preprocess_report.json` | `models/collaborative/preprocess_report.json` | CF training matrix shape, Surprise dataset stats | §7.1 |
| `ml/collaborative/train_report.json` | `models/collaborative/train_report.json` | SVD hyperparameters (`n_factors=100`, …), validation RMSE **0.882**, MAE **0.700** | §7.1 |
| `ml/clustering/preprocess_report.json` | `models/clustering/preprocess_report.json` | Clustering input feature matrix metadata | §7.2 |
| `ml/clustering/train_report.json` | `models/clustering/train_report.json` | K=3, silhouette **0.47**, inertia, cluster sizes {1378, 611, 1322}, silhouette_by_k sweep | §7.2 |
| `ml/content/preprocess_report.json` | `models/content/preprocess_report.json` | TF-IDF preprocessing steps, catalog row count | §7.3 |
| `ml/content/train_report.json` | `models/content/train_report.json` | TF-IDF blocks: authors 3 000, genres 500, tags **0**, content 5 000, combined **8 500** features, 6.7 M nnz | §7.3 |
| `ml/content/tfidf_*.meta.json` | `models/content/tfidf_*.meta.json` | Block name tags (`authors`, `genres`, `tags`, `content`, `combined`) | §7.3 |
| `ml/sentiment/preprocess_report.json` | `models/sentiment/preprocess_report.json` | Sentiment vectorizer / label encoding metadata | §7.4 |
| `ml/sentiment/train_report.json` | `models/sentiment/train_report.json` | LogReg hyperparameters, validation accuracy **91.8%**, F1 macro **0.85**, classification report | §7.4 |

---

### 9.6 ML evaluation reports (per module)

| File in `reports/` | Mirror in `data/processed/` | What it contains | Used in doc |
|--------------------|-----------------------------|------------------|-------------|
| `ml/evaluation/collaborative/evaluate_report.json` | `models/evaluation/collaborative/evaluate_report.json` | Test RMSE **0.890**, MAE **0.706**, Precision@10 **0.56%**, Recall@10 **0.66%** | §7.1 |
| `ml/evaluation/clustering/evaluate_report.json` | `models/evaluation/clustering/evaluate_report.json` | Cluster profile means (ratings count, mean rating, std per cluster) | §7.2 |
| `ml/evaluation/content/evaluate_report.json` | `models/evaluation/content/evaluate_report.json` | Genre overlap@10 **71.4%**, mean cosine **0.633**, n_books **149 342** | §7.3 |
| `ml/evaluation/sentiment/evaluate_report.json` | `models/evaluation/sentiment/evaluate_report.json` | Test accuracy **91.8%**, F1 macro **0.85**, per-class precision/recall | §7.4 |

**Aggregated (start here):** `ml/evaluation/evaluate_all.json` — merges the four rows above into one JSON.

---

### 9.7 Binary artifacts (not in git)

These paths appear inside JSON reports but are **not** exported to `reports/`:

| Path | Contents |
|------|----------|
| `data/processed/models/collaborative/svd_model.pkl` | Trained Surprise SVD model (~41 MB) |
| `data/processed/models/clustering/kmeans_model.joblib` | Fitted K-Means |
| `data/processed/models/clustering/user_cluster_assignments.parquet` | User → cluster id |
| `data/processed/models/content/tfidf_combined.npz` | L2-normalized sparse TF-IDF matrix |
| `data/processed/models/content/tfidf_vectorizers.joblib` | Fitted vectorizers per block |
| `data/processed/models/sentiment/sentiment_pipeline.joblib` | TF-IDF + LogisticRegression pipeline |
| `data/processed/features/**/*.parquet`, `*.npz` | Intermediate feature tables |

Copy `data/processed/models/` manually or re-run `python scripts/ml/run_ml_pipeline.py --stages all` on another machine.

---

### 9.8 Recommended reading order

1. `reports/ml/evaluation/evaluate_all.json` — all ML metrics in one file  
2. `reports/features/features_summary.json` — all data-feature counts in one file  
3. `reports/data_pipeline/preprocess_summary.json` — raw → clean row counts  
4. Per-module `train_report.json` + `evaluate_report.json` under `reports/ml/` — details for thesis tables  
5. `reports/eda/*.png` — figures for DS1 exploratory analysis  

---

## 10. Podsumowanie — gotowość do następnego etapu

| Komponent | API | Raport / praca |
|-----------|-----|----------------|
| SVD — rekomendacje personalizowane | Tak | RMSE 0,89, MAE 0,71 |
| K-Means — segmenty użytkowników | Tak | 3 klastry, silhouette 0,47 |
| TF-IDF — podobne książki | Tak | overlap gatunków 71%, cosinus 0,63 |
| Sentyment Amazon | Tak | accuracy 92%, F1 macro 0,85 |

**Następny etap projektu:** integracja z bazą SQLite (`data/joybookers.db`), wypełnienie katalogu książek z DS2, testy end-to-end API, rozdział wyników w pracy dyplomowej.

---

## 11. Szybka ściąga poleceń

```powershell
pip install -r requirements.txt

# Pełny pipeline (pierwszy raz)
python scripts/run_data_pipeline.py --stages all
python scripts/ml/run_ml_pipeline.py --stages all
python scripts/export_reports.py

# Tylko content (ponowne uruchomienie)
python scripts/run_data_pipeline.py --stages features
python scripts/ml/run_ml_pipeline.py --module content --stages train evaluate
python scripts/export_reports.py

# API
uvicorn main:app --reload
```

---

*Last updated from `reports/ml/evaluation/evaluate_all.json`, `reports/features/features_summary.json`, and related files in `reports/` (see §9).*
