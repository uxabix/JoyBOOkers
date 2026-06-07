# System rekomendacji JoyBookers — pełny raport techniczny

**Wersja:** unified hybrid + Ridge weights + genre priors + explanations (aktualne na 2026)  
**Główny orchestrator:** `app/services/recommendation_service.py`  
**Powiązane raporty:** `ETAP_DANYCH_I_ML.md`, `data/processed/models/evaluation/hybrid/baseline_comparison.json`

---

## Spis treści

1. [Architektura ogólna](#1-architektura-ogólna)
2. [Zbiory danych DS1–DS4](#2-zbiory-danych-ds1ds4)
3. [Artefakty ML i ścieżki](#3-artefakty-ml-i-ścieżki)
4. [Typy użytkowników](#4-typy-użytkowników)
5. [Pełny pipeline rekomendacji](#5-pełny-pipeline-rekomendacji)
6. [Pięć sygnałów: ML vs heurystyki](#6-pięć-sygnałów-ml-vs-heurystyki)
7. [Wagi: manual i learned (Ridge)](#7-wagi-manual-i-learned-ridge)
8. [Cold start i genre priors](#8-cold-start-i-genre-priors)
9. [Wyjaśnienia (explanations)](#9-wyjaśnienia-explanations)
10. [Punkty wejścia (UI / API)](#10-punkty-wejścia-ui--api)
11. [Podobne książki (osobna ścieżka)](#11-podobne-książki-osobna-ścieżka)
12. [Skrypty przygotowania i uczenia](#12-skrypty-przygotowania-i-uczenia)
13. [Mapa plików](#13-mapa-plików)
14. [Przykłady według typów użytkowników](#14-przykłady-według-typów-użytkowników)
15. [Sformułowanie na obronę](#15-sformułowanie-na-obronę)

---

## 1. Architektura ogólna

System to **ujednolicony hybrydowy recommender** dla wszystkich użytkowników. W kodzie scoringu nie ma osobnych gałęzi „dla DS1” i „dla zarejestrowanych” — jest **jeden pipeline**, ale **różna dostępność sygnałów** (kluczowe: CF tylko dla użytkowników DS1 obecnych w zbiorze treningowym SVD).

```
Żądanie HTTP (/me, /recommendations, API)
        │
        ▼
app/dependencies.py → get_recommendation_service()
        │
        ▼
app/services/recommendation_service.py :: recommend_for_user()
        │
        ├─► UserProfileBuilder.build()          app/ml/user_profile.py
        │
        ├─► HybridScoringEngine.recommend()     app/ml/hybrid_scoring.py
        │       ├─ zbieranie kandydatów
        │       ├─ obliczenie 5 sygnałów (raw + znormalizowane)
        │       ├─ wynik końcowy (Ridge LUB manual blend)
        │       └─ explanations
        │
        └─► zapis + RecommendationResponse       app/schemas/recommendation.py
```

**Ładowanie modeli przy starcie:** `app/ml/registry.py` (`MLModelRegistry.load_all()`), ścieżki z `app/config.py`.

---

## 2. Zbiory danych DS1–DS4

### DS1 — Goodreads 2M (interakcje)

| Rola | Co dostarcza |
|------|--------------|
| **Collaborative Filtering** | Uczenie Surprise SVD |
| **User clustering** | Cechy behawioralne + K-Means |
| **Cluster affinity** | Jakie książki są popularne w każdym klastrze |
| **Użytkownicy/oceny w runtime** | SQLite po `scripts/load_db.py` |

**Pipeline (offline):** `bookrec/pipeline/runner.py`  
**Interakcje:** `data/processed/features/interactions/interactions_indexed.parquet`  
**Podział treningowy SVD:** `data/processed/splits/cf_train.parquet`  
**Model SVD:** `data/processed/models/collaborative/svd_model.pkl`  
**K-Means:** `data/processed/models/clustering/kmeans_model.joblib`  
**Przypisania klastrów:** `data/processed/models/clustering/user_cluster_assignments.parquet`

**W aplikacji:** użytkownicy DS1 mają `users.is_registered = False`, `external_id` = ID z Goodreads (zgodne z train SVD).

---

### DS2 — Goodreads 100k (katalog)

| Rola | Co dostarcza |
|------|--------------|
| **Główny katalog książek** | SQLite `books` |
| **Content-based** | Tekst do TF-IDF (tytuł, autor, opis…) |
| **Gatunki** | Pole `books.genre` (z katalogu) |

**Pipeline:** `bookrec/ingest/ds2_goodreads_100k.py`, etap feature  
**Katalog:** `data/processed/features/content/content_catalog.parquet`  
**Macierz TF-IDF:** `data/processed/models/content/tfidf_combined.npz` (+ `book_ids.npy`)

---

### DS3 — Goodreads Best (wzbogacenie)

| Rola | Co dostarcza |
|------|--------------|
| **Rozszerzenie treści** | Tagi, postacie, dodatkowe gatunki |
| **TF-IDF** | Dodatkowe cechy w macierzy combined |
| **SQLite** | `book_enrichments` |

**Powiązanie:** książki DS2 są wzbogacane przez `bookrec/resolution/book_linker.py` → enrichment w SQLite.

---

### DS4 — Amazon Reviews

| Rola w rekomendacjach | **Nie jest używany** |
|-----------------------|---------------------|
| Używany do | Analizy sentymentu (`/sentiment`, `app/ml/sentiment.py`) |

---

### SQLite (warstwa runtime)

| Tabela | Źródło | Rola w rekomendacjach |
|--------|--------|------------------------|
| `users` | DS1 + `/register` | Profil, `external_id`, `cluster_id`, `is_registered` |
| `books` | DS2 (+ stuby DS1) | Katalog, `source_book_id`, gatunki, `rating_count` |
| `ratings` | DS1 + oceny zarejestrowanych | Historia użytkownika |
| `recommendations` | Wyniki systemu | Audit / historia |
| `book_enrichments` | DS3 | UI, nie bezpośrednio w hybrid score |

**Ładowanie:** `scripts/load_db.py` ← parquet katalogu + interakcje.

---

## 3. Artefakty ML i ścieżki

Wszystkie ścieżki w `app/config.py`:

| Parametr | Ścieżka | Plik kodu |
|----------|---------|-----------|
| `cf_model_path` | `data/processed/models/collaborative/svd_model.pkl` | `app/ml/collaborative.py` |
| `cf_train_path` | `data/processed/splits/cf_train.parquet` | `app/ml/collaborative.py` |
| `content_tfidf_path` | `data/processed/models/content/tfidf_combined.npz` | `app/ml/content_based.py` |
| `clustering_model_path` | `data/processed/models/clustering/kmeans_model.joblib` | `app/ml/user_clustering.py` |
| `cluster_affinity_path` | `data/processed/features/clustering/cluster_affinity.json` | `app/ml/cluster_affinity.py` |
| `genre_priors_path` | `data/processed/features/clustering/genre_priors.json` | `app/ml/genre_priors.py` |
| `hybrid_weights_path` | `data/processed/models/hybrid/ridge_weights.joblib` | `app/ml/hybrid_weights.py` |

**Rejestr:** `app/ml/registry.py` — ładuje wszystkie silniki przy `ml_eager_load=True`.

---

## 4. Typy użytkowników

### 4.1. Użytkownicy zbioru danych (DS1)

- `is_registered = False`
- `external_id` — numeryczne ID Goodreads
- Oceny z `load_db.py`
- **CF dostępny** przy ≥3 ocenach i obecności w `cf_train.parquet`
- UI: `/recommendations`, `/users/{id}`

### 4.2. Zarejestrowani

- `is_registered = True`, `external_id = reg:xxxxxxxx`
- Oceny przez `/me`, API `/api/v1/ratings/me`
- **CF zawsze niedostępny** (brak w train SVD)
- Scoring: **manual blend** (content + genre + cluster + pop)
- UI: `/me`

**Sprawdzenie CF:** `UserProfileBuilder.build()` w `app/ml/user_profile.py`:

```python
cf_available = (
    not user.is_registered
    and n_ratings >= min_cf_ratings_per_user  # 3
    and external_id in cf_engine.known_user_ids()
)
```

---

## 5. Pełny pipeline rekomendacji

### Krok 0 — Punkt wejścia

`RecommendationService.recommend_for_user(user_id, limit, algorithm)`  
Plik: `app/services/recommendation_service.py`

| `algorithm` | Zachowanie |
|-------------|------------|
| `auto`, `hybrid` | `HybridScoringEngine.recommend()` |
| `collaborative` | Tylko SVD (`recommend_cf_only`) |
| `content` | Tylko TF-IDF (`recommend_content_only`) |

---

### Krok 1 — Budowa profilu

**Plik:** `app/ml/user_profile.py` → `UserProfileBuilder.build()`

| Pole | Jak obliczane |
|------|---------------|
| `rated_books` | Do 500 ocen z `RatingRepository` + `BookRepository` |
| `cluster_id` | `UserClusteringEngine.predict_cluster()` — `app/ml/user_clustering.py` |
| `cluster_label` | Etykiety 0/1/2 («Power users», «Lenient occasional», «Moderate») |
| `genre_weights` | Z ocenionych książek LUB genre priors przy 0 ocenach |
| `genre_prior_active` | `True` jeśli gatunki z priors, nie z ocen |
| `profile_strength` | `min(1.0, n_ratings / 10)` |
| `cf_available` | Patrz §4 |
| `content_vector` | Opcjonalnie; zwykle budowany w hybrid scoring |

---

### Krok 2 — Zbieranie kandydatów

**Plik:** `app/ml/hybrid_scoring.py` → `_gather_candidates()`  
Limit: `hybrid_candidate_limit = 2500` (`app/config.py`)

Połączenie (union) ze źródeł:

| # | Źródło | Warunek | Kod |
|---|--------|---------|-----|
| 1 | CF top | `cf_available` | `CollaborativeFilteringService.recommend_for_user()` |
| 2 | Sąsiedzi content | Do 5 ocenionych książek | `ContentRecommendationEngine.similar_books()` |
| 3 | Cluster top | Zawsze | `ClusterAffinityStore.top_books(cluster_id)` |
| 4 | Popularne | Zawsze | `BookRepository.list_starter_books()` |

Już ocenione książki są wykluczane.

---

### Krok 3 — Pięć sygnałów (raw)

Dla każdego kandydata `source_book_id`:

| Sygnał | Raw score | Plik / metoda |
|--------|-----------|---------------|
| **cf** | SVD predict 1–5 | `hybrid_scoring._cf_scores()` → `CollaborativeFilteringEngine` |
| **content** | Cosine TF-IDF | `content_based.score_candidates()` / `build_user_vector()` |
| **cluster** | Affinity 0–1 | `cluster_affinity.score(cluster_id, book_id)` |
| **pop** | `books.rating_count` | `hybrid_scoring._popularity_raw()` |
| **genre** | Nakładanie z `genre_weights` | `user_profile.genre_match_score()` |

---

### Krok 4 — Normalizacja dla UI (breakdown)

**Plik:** `hybrid_scoring._normalize_dict()` — min-max **w puli kandydatów** danego żądania.

Wyświetlane w UI jako **C / F / K / P / G** (content, CF, cluster, popularity, genre).

> **Ważne:** breakdown w UI to wartości względne do porównania kandydatów.  
> Dla Ridge używane są cechy **raw** (`_ridge_features()`), nie breakdown.

---

### Krok 5 — Wynik końcowy (score)

**Plik:** `app/ml/hybrid_scoring.py`, linie 106–124

#### Wariant A — Learned (Ridge)

**Warunek:** model załadowany **I** `profile.cf_available == True`

```text
ridge_score = Ridge.predict([cf, content, cluster, pop, genre])  # cechy raw
final = ridge_score  jeśli ridge_score > 0
        w przeciwnym razie manual_score (fallback)
```

**Pliki:** `app/ml/hybrid_weights.py`, uczenie: `app/ml/hybrid_training.py`, `scripts/train_hybrid_weights.py`

#### Wariant B — Manual blend

**Warunek:** użytkownik zarejestrowany, lub CF niedostępny, lub Ridge ≤ 0

```text
final = Σ manual_weight[s] × breakdown_norm[s]
```

**Plik:** `app/ml/user_profile.py` → `blend_signal_weights()`

`weight_source` w odpowiedzi: `"learned"` lub `"manual"` — `hybrid.last_weight_source`.

---

### Krok 6 — Explanations + odpowiedź

- `app/ml/explanations.py` → `build_explanations()`
- Budowa DTO: `app/schemas/recommendation.py`
- Zapis w BD: `recommendation_service._persist()` → tabela `recommendations`

---

## 6. Pięć sygnałów: ML vs heurystyki

**Taksonomia:** `app/ml/signals.py`

| Klucz | Typ | Metoda | Zbiór danych | Plik inference |
|-------|-----|--------|--------------|----------------|
| `cf` | **ML** | Surprise SVD | DS1 | `app/ml/collaborative.py` |
| `content` | **ML** | TF-IDF cosine | DS2+DS3 | `app/ml/content_based.py` |
| `cluster` | **ML** | K-Means + affinity | DS1 | `app/ml/user_clustering.py`, `app/ml/cluster_affinity.py` |
| `pop` | **Heurystyka** | `rating_count` | Agregat SQLite | `app/repositories/book_repository.py` |
| `genre` | **Heurystyka** | Nakładanie gatunków / priors | DS2 + priors | `app/ml/user_profile.py`, `app/ml/genre_priors.py` |

### 6.1. Collaborative Filtering (SVD)

- **Uczenie:** `bookrec/ml/collaborative/` (Surprise, `n_factors=50`)
- **Inference:** `CollaborativeFilteringEngine.predict(user_id, book_id)`
- **Serwis:** `app/services/collaborative_filtering_service.py`
- **Kandydaci:** książki z `cf_train.parquet`, nieocenione przez użytkownika (do 2000)
- **Normalizacja dla Ridge:** `(predict - 1) / 4` → [0, 1]

### 6.2. Content (TF-IDF)

- **Macierz:** sparse L2-normalized TF-IDF, ~149k książek
- **Wektor użytkownika:** ważona średnia TF-IDF do 10 ocenionych książek (waga = ocena)
- **Podobieństwo:** iloczyn skalarny = cosine
- **Kod:** `ContentRecommendationEngine.build_user_vector()`, `score_candidates()`, `similar_books()`

### 6.3. Cluster

**Online K-Means** (`app/ml/user_clustering.py`):
- Cechy: `n_ratings`, `mean_rating`, `std_rating`, `rating_range`, activity one-hot
- Model uczony na DS1 (`bookrec/features/clustering.py`)

**Cluster affinity** (`app/ml/cluster_affinity.py`):
- Offline: `scripts/build_cluster_affinity.py`
- Dla każdego klastra: top-200 książek wg `count × avg_rating` z interakcji DS1
- JSON: `cluster_affinity.json`

### 6.4. Popularity

- `books.rating_count` — denormalizowane przy `load_db` / `app/db/book_stats.py`
- W Ridge: `count / max(count)` wśród kandydatów
- W manual: min-max wśród kandydatów

### 6.5. Genre

- **Z ocenami:** suma ocen wg gatunków ocenionych książek → normalizacja
- **Bez ocen:** `GenrePriorStore.for_cluster()` — 35% global + 65% cluster
- **Match:** suma wag użytkownika po tagach książki

---

## 7. Wagi: manual i learned (Ridge)

### 7.1. Wagi manualne (`blend_signal_weights`)

**Plik:** `app/ml/user_profile.py`

Wagi bazowe wg liczby ocen `n`:

| n ocen | cf | content | cluster | pop | genre |
|--------|-----|---------|---------|-----|-------|
| 0 | 0 | 0 | 0.30 | 0.45 | 0.25 |
| 1–2 | 0 | 0.50 | 0.20 | 0.20 | 0.10 |
| 3–9 | 0.25 | 0.45 | 0.15 | 0.10 | 0.05 |
| ≥10 | 0.40 | 0.35 | 0.15 | 0.05 | 0.05 |

**Jeśli `cf_available = False`** (wszyscy zarejestrowani + cold DS1):

Udział `cf` jest przenoszony:
- 60% → content
- 25% → cluster
- 15% → pop

**Przykład: zarejestrowany, 11 ocen** (jak user #3981):

```text
Bazowy cf = 0.40 → przeniesienie
content ≈ 0.35 + 0.24 = 0.59
cluster ≈ 0.15 + 0.10 = 0.25
pop     ≈ 0.05 + 0.06 = 0.11
genre   = 0.05

final ≈ 0.59×C + 0.25×K + 0.11×P + 0.05×G
```

### 7.2. Wagi uczone (Ridge)

**Uczenie:** `scripts/train_hybrid_weights.py`  
**Macierz cech:** `app/ml/hybrid_training.py`  
**Model:** `sklearn.linear_model.Ridge`  
**Target:** `(rating - 1) / 4` z `cf_train.parquet`  
**Artefakt:** `ridge_weights.joblib` + `ridge_weights.json`

**Aktualne współczynniki** (po uczeniu na próbce 12000):

| Cecha | Współczynnik | Intercept |
|-------|--------------|-----------|
| cf | **+1.48** | -0.34 |
| content | +0.01 | |
| cluster | -0.05 | |
| pop | 0.00 | |
| genre | -0.005 | |

**Metryki hold-out:** RMSE ≈ 0.152, R² ≈ 0.63

**Kiedy stosowany Ridge:** tylko `cf_available=True` (DS1 w train).  
**Dlaczego zarejestrowani nie używają Ridge:** bez CF predykcja jest ujemna (intercept + małe content/genre) → obcinana do 0. Naprawione: dla zarejestrowanych zawsze manual.

**Porównanie baseline'ów:** `scripts/evaluate_hybrid_baselines.py`  
Wynik: `data/processed/models/evaluation/hybrid/baseline_comparison.json`

| Metoda | RMSE |
|--------|------|
| Tylko CF | ~0.164 |
| Tylko content | ~0.739 |
| Manual hybrid | ~0.559 |
| **Learned Ridge** | **~0.149** |

---

## 8. Cold start i genre priors

### 8.1. Użytkownik z 0 ocenami

| Komponent | Wartość |
|-----------|---------|
| `profile_strength` | 0 |
| `cf_available` | False |
| `content_vector` | None |
| `cluster_id` | ≈ 1 (domyślny K-Means) |
| `genre_weights` | Z `genre_priors.json` |
| `genre_prior_active` | True |

**Wagi manualne:** cluster 30% + pop 45% + genre 25%

### 8.2. Genre priors

**Budowa:** `scripts/build_genre_priors.py`  
**Plik:** `data/processed/features/clustering/genre_priors.json`  
**Kod:** `app/ml/genre_priors.py`

- **global:** top-40 gatunków z `content_catalog.parquet` (DS2)
- **clusters:** gatunki top-książek klastra z `cluster_affinity.json`
- **Mieszanie:** `0.35 × global + 0.65 × cluster`

---

## 9. Wyjaśnienia (explanations)

**Plik:** `app/ml/explanations.py`

Do 3 linii na książkę, przykłady:
- „Similar to books you rated (nonfiction, science)»
- „Users with similar tastes rated this highly (collaborative filtering)»
- „Popular among readers like you (Moderate activity)»
- „Matches your preference for nonfiction»

**API:** `RecommendationItem.explanations`  
**UI:** `app/templates/partials/_recommendation_table.html`

---

## 10. Punkty wejścia (UI / API)

| URL / Endpoint | Plik routera | Wywołanie |
|----------------|--------------|-----------|
| `GET /me` | `app/routers/web/account.py` | `recommend_for_user(profile.id)` |
| `GET/POST /recommendations` | `app/routers/web/pages.py` | `recommend_for_user(user_id)` |
| `GET /users/{id}` | `app/routers/web/users.py` | podgląd rekomendacji |
| `GET /api/v1/users/{id}/recommendations` | `app/routers/api/users.py` | API |
| `POST /api/v1/recommendations/for-user` | `app/routers/api/recommendations.py` | API |

**DI wiring:** `app/dependencies.py` → `get_recommendation_service()` podłącza CF, content, clustering, cluster_affinity, genre_priors, hybrid_weights.

**Format odpowiedzi:** `app/schemas/recommendation.py` → `RecommendationResponse` z `profile`, `items[].score_breakdown`, `items[].explanations`.

---

## 11. Podobne książki (osobna ścieżka)

**Nie używa profilu użytkownika** — tylko content-based.

| Endpoint | Metoda |
|----------|--------|
| `/books/similar` | `RecommendationService.similar_books(book_id)` |
| API books similar | to samo |

**Kod:** `content_engine.similar_books(source_book_id)` → `app/ml/content_based.py`  
**algorithm w odpowiedzi:** `"content"`

---

## 12. Skrypty przygotowania i uczenia

```powershell
# 1. Ładowanie SQLite z parquet
python scripts/load_db.py

# 2. Cluster → book affinity (po ML pipeline DS1)
python scripts/build_cluster_affinity.py

# 3. Genre priors dla cold start
python scripts/build_genre_priors.py

# 4. Uczenie wag fuzji Ridge
python scripts/train_hybrid_weights.py
# opcjonalnie: --sample-size 12000 (domyślnie)

# 5. Porównanie CF / content / hybrid
python scripts/evaluate_hybrid_baselines.py

# 6. Uruchomienie aplikacji
uvicorn app.main:app --reload
```

**Pełny ML pipeline (offline):** `bookrec/ml/runner.py` / `scripts/run_data_pipeline.py`

---

## 13. Mapa plików

### Orchestration & API

```
app/services/recommendation_service.py    # Główny orchestrator
app/services/collaborative_filtering_service.py
app/services/content_recommendation_service.py
app/services/cold_start_service.py        # Tylko UI „Books to rate first»
app/dependencies.py                       # DI
app/schemas/recommendation.py             # Schematy API
```

### ML core

```
app/ml/user_profile.py          # UserProfile, wagi manualne, genre match
app/ml/hybrid_scoring.py        # Kandydaci + scoring + explanations
app/ml/hybrid_weights.py        # Ridge load/predict
app/ml/hybrid_training.py       # Cechy offline do uczenia
app/ml/signals.py               # Taksonomia ML vs heurystyki
app/ml/collaborative.py         # Surprise SVD
app/ml/content_based.py         # Silnik TF-IDF
app/ml/user_clustering.py       # Online K-Means
app/ml/cluster_affinity.py      # Cluster→book scores
app/ml/genre_priors.py          # Rozkład gatunków cold-start
app/ml/explanations.py          # Czytelne uzasadnienia
app/ml/registry.py              # Ładowanie modeli przy starcie
app/ml/sparse_loader.py         # Macierze sparse NPZ
```

### Data & config

```
app/config.py                   # Wszystkie ścieżki i limity
app/repositories/book_repository.py
app/repositories/rating_repository.py
app/repositories/user_repository.py
bookrec/paths.py                # Katalogi PROC_*, MODEL_*
scripts/load_db.py
scripts/build_cluster_affinity.py
scripts/build_genre_priors.py
scripts/train_hybrid_weights.py
scripts/evaluate_hybrid_baselines.py
```

### UI

```
app/templates/recommendations/_results.html
app/templates/partials/_recommendation_table.html
app/routers/web/account.py      # /me
app/routers/web/pages.py        # /recommendations
```

---

## 14. Przykłady według typów użytkowników

### A. Użytkownik DS1 #1 (~300 ocen)

1. `cf_available = True`
2. Kandydaci: CF top + sąsiedzi content + cluster + popularne
3. Score: **Ridge (learned)** jeśli predict > 0, w przeciwnym razie manual
4. Dominuje CF (współcz. 1.48 w Ridge)
5. `weight_source: learned`

### B. Zarejestrowany, 11 ocen (user #3981)

1. `cf_available = False` (reg:xxx nie w SVD)
2. Score: **manual blend** — content ~59%, genre ~5%
3. Rekomendacje: science/nonfiction/evolution (TF-IDF + gatunki)
4. `weight_source: manual`
5. Wysokie C/G w breakdown, F=0 — oczekiwane

### C. Nowy zarejestrowany, 0 ocen

1. `genre_prior_active = True`
2. Wagi: cluster 30% + pop 45% + genre 25%
3. Widget „Books to rate first»: `ColdStartService` (osobno, ten sam popular pool)

### D. DS1 z 2 ocenami

1. `cf_available = False` (< 3)
2. Manual: content 50%, cluster 20%, pop 20%, genre 10%

---

## 15. Sformułowanie na obronę

> Zbudowaliśmy **ujednolicony hybrydowy system rekomendacji** ze wspólnym `UserProfile` dla użytkowników ze zbioru danych i zarejestrowanych. Pięć sygnałów — trzy oparte na ML (SVD collaborative filtering na DS1, podobieństwo treści TF-IDF na DS2+DS3, affinity klastrów K-Means) oraz dwie heurystyki (popularność w katalogu i nakładanie gatunków) — są łączone przez **regresję Ridge uczone na interakcjach DS1**, z **manualnym adaptacyjnym fallbackiem**, gdy collaborative filtering jest niedostępny. Cold-start jest modelowany jako **rzadki profil** z wykorzystaniem affinity klastrów, popularności i **globalnych/klastrowych priorów gatunkowych**, a nie jako osobny pipeline. Wyjaśnienia są generowane na podstawie wkładu poszczególnych sygnałów dla przejrzystości.

---

*Przy zmianie logiki aktualizuj ten plik oraz `app/ml/signals.py`.*
