# Система рекомендаций JoyBookers — полный технический отчёт

**Версия:** unified hybrid + Ridge weights + genre priors + explanations (актуально на 2026)  
**Главный orchestrator:** `app/services/recommendation_service.py`  
**Связанные отчёты:** `ETAP_DANYCH_I_ML.md`, `data/processed/models/evaluation/hybrid/baseline_comparison.json`

---

## Содержание

1. [Общая архитектура](#1-общая-архитектура)
2. [Наборы данных DS1–DS4](#2-наборы-данных-ds1ds4)
3. [Артефакты ML и пути](#3-артефакты-ml-и-пути)
4. [Типы пользователей](#4-типы-пользователей)
5. [Полный pipeline рекомендаций](#5-полный-pipeline-рекомендаций)
6. [Пять сигналов: ML vs эвристики](#6-пять-сигналов-ml-vs-эвристики)
7. [Веса: manual и learned (Ridge)](#7-веса-manual-и-learned-ridge)
8. [Cold start и genre priors](#8-cold-start-и-genre-priors)
9. [Explanations](#9-explanations)
10. [Точки входа (UI / API)](#10-точки-входа-ui--api)
11. [Похожие книги (отдельный путь)](#11-похожие-книги-отдельный-путь)
12. [Скрипты подготовки и обучения](#12-скрипты-подготовки-и-обучения)
13. [Карта файлов](#13-карта-файлов)
14. [Примеры по типам пользователей](#14-примеры-по-типам-пользователей)
15. [Формулировка для защиты](#15-формулировка-для-защиты)

---

## 1. Общая архитектура

Система — **единый гибридный recommender** для всех пользователей. Нет отдельных веток «для DS1» и «для registered» в коде скоринга — есть **один pipeline**, но **разная доступность сигналов** (главное: CF только для пользователей DS1 из train SVD).

```
HTTP запрос (/me, /recommendations, API)
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
        │       ├─ gather candidates
        │       ├─ compute 5 signals (raw + normalized)
        │       ├─ final score (Ridge OR manual blend)
        │       └─ explanations
        │
        └─► persist + RecommendationResponse    app/schemas/recommendation.py
```

**Загрузка моделей при старте:** `app/ml/registry.py` (`MLModelRegistry.load_all()`), пути из `app/config.py`.

---

## 2. Наборы данных DS1–DS4

### DS1 — Goodreads 2M (взаимодействия)

| Роль | Что даёт |
|------|----------|
| **Collaborative Filtering** | Обучение Surprise SVD |
| **User clustering** | Признаки поведения + K-Means |
| **Cluster affinity** | Какие книги популярны в каждом кластере |
| **Пользователи/оценки в runtime** | SQLite после `scripts/load_db.py` |

**Пайплайн (offline):** `bookrec/pipeline/runner.py`  
**Interactions:** `data/processed/features/interactions/interactions_indexed.parquet`  
**Train split для SVD:** `data/processed/splits/cf_train.parquet`  
**Модель SVD:** `data/processed/models/collaborative/svd_model.pkl`  
**K-Means:** `data/processed/models/clustering/kmeans_model.joblib`  
**Назначения кластеров:** `data/processed/models/clustering/user_cluster_assignments.parquet`

**В приложении:** пользователи DS1 имеют `users.is_registered = False`, `external_id` = ID из Goodreads (совпадает с train SVD).

---

### DS2 — Goodreads 100k (каталог)

| Роль | Что даёт |
|------|----------|
| **Основной каталог книг** | SQLite `books` |
| **Content-based** | Текст для TF-IDF (title, author, description…) |
| **Жанры** | Поле `books.genre` (из каталога) |

**Пайплайн:** `bookrec/ingest/ds2_goodreads_100k.py`, feature stage  
**Каталог:** `data/processed/features/content/content_catalog.parquet`  
**TF-IDF матрица:** `data/processed/models/content/tfidf_combined.npz` (+ `book_ids.npy`)

---

### DS3 — Goodreads Best (обогащение)

| Роль | Что даёт |
|------|----------|
| **Расширение контента** | Теги, персонажи, доп. жанры |
| **TF-IDF** | Дополнительные признаки в combined matrix |
| **SQLite** | `book_enrichments` |

**Связь:** книги DS2 обогащаются через `bookrec/resolution/book_linker.py` → enrichment в SQLite.

---

### DS4 — Amazon Reviews

| Роль в рекомендациях | **Не используется** |
|----------------------|---------------------|
| Используется для | Sentiment analysis (`/sentiment`, `app/ml/sentiment.py`) |

---

### SQLite (runtime слой)

| Таблица | Источник | Зачем в рекомендациях |
|---------|----------|------------------------|
| `users` | DS1 + `/register` | Профиль, `external_id`, `cluster_id`, `is_registered` |
| `books` | DS2 (+ stubs DS1) | Каталог, `source_book_id`, жанры, `rating_count` |
| `ratings` | DS1 + оценки registered | История пользователя |
| `recommendations` | Выдачи системы | Audit / история |
| `book_enrichments` | DS3 | UI, не прямо в hybrid score |

**Загрузка:** `scripts/load_db.py` ← parquet каталог + interactions.

---

## 3. Артефакты ML и пути

Все пути задаются в `app/config.py`:

| Параметр | Путь | Файл кода |
|----------|------|-----------|
| `cf_model_path` | `data/processed/models/collaborative/svd_model.pkl` | `app/ml/collaborative.py` |
| `cf_train_path` | `data/processed/splits/cf_train.parquet` | `app/ml/collaborative.py` |
| `content_tfidf_path` | `data/processed/models/content/tfidf_combined.npz` | `app/ml/content_based.py` |
| `clustering_model_path` | `data/processed/models/clustering/kmeans_model.joblib` | `app/ml/user_clustering.py` |
| `cluster_affinity_path` | `data/processed/features/clustering/cluster_affinity.json` | `app/ml/cluster_affinity.py` |
| `genre_priors_path` | `data/processed/features/clustering/genre_priors.json` | `app/ml/genre_priors.py` |
| `hybrid_weights_path` | `data/processed/models/hybrid/ridge_weights.joblib` | `app/ml/hybrid_weights.py` |

**Реестр:** `app/ml/registry.py` — загружает все движки при `ml_eager_load=True`.

---

## 4. Типы пользователей

### 4.1. Пользователи датасета (DS1)

- `is_registered = False`
- `external_id` — числовой ID Goodreads
- Оценки из `load_db.py`
- **CF доступен** при ≥3 оценках и наличии в `cf_train.parquet`
- UI: `/recommendations`, `/users/{id}`

### 4.2. Зарегистрированные

- `is_registered = True`, `external_id = reg:xxxxxxxx`
- Оценки через `/me`, API `/api/v1/ratings/me`
- **CF всегда недоступен** (нет в train SVD)
- Скоринг: **manual blend** (content + genre + cluster + pop)
- UI: `/me`

**Проверка CF:** `UserProfileBuilder.build()` в `app/ml/user_profile.py`:

```python
cf_available = (
    not user.is_registered
    and n_ratings >= min_cf_ratings_per_user  # 3
    and external_id in cf_engine.known_user_ids()
)
```

---

## 5. Полный pipeline рекомендаций

### Шаг 0 — Точка входа

`RecommendationService.recommend_for_user(user_id, limit, algorithm)`  
Файл: `app/services/recommendation_service.py`

| `algorithm` | Поведение |
|-------------|-----------|
| `auto`, `hybrid` | `HybridScoringEngine.recommend()` |
| `collaborative` | Только SVD (`recommend_cf_only`) |
| `content` | Только TF-IDF (`recommend_content_only`) |

---

### Шаг 1 — Построение профиля

**Файл:** `app/ml/user_profile.py` → `UserProfileBuilder.build()`

| Поле | Как вычисляется |
|------|-----------------|
| `rated_books` | До 500 оценок из `RatingRepository` + `BookRepository` |
| `cluster_id` | `UserClusteringEngine.predict_cluster()` — `app/ml/user_clustering.py` |
| `cluster_label` | Метки 0/1/2 («Power users», «Lenient occasional», «Moderate») |
| `genre_weights` | Из оценённых книг ИЛИ genre priors при 0 оценок |
| `genre_prior_active` | `True` если жанры из priors, не из оценок |
| `profile_strength` | `min(1.0, n_ratings / 10)` |
| `cf_available` | См. §4 |
| `content_vector` | Опционально; обычно строится в hybrid scoring |

---

### Шаг 2 — Сбор кандидатов

**Файл:** `app/ml/hybrid_scoring.py` → `_gather_candidates()`  
Лимит: `hybrid_candidate_limit = 2500` (`app/config.py`)

Объединение (union) из источников:

| # | Источник | Условие | Код |
|---|----------|---------|-----|
| 1 | CF top | `cf_available` | `CollaborativeFilteringService.recommend_for_user()` |
| 2 | Content neighbors | До 5 оценённых книг | `ContentRecommendationEngine.similar_books()` |
| 3 | Cluster top | Всегда | `ClusterAffinityStore.top_books(cluster_id)` |
| 4 | Popular | Всегда | `BookRepository.list_starter_books()` |

Уже оценённые книги исключаются.

---

### Шаг 3 — Пять сигналов (raw)

Для каждого кандидата `source_book_id`:

| Сигнал | Raw score | Файл / метод |
|--------|-----------|--------------|
| **cf** | SVD predict 1–5 | `hybrid_scoring._cf_scores()` → `CollaborativeFilteringEngine` |
| **content** | Cosine TF-IDF | `content_based.score_candidates()` / `build_user_vector()` |
| **cluster** | Affinity 0–1 | `cluster_affinity.score(cluster_id, book_id)` |
| **pop** | `books.rating_count` | `hybrid_scoring._popularity_raw()` |
| **genre** | Overlap с `genre_weights` | `user_profile.genre_match_score()` |

---

### Шаг 4 — Нормализация для UI (breakdown)

**Файл:** `hybrid_scoring._normalize_dict()` — min-max **внутри пула кандидатов** запроса.

Отображается в UI как **C / F / K / P / G** (content, CF, cluster, popularity, genre).

> **Важно:** breakdown в UI — относительные значения для сравнения кандидатов.  
> Для Ridge используются **raw** фичи (`_ridge_features()`), не breakdown.

---

### Шаг 5 — Итоговый score

**Файл:** `app/ml/hybrid_scoring.py`, строки 106–124

#### Вариант A — Learned (Ridge)

**Условие:** модель загружена **И** `profile.cf_available == True`

```text
ridge_score = Ridge.predict([cf, content, cluster, pop, genre])  # raw features
final = ridge_score  если ridge_score > 0
        иначе manual_score (fallback)
```

**Файлы:** `app/ml/hybrid_weights.py`, обучение: `app/ml/hybrid_training.py`, `scripts/train_hybrid_weights.py`

#### Вариант B — Manual blend

**Условие:** зарегистрированный пользователь, или CF недоступен, или Ridge ≤ 0

```text
final = Σ manual_weight[s] × breakdown_norm[s]
```

**Файл:** `app/ml/user_profile.py` → `blend_signal_weights()`

`weight_source` в ответе: `"learned"` или `"manual"` — `hybrid.last_weight_source`.

---

### Шаг 6 — Explanations + ответ

- `app/ml/explanations.py` → `build_explanations()`
- Сборка DTO: `app/schemas/recommendation.py`
- Сохранение в БД: `recommendation_service._persist()` → таблица `recommendations`

---

## 6. Пять сигналов: ML vs эвристики

**Таксономия:** `app/ml/signals.py`

| Ключ | Тип | Метод | Датасет | Файл inference |
|------|-----|-------|---------|----------------|
| `cf` | **ML** | Surprise SVD | DS1 | `app/ml/collaborative.py` |
| `content` | **ML** | TF-IDF cosine | DS2+DS3 | `app/ml/content_based.py` |
| `cluster` | **ML** | K-Means + affinity | DS1 | `app/ml/user_clustering.py`, `app/ml/cluster_affinity.py` |
| `pop` | **Эвристика** | `rating_count` | SQLite агрегат | `app/repositories/book_repository.py` |
| `genre` | **Эвристика** | Жанровое пересечение / priors | DS2 + priors | `app/ml/user_profile.py`, `app/ml/genre_priors.py` |

### 6.1. Collaborative Filtering (SVD)

- **Обучение:** `bookrec/ml/collaborative/` (Surprise, `n_factors=50`)
- **Inference:** `CollaborativeFilteringEngine.predict(user_id, book_id)`
- **Сервис:** `app/services/collaborative_filtering_service.py`
- **Кандидаты:** книги из `cf_train.parquet`, не оценённые пользователем (до 2000)
- **Нормализация для Ridge:** `(predict - 1) / 4` → [0, 1]

### 6.2. Content (TF-IDF)

- **Матрица:** sparse L2-normalized TF-IDF, ~149k книг
- **User vector:** взвешенное среднее TF-IDF до 10 оценённых книг (вес = оценка)
- **Similarity:** dot product = cosine
- **Код:** `ContentRecommendationEngine.build_user_vector()`, `score_candidates()`, `similar_books()`

### 6.3. Cluster

**Online K-Means** (`app/ml/user_clustering.py`):
- Признаки: `n_ratings`, `mean_rating`, `std_rating`, `rating_range`, activity one-hot
- Модель обучена на DS1 (`bookrec/features/clustering.py`)

**Cluster affinity** (`app/ml/cluster_affinity.py`):
- Offline: `scripts/build_cluster_affinity.py`
- Для каждого кластера: top-200 книг по `count × avg_rating` из DS1 interactions
- JSON: `cluster_affinity.json`

### 6.4. Popularity

- `books.rating_count` — денормализовано при `load_db` / `app/db/book_stats.py`
- В Ridge: `count / max(count)` среди кандидатов
- В manual: min-max среди кандидатов

### 6.5. Genre

- **С оценками:** сумма оценок по жанрам оценённых книг → нормализация
- **Без оценок:** `GenrePriorStore.for_cluster()` — 35% global + 65% cluster
- **Match:** сумма весов пользователя по тегам книги

---

## 7. Веса: manual и learned (Ridge)

### 7.1. Manual weights (`blend_signal_weights`)

**Файл:** `app/ml/user_profile.py`

Базовые веса по числу оценок `n`:

| n оценок | cf | content | cluster | pop | genre |
|----------|-----|---------|---------|-----|-------|
| 0 | 0 | 0 | 0.30 | 0.45 | 0.25 |
| 1–2 | 0 | 0.50 | 0.20 | 0.20 | 0.10 |
| 3–9 | 0.25 | 0.45 | 0.15 | 0.10 | 0.05 |
| ≥10 | 0.40 | 0.35 | 0.15 | 0.05 | 0.05 |

**Если `cf_available = False`** (все registered + cold DS1):

Доля `cf` перераспределяется:
- 60% → content
- 25% → cluster
- 15% → pop

**Пример: registered, 11 оценок** (как user #3981):

```text
Базовый cf = 0.40 → перераспределение
content ≈ 0.35 + 0.24 = 0.59
cluster ≈ 0.15 + 0.10 = 0.25
pop     ≈ 0.05 + 0.06 = 0.11
genre   = 0.05

final ≈ 0.59×C + 0.25×K + 0.11×P + 0.05×G
```

### 7.2. Learned weights (Ridge)

**Обучение:** `scripts/train_hybrid_weights.py`  
**Feature matrix:** `app/ml/hybrid_training.py`  
**Модель:** `sklearn.linear_model.Ridge`  
**Target:** `(rating - 1) / 4` из `cf_train.parquet`  
**Артефакт:** `ridge_weights.joblib` + `ridge_weights.json`

**Текущие коэффициенты** (после обучения на 12000 sample):

| Feature | Coefficient | Intercept |
|---------|-------------|-----------|
| cf | **+1.48** | -0.34 |
| content | +0.01 | |
| cluster | -0.05 | |
| pop | 0.00 | |
| genre | -0.005 | |

**Метрики hold-out:** RMSE ≈ 0.152, R² ≈ 0.63

**Когда применяется Ridge:** только `cf_available=True` (DS1 в train).  
**Почему registered не используют Ridge:** без CF предсказание отрицательное (intercept + малые content/genre) → обрезалось до 0. Исправлено: для registered всегда manual.

**Сравнение baselines:** `scripts/evaluate_hybrid_baselines.py`  
Результат: `data/processed/models/evaluation/hybrid/baseline_comparison.json`

| Метод | RMSE |
|-------|------|
| CF only | ~0.164 |
| Content only | ~0.739 |
| Manual hybrid | ~0.559 |
| **Learned Ridge** | **~0.149** |

---

## 8. Cold start и genre priors

### 8.1. Пользователь с 0 оценок

| Компонент | Значение |
|-----------|----------|
| `profile_strength` | 0 |
| `cf_available` | False |
| `content_vector` | None |
| `cluster_id` | ≈ 1 (дефолт K-Means) |
| `genre_weights` | Из `genre_priors.json` |
| `genre_prior_active` | True |

**Manual веса:** cluster 30% + pop 45% + genre 25%

### 8.2. Genre priors

**Сборка:** `scripts/build_genre_priors.py`  
**Файл:** `data/processed/features/clustering/genre_priors.json`  
**Код:** `app/ml/genre_priors.py`

- **global:** top-40 жанров из `content_catalog.parquet` (DS2)
- **clusters:** жанры top-книг кластера из `cluster_affinity.json`
- **Смешивание:** `0.35 × global + 0.65 × cluster`

---

## 9. Explanations

**Файл:** `app/ml/explanations.py`

До 3 строк на книгу, примеры:
- «Similar to books you rated (nonfiction, science)»
- «Users with similar tastes rated this highly (collaborative filtering)»
- «Popular among readers like you (Moderate activity)»
- «Matches your preference for nonfiction»

**API:** `RecommendationItem.explanations`  
**UI:** `app/templates/partials/_recommendation_table.html`

---

## 10. Точки входа (UI / API)

| URL / Endpoint | Файл роутера | Вызов |
|----------------|--------------|-------|
| `GET /me` | `app/routers/web/account.py` | `recommend_for_user(profile.id)` |
| `GET/POST /recommendations` | `app/routers/web/pages.py` | `recommend_for_user(user_id)` |
| `GET /users/{id}` | `app/routers/web/users.py` | превью рекомендаций |
| `GET /api/v1/users/{id}/recommendations` | `app/routers/api/users.py` | API |
| `POST /api/v1/recommendations/for-user` | `app/routers/api/recommendations.py` | API |

**DI wiring:** `app/dependencies.py` → `get_recommendation_service()` подключает CF, content, clustering, cluster_affinity, genre_priors, hybrid_weights.

**Формат ответа:** `app/schemas/recommendation.py` → `RecommendationResponse` с `profile`, `items[].score_breakdown`, `items[].explanations`.

---

## 11. Похожие книги (отдельный путь)

**Не использует user profile** — только content-based.

| Endpoint | Метод |
|----------|-------|
| `/books/similar` | `RecommendationService.similar_books(book_id)` |
| API books similar | то же |

**Код:** `content_engine.similar_books(source_book_id)` → `app/ml/content_based.py`  
**algorithm в ответе:** `"content"`

---

## 12. Скрипты подготовки и обучения

```powershell
# 1. Загрузка SQLite из parquet
python scripts/load_db.py

# 2. Cluster → book affinity (после ML pipeline DS1)
python scripts/build_cluster_affinity.py

# 3. Genre priors для cold start
python scripts/build_genre_priors.py

# 4. Обучение Ridge fusion weights
python scripts/train_hybrid_weights.py
# опционально: --sample-size 12000 (дефолт)

# 5. Сравнение CF / content / hybrid
python scripts/evaluate_hybrid_baselines.py

# 6. Запуск приложения
uvicorn app.main:app --reload
```

**Полный ML pipeline (offline):** `bookrec/ml/runner.py` / `scripts/run_data_pipeline.py`

---

## 13. Карта файлов

### Orchestration & API

```
app/services/recommendation_service.py    # Главный orchestrator
app/services/collaborative_filtering_service.py
app/services/content_recommendation_service.py
app/services/cold_start_service.py        # Только UI «Books to rate first»
app/dependencies.py                       # DI
app/schemas/recommendation.py             # API schemas
```

### ML core

```
app/ml/user_profile.py          # UserProfile, manual weights, genre match
app/ml/hybrid_scoring.py        # Candidates + scoring + explanations trigger
app/ml/hybrid_weights.py        # Ridge model load/predict
app/ml/hybrid_training.py       # Offline training features
app/ml/signals.py               # ML vs heuristic taxonomy
app/ml/collaborative.py         # Surprise SVD
app/ml/content_based.py         # TF-IDF engine
app/ml/user_clustering.py       # Online K-Means
app/ml/cluster_affinity.py      # Cluster→book scores
app/ml/genre_priors.py          # Cold-start genre distribution
app/ml/explanations.py          # Human-readable reasons
app/ml/registry.py              # Model loading at startup
app/ml/sparse_loader.py         # NPZ sparse matrices
```

### Data & config

```
app/config.py                   # Все пути и лимиты
app/repositories/book_repository.py
app/repositories/rating_repository.py
app/repositories/user_repository.py
bookrec/paths.py                # PROC_*, MODEL_* dirs
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

## 14. Примеры по типам пользователей

### A. DS1 user #1 (~300 оценок)

1. `cf_available = True`
2. Кандидаты: CF top + content neighbors + cluster + popular
3. Score: **Ridge (learned)** если predict > 0, иначе manual
4. Доминирует CF (коэф. 1.48 в Ridge)
5. `weight_source: learned`

### B. Зарегистрированный, 11 оценок (user #3981)

1. `cf_available = False` (reg:xxx не в SVD)
2. Score: **manual blend** — content ~59%, genre ~5%
3. Рекомендации: science/nonfiction/evolution (по TF-IDF + жанрам)
4. `weight_source: manual`
5. C/G в breakdown высокие, F=0 — ожидаемо

### C. Новый registered, 0 оценок

1. `genre_prior_active = True`
2. Веса: cluster 30% + pop 45% + genre 25%
3. Виджет «Books to rate first»: `ColdStartService` (отдельно, тот же popular pool)

### D. DS1 с 2 оценками

1. `cf_available = False` (< 3)
2. Manual: content 50%, cluster 20%, pop 20%, genre 10%

---

## 15. Формулировка для защиты

> We built a **unified hybrid recommender** with a shared `UserProfile` for dataset and registered users. Five signals — three ML-based (SVD collaborative filtering on DS1, TF-IDF content similarity on DS2+DS3, K-Means cluster affinity) and two heuristics (catalog popularity and genre overlap) — are fused via **Ridge regression learned on DS1 interactions**, with a **manual adaptive fallback** when collaborative filtering is unavailable. Cold-start is modeled as a **sparse profile** using cluster affinity, popularity, and **cluster/global genre priors**, not a separate pipeline. Explanations are generated from per-signal contributions for transparency.

---

*При изменении логики обновляйте этот файл и `app/ml/signals.py`.*
