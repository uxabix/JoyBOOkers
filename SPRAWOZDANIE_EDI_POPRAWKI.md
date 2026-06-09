# Poprawki i uzupełnienia sprawozdania EDI

**Dokument:** `EDI_projekt_Alishkevich_Kulesza.pdf`  
**Projekt:** System rekomendacji książek (JoyBOOkers)  
**Autorzy:** Kiryl Alishkevich, Damian Kulesza  
**Cel tego pliku:** szczegółowa lista tego, co należy poprawić, uzupełnić lub dopisać w sprawozdaniu z pracowni PS5 (EDI), wraz z gotowymi propozycjami tekstu do wklejenia.

---

## Spis treści

1. [Podsumowanie — co jest OK, a co pilne](#1-podsumowanie--co-jest-ok-a-co-pilne)
2. [Błędy krytyczne do natychmiastowej poprawy](#2-błędy-krytyczne-do-natychmiastowej-poprawy)
3. [Struktura dokumentu — co dodać i w jakiej kolejności](#3-struktura-dokumentu--co-dodać-i-w-jakiej-kolejności)
4. [Poprawki sekcja po sekcji](#4-poprawki-sekcja-po-sekcji)
5. [Rzeczy całkowicie pominięte w sprawozdaniu](#5-rzeczy-całkowicie-pominięte-w-sprawozdaniu)
6. [Gotowe fragmenty tekstu (do wklejenia)](#6-gotowe-fragmenty-tekstu-do-wklejenia)
7. [Bibliografia — propozycja pozycji](#7-bibliografia--propozycja-pozycji)
8. [Styl, język i formatowanie](#8-styl-język-i-formatowanie)
9. [Checklista przed oddaniem](#9-checklista-przed-oddaniem)

---

## 1. Podsumowanie — co jest OK, a co pilne

### Co już działa w sprawozdaniu

- Poprawna strona tytułowa (PB, Wydział Informatyki, EDI, prowadząca, autorzy, temat).
- Temat projektu jest zgodny z zakresem przedmiotu EDI.
- Większość wymaganych zagadnień z planu projektu jest **wymieniona** (9 punktów).
- Są fragmenty kodu z realnego repozytorium — widać, że projekt istnieje.
- Sekcja EDA zawiera interpretację wykresów (nie tylko kod).
- Opis klastrów użytkowników (K-Means, k=3) jest sensowny i zgodny z raportami ML.
- Jest szczera ocena jakości rekomendacji („nie zawsze trafne na najwyższych pozycjach”).

### Co jest pilne (bez tego sprawozdanie wygląda niedokończone)

| Priorytet | Problem |
|-----------|---------|
| 🔴 Krytyczny | Błędna interpretacja RMSE (89% zamiast ~0,89) |
| 🔴 Krytyczny | Brak **Podsumowania / Wniosków** |
| 🔴 Krytyczny | Brak **Bibliografia** |
| 🔴 Krytyczny | Sekcje 4 i 5 praktycznie puste |
| 🟠 Ważny | Brak osobnego rozdziału o **aplikacji webowej** ze zrzutami ekranu |
| 🟠 Ważny | Czas przyszły w opisie zakresu („będzie obejmować”) |
| 🟠 Ważny | Zbyt dużo surowego kodu, za mało opisu wyników |
| 🟡 Pożądany | Brak opisu **architektury wielu zbiorów danych** (DS1–DS4) |
| 🟡 Pożądany | Brak opisu **modelu hybrydowego** i porównania baseline'ów |
| 🟡 Pożądany | Brak spisu treści, wprowadzenia akademickiego, diagramu architektury |

---

## 2. Błędy krytyczne do natychmiastowej poprawy

### 2.1. RMSE — błędna jednostka i interpretacja

**Obecny tekst (błędny):**
> „Dla algorytmu SVD błąd średniokwadratowy wynosi 89%, co jest dobrym wynikiem, ale precision i recall są niskie.”

**Poprawny tekst:**

> Dla algorytmu SVD (biblioteka Surprise) na zbiorze testowym uzyskano **RMSE = 0,89** oraz **MAE = 0,71** (skala ocen 1–5). Wartość RMSE bliska 1 oznacza, że średni błąd predykcji wynosi około jednej „gwiazdki”, co jest typowe dla rzadkiej macierzy ocen (gęstość ≈ 0,12%). Metryki rankingowe **Precision@10 = 0,0056** i **Recall@10 = 0,0066** są niskie — wynika to z dużej liczby kandydatów i klasycznego problemu rekomendacji w warunkach wysokiej rzadkości danych, a nie z samego błędu regresji.

**Źródło liczb:** `reports/ml/evaluation/evaluate_all.json`

**Co jeszcze dopisać przy tej metryce:**
- Wyjaśnij różnicę między **metrykami regresji** (RMSE, MAE) a **metrykami rankingowymi** (Precision@K, Recall@K).
- Dodaj zdanie: „Niska precision@K nie oznacza, że model jest bezużyteczny — w praktyce użytkownik widzi top-N propozycji, a jakość subiektywna bywa wyższa niż surowa metryka.”

---

### 2.2. Czas gramatyczny — plan zamiast sprawozdania

**Obecny fragment (błędny styl):**
> „Realizacja projektu **będzie** obejmować następujące elementy…”  
> „Projekt **będzie** zawierał aplikację internetową…”

**Poprawka — zamień na czas przeszły:**

> „Realizacja projektu **obejmowała** następujące elementy…”  
> „Projekt **zawiera** aplikację internetową…”

Cały wstęp (cele, zakres) powinien brzmieć jak **raport z wykonanej pracy**, nie jak opis planu na przyszłość.

---

### 2.3. Duplikacja sekcji o klasteryzacji

K-Means jest opisany szczegółowo w **§6** (UserClusteringEngine, profile klastrów, PCA), a potem **§9** powtarza to samo w 3 zdaniach.

**Propozycja:**
- **§6** — trening modelu, wybór k, metryki (Silhouette = 0,47), profile klastrów, wizualizacja PCA.
- **§9** — **zastąp** opisem **zastosowania** klasteryzacji w systemie: endpoint `/clustering/predict`, cold-start, wpływ klastra na rekomendacje hybrydowe, przypisanie nowego użytkownika po kilku ocenach.

---

## 3. Struktura dokumentu — co dodać i w jakiej kolejności

### Proponowana nowa struktura (49+ stron → ~35–45 stron, ale bogatsza merytorycznie)

```
Strona tytułowa
Spis treści                          ← DODAĆ
1. Wprowadzenie                      ← DODAĆ (2–3 strony)
2. Opis problemu i cele projektu     ← przerobić z obecnego wstępu
3. Zbiory danych i architektura      ← DODAĆ (ważne!)
4. Przygotowanie i czyszczenie danych
5. Analiza eksploracyjna (EDA)
6. Normalizacja i przetwarzanie tekstu
7. Selekcja cech                     ← ROZSZERZYĆ
8. Miary podobieństwa                ← ROZSZERZYĆ
9. Algorytmy ML i ocena jakości
10. System rekomendacji hybrydowy    ← DODAĆ / wyodrębnić z §7
11. Analiza sentymentu
12. Grupowanie użytkowników (K-Means)
13. Aplikacja internetowa            ← DODAĆ (osobny rozdział!)
14. Podsumowanie i wnioski           ← DODAĆ
15. Bibliografia                     ← DODAĆ
Załączniki (opcjonalnie)             ← zrzuty ekranu, tabele metryk
```

---

## 4. Poprawki sekcja po sekcji

### 4.0. NOWY rozdział: Wprowadzenie (brakuje całkowicie)

**Co napisać (ok. 1–2 strony):**

1. Kontekst — platformy typu LubimyCzytać, problem informacyjny (miliony książek, ograniczona uwaga użytkownika).
2. Cel pracy — zbudowanie systemu łączącego CF, content-based, sentyment i segmentację użytkowników.
3. Zakres — cztery zbiory danych, pipeline danych, modele ML, aplikacja FastAPI.
4. Technologie — Python, pandas, scikit-learn, Surprise, FastAPI, Jinja2/HTMX, SQLite.
5. Struktura sprawozdania — krótko, co w którym rozdziale.

**Propozycja akapitu otwierającego:**

> W dobie cyfrowej czytelnictwa użytkownicy mają dostęp do setek tysięcy tytułów, co utrudnia samodzielne odkrywanie interesujących pozycji. Systemy rekomendacji rozwiązują ten problem, analizując historię ocen oraz treść opisów i recenzji. Celem niniejszego projektu było zaprojektowanie i implementacja systemu rekomendacji książek inspirowanego platformami społecznościowymi dla czytelników, z pełnym pipeline'em eksploracji danych internetowych oraz aplikacją webową prezentującą wyniki analiz.

---

### 4.0b. NOWY rozdział: Zbiory danych i architektura systemu (całkowicie pominięte!)

To jeden z **najważniejszych** braków. W projekcie są **cztery datasety** z różnymi rolami — sprawozdanie mówi o „CSV” ogólnie, ale nie wyjaśnia strategii.

**Tabela do wstawienia:**

| Zbiór | Nazwa | Rola w projekcie | Liczba rekordów (po czyszczeniu) |
|-------|-------|------------------|----------------------------------|
| DS1 | Goodreads 2M | CF (SVD), K-Means użytkowników, EDA ocen | ~1,55 mln książek, 235 484 interakcji, 3980 użytkowników |
| DS2 | Goodreads 100k | Content-based (TF-IDF): opisy, gatunki, autorzy | ~100 000 książek |
| DS3 | Goodreads Best | Wzbogacenie DS2 (tagi, postacie) przez `match_key` | merge z DS2 |
| DS4 | Amazon Reviews | NLP / analiza sentymentu (niezależny od Goodreads) | recenzje z etykietami |

**Ważna decyzja projektowa do opisania:**

> Ze względu na rozmiar katalogu DS1 (ponad 1,5 mln książek) **nie** budowano macierzy TF-IDF na pełnym katalogu DS1. Rekomendacje content-based opierają się na DS2 (+ wzbogacenie DS3). Filtracja kolaboratywna korzysta z DS1. Analiza sentymentu — z DS4. Takie rozdzielenie ról zbiorów jest typowe w projektach EDI i pokazuje świadome zarządzanie skalą danych.

**Diagram architektury (mermaid lub rysunek):**

```
[DS1 Goodreads 2M]  → preprocess → interactions → SVD + K-Means
[DS2 + DS3]         → preprocess → TF-IDF matrix → Content engine
[DS4 Amazon]        → NLP splits → Logistic Regression → Sentiment
                              ↓
                    Hybrid Recommendation Service
                              ↓
                    FastAPI + SQLite + Web UI
```

Źródła: `reports/data_pipeline/pipeline_summary.json`, `DOKUMENTACJA_PROJEKTU.md`, `ETAP_DANYCH_I_ML.md`

---

### 4.1. §1 Przygotowanie i czyszczenie danych

**Co jest OK:** logika `load_books_csv`, `clean_books`, `clean_interactions`, fuzzy matching tytułów.

**Co poprawić:**

1. **Skrócić kod** — zostaw 15–25 linii reprezentatywnych + odesłanie „pełna implementacja w module `bookrec/preprocess/`”.
2. **Dodać tabelę wyników czyszczenia** (liczby z raportów):

| Etap | Przed | Po | Uwagi |
|------|-------|-----|-------|
| Książki DS1 | 1 850 310 | 1 548 594 | usunięto duplikaty ID, błędne pola |
| Oceny DS1 | 362 596 | 235 484 | dopasowanie tytułów (fuzzy), filtrowanie nieznanych książek |
| Użytkownicy | — | 3 980 | — |
| Książki z ocenami | — | 48 920 | — |
| Gęstość macierzy | — | 0,12% | typowa rzadkość dla CF |

3. **Opisać fuzzy matching** słownie (nie tylko kod):
   - exact po normalizacji tytułu,
   - dopasowanie „core” tytułu,
   - fuzzy (rapidfuzz, próg 88) dla pozostałych,
   - statystyki: ~209k exact, ~27k core, ~347 fuzzy, ~121k bez dopasowania.

4. **Dodać akapit o outlierach** z `preprocess_summary.json`:
   - max 2231 ocen na użytkownika,
   - 454 użytkowników z tylko 1 oceną,
   - 28 592 książek z tylko 1 oceną.

5. **Wspomnieć o formatach wyjściowych:** Parquet/CSV, macierz user–item, indeksy użytkowników i książek.

---

### 4.2. §2 Analiza eksploracyjna

**Co jest OK:** wykresy, interpretacja języka, wydawców, rozkładu ocen.

**Co dodać:**

1. **Tabela statystyk opisowych** interakcji:
   - średnia ocena: **3,80**
   - mediana liczby ocen na użytkownika,
   - rozkład 1★–5★ (procenty).

2. **Wykresy** — upewnij się, że w PDF są **obrazy** (nie tylko opisy „Widać, że…”). Pliki PNG są w `reports/eda/`:
   - `01_rating_distribution.png`
   - `02_top_books.png`
   - `03_top_users.png`

3. **Akapit o outlierach w EDA:**
   > Analiza outlierów wykazała długi ogon aktywności użytkowników — najaktywniejszy użytkownik wystawił 2231 ocen, podczas gdy mediana wynosi znacznie mniej. Większość książek ma bardzo mało ocen (28 592 tytuły z jedną oceną), co wpływa na stabilność średnich ratingów w katalogu.

4. **Wnioski z EDA** (lista 3–5 punktów na końcu sekcji):
   - dominacja ocen 4★ i 5★,
   - angielski jako główny język katalogu,
   - silna koncentracja u wydawców,
   - rzadka macierz ocen → uzasadnienie dla modeli latent-factor (SVD).

---

### 4.3. §3 Normalizacja i przetwarzanie tekstu

**Co jest OK:** `normalize_review_text`, TF-IDF, tokenizacja.

**Co dodać:**

1. **Pipeline NLP krok po kroku** (lista numerowana):
   1. Usunięcie tagów HTML
   2. Normalizacja Unicode (NFKD)
   3. Tokenizacja (regex `[a-z0-9']+`)
   4. Budowa słownika z min_df
   5. TF-IDF z sublinear_tf
   6. L2-normalizacja wierszy (dla cosine similarity)

2. **Parametry wektoryzacji** — tabela:

| Parametr | Wartość | Znaczenie |
|----------|---------|-----------|
| max_features | 5000–20000 | ograniczenie wymiaru |
| min_df | 2 | ignorowanie hapaksów |
| ngram_range | (1, 2) | sentyment — bigramy |
| sublinear_tf | True | łagodzenie częstotliwości |

3. **Różnica DS2 (content) vs DS4 (sentyment)** — wyjaśnij, że to **dwa niezależne korpusy** z różnym celem.

4. **Skrócić listingi kodu** — zostaw `normalize_review_text` i fragment `TfidfVectorizer`, resztę usuń.

---

### 4.4. §4 Selekcja cech — OBECNIE PUSTA (1 akapit!)

**To musi być pełny rozdział (1,5–2 strony).**

**Co opisać:**

#### A. Cechy użytkownika do K-Means (`bookrec/features/clustering.py`)

| Cecha | Opis | Uzasadnienie |
|-------|------|--------------|
| `n_ratings` | liczba wystawionych ocen | poziom aktywności |
| `mean_rating` | średnia ocena | „hojność” vs krytycyzm |
| `std_rating` | odchylenie ocen | stabilność preferencji |
| `rating_range` | max − min | rozpiętość gustu |
| `activity_low/medium/high` | one-hot z kwantyli q33, q66 | dyskretna aktywność |

Filtr: użytkownicy z **min. 3 ocenami** → 3311 użytkowników (669 odrzuconych).

#### B. Cechy książek do content-based

- TF-IDF opisu (`description`)
- TF-IDF gatunków (`genres`)
- TF-IDF tytułu i autora
- macierz sparse CSR, łącznie ~149k książek w modelu content

#### C. Cechy do hybrydy (sygnały w `recommend_for_user`)

Wymień: CF score, content similarity, cluster affinity, genre priors, popularity, cold-start heuristics.

#### D. Selekcja — dlaczego te, a nie inne?

> Nie stosowano automatycznej selekcji typu RFE czy LASSO — zestaw cech behawioralnych został dobrany ekspercko na podstawie literatury o segmentacji użytkowników w systemach rekomendacji. Dla TF-IDF selekcja odbywa się implicit przez `max_features` i `min_df` (odcięcie rzadkich terminów).

**Opcjonalnie:** jeśli w `analytics` jest tabela cech — wstaw screenshot lub tabelę.

---

### 4.5. §5 Miary podobieństwa — OBECNIE PUSTA (2 zdania!)

**Rozdział 1–1,5 strony.**

**Co opisać:**

1. **Cosine similarity** — definicja, dlaczego przy TF-IDF (wektory L2-normalizowane → iloczyn skalarny = cosinus).

2. **Gdzie używana:**
   - `ContentRecommendationEngine.similar_books` — podobne książki,
   - `build_user_vector` — profil użytkownika jako ważona średnia wektorów,
   - `score_candidates` — dopasowanie profilu do kandydatów.

3. **SVD (latent factors)** — podobieństwo implicit w przestrzeni czynników latentnych (user/item factors), inna natura niż cosinus na TF-IDF.

4. **Silhouette coefficient** — ocena jakości klastrów K-Means (wartość **0,47** dla k=3).

5. **Metryki ewaluacji content:** `mean_neighbor_cosine = 0,63`, `mean_genre_overlap_at_k = 0,71` — dopisz interpretację:
   > Wysokie pokrycie gatunków w top-10 sąsiadów (71%) potwierdza, że podobieństwo tekstowe koreluje z rzeczywistą tematyką książek.

---

### 4.6. §6 Budowa i ocena algorytmów

**Co jest OK:** K-Means, SVD, Content engine, profile klastrów.

**Co poprawić/dodać:**

1. **Tabela hyperparametrów SVD:**

| Parametr | Wartość |
|----------|---------|
| n_factors | 50 |
| n_epochs | 20 |
| test_size | 20% |
| random_state | 42 |
| rating_scale | 1–5 |

2. **Tabela wyboru k dla K-Means** (z `train_report.json`):

| k | Silhouette |
|---|------------|
| 3 | **0,470** ← wybrane |
| 4 | 0,422 |
| 5 | 0,436 |
| 6 | 0,446 |

3. **Wykres/wzmianka o PCA** — jeśli jest wizualizacja klastrów, wstaw obraz z `reports/ml/clustering/` lub ze strony `/clustering` w aplikacji.

4. **Metryki sentymentu** — popraw i rozwiń:

| Metryka | Wartość |
|---------|---------|
| Accuracy | 91,8% |
| F1 macro | 85,0% |
| Precision klasy negatywnej | 0,64 |
| Recall klasy negatywnej | 0,91 |

Dopisz: klasa pozytywna dominuje w zbiorze → `class_weight='balanced'`.

5. **Skrócić** listing `UserClusteringEngine` — zostaw opis słowny + 10 linii kluczowej metody `features_from_scores`.

---

### 4.7. §7 Generowanie rekomendacji — ROZSZERZYĆ o hybrydę

Obecna sekcja wspomina hybrid, ale **nie opisuje go naprawdę**. To serce projektu!

**Co dodać — nowy podrozdział „Model hybrydowy”:**

#### Sygnały łączone w rekomendacji

1. **Collaborative filtering (SVD)** — predykcja oceny
2. **Content-based (TF-IDF)** — podobieństwo do profilu użytkownika
3. **Cluster affinity** — preferencje grupy (klaster)
4. **Genre priors** — cold-start na podstawie gatunków
5. **Popularity** — fallback dla nowych użytkowników

#### Wagi — manual vs Ridge

> System obsługuje dwa źródła wag: ręczne (`blend_signal_weights`) oraz **nauczone regresją Ridge** (`train_hybrid_weights.py`). Model Ridge osiąga znacznie lepsze wyniki na hold-out.

**Tabela porównania baseline'ów** (z `reports/ml/evaluation/hybrid/baseline_comparison.json`):

| Model | RMSE | MAE |
|-------|------|-----|
| CF only | 0,170 | 0,134 |
| Content only | 0,724 | 0,677 |
| Cluster only | 0,668 | 0,610 |
| Popularity only | 0,735 | 0,689 |
| Manual hybrid | 0,544 | 0,500 |
| **Learned Ridge hybrid** | **0,154** | **0,118** |

**Interpretacja do napisania:**

> Model hybrydowy z wagami Ridge redukuje RMSE o ok. 83% względem samego content-based i o ok. 9% względem samego CF. Potwierdza to, że łączenie sygnałów behawioralnych i treściowych daje lepsze predykcje niż pojedyncze podejścia. W aplikacji domyślnie używany jest tryb `auto` / `hybrid`.

#### Cold-start

Opisz co się dzieje gdy:
- użytkownik nowy (brak historii) → genre priors, popularność,
- książka spoza macierzy CF → content + popularity,
- `cf_available=False` w profilu użytkownika.

#### Endpointy API

| Endpoint | Metoda | Opis |
|----------|--------|------|
| `/api/v1/recommendations/{user_id}` | GET | rekomendacje dla użytkownika |
| `/recommendations` | GET/POST | formularz webowy |
| `/me/recommendations` | GET | rekomendacje zalogowanego użytkownika |

Parametr `algorithm`: `auto`, `hybrid`, `collaborative`, `content`.

---

### 4.8. §8 Analiza sentymentu

**Co dodać:**

1. **Zbiór DS4** — skąd recenzje, jak powstały etykiety (`sentiment_label` 0/1).
2. **Podział train/val/test** — liczby wierszy.
3. **Przykład klasyfikacji** — 2–3 zdania recenzji z predykcją (zrób screenshot ze strony `/sentiment`).
4. **Ograniczenia:**
   - model trenowany na recenzjach Amazon, nie Goodreads,
   - klasa negatywna rzadsza → niższa precision,
   - sentyment nie jest bezpośrednio używany w rankingu rekomendacji (to osobny moduł analityczny) — **jeśli tak jest, napisz to jasno**.

---

### 4.9. §9 Grupowanie — PRZEBUDOWAĆ

Zamiast powtarzać K-Means, napisz o **zastosowaniu w systemie:**

1. Formularz `/clustering/predict` — użytkownik podaje kilka ocen → przypisanie do klastra.
2. Strona `/clustering` — wizualizacja PCA, opisy klastrów.
3. Wpływ klastra na rekomendacje (cluster affinity w hybrydzie).
4. `UserClusteringEngine` w runtime — ładowanie modelu, standaryzacja, predykcja.

---

### 4.10. NOWY rozdział: Aplikacja internetowa (prawie nie ma w PDF!)

**To wymóg projektu z pierwszej strony sprawozdania.** Potrzebujesz **3–5 stron + zrzuty ekranu**.

#### Stack technologiczny

| Warstwa | Technologia |
|---------|-------------|
| Backend | FastAPI (Python) |
| Frontend | Jinja2 + HTMX (server-rendered) |
| Baza | SQLite (`joybookers.db`) |
| API docs | Swagger `/docs`, ReDoc `/redoc` |
| Modele ML | ładowane przy starcie (`app/startup.py`) |

#### Mapa funkcjonalności (strony webowe)

| URL | Funkcja |
|-----|---------|
| `/` | Strona główna |
| `/books` | Wyszukiwanie i filtrowanie książek |
| `/books/{id}` | Szczegóły książki |
| `/books/similar` | Podobne książki (content-based) |
| `/recommendations` | Generator rekomendacji |
| `/me/recommendations` | Rekomendacje dla zalogowanego użytkownika |
| `/sentiment` | Analiza sentymentu tekstu |
| `/clustering` | Wizualizacja klastrów + formularz predykcji |
| `/analytics` | Raporty EDA (wykresy z `reports/eda/`) |
| `/users` | Lista użytkowników |
| `/register`, `/login` | Autentykacja sesyjna |

#### Co opisać przy każdym zrzucie ekranu

1. **Co użytkownik widzi**
2. **Jaki model/analiza stoi za tym**
3. **Przykład użycia** (scenariusz: „użytkownik X szuka fantasy → dostaje listę Y”)

#### Minimalne zrzuty ekranu do wstawienia (6 szt.)

1. Strona główna
2. Lista rekomendacji z opisem profilu (cluster, wagi, algorytm)
3. Podobne książki
4. Analiza sentymentu z wynikiem
5. Klasteryzacja / PCA
6. Analytics / wykresy EDA

---

## 5. Rzeczy całkowicie pominięte w sprawozdaniu

Poniżej lista elementów **istniejących w repozytorium**, których **nie ma** w PDF — warto je opisać choćby krótko.

### 5.1. Architektura wielu zbiorów danych (DS1–DS4)
→ patrz §4.0b powyżej.

### 5.2. Model hybrydowy Ridge
→ patrz §4.7 — to najlepszy wynik projektu (RMSE 0,154)!

### 5.3. Genre priors i cold-start
Skrypty: `build_genre_priors.py`, `build_cluster_affinity.py`.  
Opisz: co robią, kiedy się aktywują.

### 5.4. Baza SQLite i warstwa aplikacyjna
- `scripts/load_db.py` — ładowanie książek i ocen do SQLite,
- limity: `--books-limit 20000 --ratings-limit 50000`,
- serwisy: `BookService`, `RatingService`, `RecommendationService`.

### 5.5. Retrenowanie CF z bazy (nowa funkcja)
Pliki: `app/ml/cf_retrain.py`, `cf_retrain_scheduler.py`, `scripts/retrain_cf_from_db.py`.  
Jeśli to działa — napisz akapit o **aktualizacji modelu** po nowych ocenach użytkowników aplikacji.

### 5.6. Dopasowywanie tytułów między zbiorami (`match_key`)
Moduł `bookrec/title_matching.py` — normalizacja tytułu + autora, join DS2 z DS3.

### 5.7. Raporty JSON jako artefakty pipeline'u
Folder `reports/` — wersjonowane metryki. Warto wspomnieć o **reprodukowalności** eksperymentów.

### 5.8. Skrypt `setup_all.py` — jedna komenda do całego pipeline'u
Pokaż, że projekt jest **automatyzowany**, nie ręczny.

### 5.9. Testy
`tests/test_cf_retrain.py` — krótka wzmianka o testach jednostkowych (nawet 2 zdania).

### 5.10. Ograniczenia i znane problemy
Szczerze opisz:
- niska precision@K dla SVD,
- DS4 ≠ Goodreads (sentyment „ogólny”),
- tylko część katalogu DS1 w SQLite,
- rekomendacje „trafione, ale nie zawsze na topie”,
- język interfejsu / danych (głównie angielski).

### 5.11. Podział pracy między autorów
Sprawozdanie zespołowe powinno mieć 3–5 zdań: kto co robił (pipeline, ML, frontend, dokumentacja).

### 5.12. Powiązanie z dokumentacją projektu
W repozytorium są `DOKUMENTACJA_PROJEKTU.md` i `ETAP_DANYCH_I_ML.md` — sprawozdanie PDF powinno być z nimi **spójne** (te same liczby, ta sama terminologia).

---

## 6. Gotowe fragmenty tekstu (do wklejenia)

### 6.1. Podsumowanie i wnioski (cały rozdział — DODAĆ na końcu)

> **Podsumowanie**
>
> W ramach pracowni EDI zrealizowano kompletny system rekomendacji książek obejmujący przygotowanie czterech zbiorów danych internetowych, eksploracyjną analizę ocen i katalogów, przetwarzanie języka naturalnego, segmentację użytkowników oraz hybrydowy silnik rekomendacji. Po stronie eksploracji danych oczyszczono ponad 1,5 mln rekordów katalogowych i 235 tys. interakcji użytkownik–książka, z zastosowaniem dopasowywania tytułów metodą fuzzy matching.
>
> Zaimplementowano trzy główne modele uczenia maszynowego: filtrację kolaboratywną (SVD, RMSE ≈ 0,89), rekomendacje oparte na treści (TF-IDF + cosinus) oraz klasyfikator sentymentu (accuracy ≈ 92%). Użytkowników podzielono na trzy klastry behawioralne (K-Means, Silhouette = 0,47). Najlepsze wyniki predykcji uzyskano dla modelu hybrydowego z wagami uczonego regresją Ridge (RMSE = 0,154).
>
> Aplikacja internetowa (FastAPI + HTMX) udostępnia wyszukiwanie książek, generowanie rekomendacji, analizę sentymentu, wizualizację klastrów oraz raporty EDA. System obsługuje scenariusze cold-start dzięki priorytom gatunkowym i mechanizmowi popularności.
>
> **Wnioski**
>
> 1. Łączenie wielu sygnałów (CF + content + klaster) znacząco poprawia jakość predykcji względem pojedynczych algorytmów.
> 2. Przy rzadkiej macierzy ocen (gęstość < 0,2%) metryki rankingowe (Precision@K) pozostają niskie mimo akceptowalnego RMSE — to znany problem w systemach rekomendacji.
> 3. Rozdzielenie ról zbiorów danych (DS1–DS4) pozwoliło efektywnie zarządzać skalą projektu bez budowania TF-IDF na pełnym katalogu 1,5 mln książek.
> 4. Analiza sentymentu na recenzjach Amazon jest użyteczna jako moduł analityczny, ale wymaga adaptacji do domeny Goodreads, jeśli miałaby wpływać na ranking.
> 5. Dalszy rozwój obejmowałby retrenowanie CF po nowych ocenach z aplikacji, rozszerzenie metryk ewaluacji (NDCG@K) oraz testy A/B w interfejsie użytkownika.

---

### 6.2. Krótki opis aplikacji webowej

> Aplikacja internetowa została zaimplementowana w frameworku FastAPI z interfejsem server-rendered (szablony Jinja2, interakcje HTMX). Dane operacyjne przechowywane są w bazie SQLite. Przy starcie serwera ładowane są wytrenowane modele ML: SVD, macierz TF-IDF, K-Means, pipeline sentymentu oraz wagi hybrydowe Ridge. Użytkownik może przeglądać katalog, wystawiać oceny, otrzymywać spersonalizowane rekomendacje, analizować sentyment własnego tekstu oraz eksplorować wizualizacje wynikające z analizy danych. Dokumentacja API dostępna jest pod adresem `/docs` (OpenAPI/Swagger).

---

### 6.3. Akapit o selekcji cech (§4)

> Selekcja cech przebiegała na dwóch poziomach. Dla segmentacji użytkowników (K-Means) wybrano siedem cech behawioralnych opisujących aktywność i profil oceniania: liczbę ocen, średnią, odchylenie standardowe, rozpiętość ocen oraz trzy binarne wskaźniki poziomu aktywności (niska/średnia/wysoka) wyznaczone z kwantyli 33% i 66%. Użytkownicy z mniej niż trzema ocenami zostali wykluczeni ze zbioru treningowego klasteryzacji. Dla rekomendacji content-based cechy tekstowe (opis, gatunki, tytuł, autor) przekształcono w sparse wektory TF-IDF z ograniczeniem słownika (`max_features`, `min_df`). W modelu hybrydowym jako cechy wejściowe wykorzystano znormalizowane sygnały: predykcję CF, podobieństwo content, affinitet klastrowy, priory gatunkowe i popularność — wagi tych sygnałów optymalizowano regresją Ridge.

---

### 6.4. Akapit o miarach podobieństwa (§5)

> W projekcie zastosowano dwie komplementarne miary podobieństwa. Dla reprezentacji tekstowej (TF-IDF) użyto podobieństwa cosinusowego, które po L2-normalizacji wektorów sprowadza się do iloczynu skalarnego — jest to standardowa praktyka w wyszukiwaniu dokumentów. Dla filtracji kolaboratywnej podobieństwo użytkowników i książek jest modelowane implicit w przestrzeni 50 czynników latentnych (dekompozycja SVD). Jakość grupowania użytkowników oceniano współczynnikiem Silhouette, osiągając wartość 0,47 dla k=3. Jakość sąsiedztwa content-based mierzono średnim cosinusem (0,63) oraz pokryciem gatunków w top-10 (0,71).

---

## 7. Bibliografia — propozycja pozycji

Dodaj rozdział **Bibliografia** (min. 8–12 pozycji). Format IEEE lub APA — ustal ze prowadzącą.

1. Aggarwal C. C., *Recommender Systems: The Textbook*, Springer, 2016.
2. Ricci F., Rokach L., Shapira B., *Recommender Systems Handbook*, Springer, 2022.
3. Koren Y., Bell R., Volinsky C., „Matrix Factorization Techniques for Recommender Systems”, *Computer*, 2009.
4. Hu Y., Koren Y., Volinsky C., „Collaborative Filtering for Implicit Feedback Datasets”, ICDM, 2008.
5. Manning C. D., Raghavan P., Schütze H., *Introduction to Information Retrieval*, Cambridge University Press, 2008. (TF-IDF, cosine)
6. Pedregosa F. et al., „Scikit-learn: Machine Learning in Python”, *JMLR*, 2011.
7. Hug N., „Surprise: A Python library for recommender systems”, JOSS, 2020.
8. Goodreads Datasets — źródło DS1/DS2/DS3 (podaj URL Kaggle / opis).
9. Amazon Product Reviews — źródło DS4.
10. FastAPI documentation — https://fastapi.tiangolo.com/
11. Rousseeuw P. J., „Silhouettes: A graphical aid to the interpretation and validation of cluster analysis”, 1987.
12. Dokumentacja projektu: `DOKUMENTACJA_PROJEKTU.md`, `ETAP_DANYCH_I_ML.md` (jeśli dopuszczalne jako załączniki).

---

## 8. Styl, język i formatowanie

### 8.1. Styl akademicki

| Unikać | Preferować |
|--------|------------|
| „Zrobiliśmy” | „Zrealizowano” / „W ramach prac…” |
| „Wykonaliśmy wizualizacje” | „Wygenerowano wizualizacje” |
| „Otrzymaliśmy 3 klastry” | „Algorytm wydzielił 3 klastry” |
| Ogólne „to jest dobre” | Konkretne liczby + interpretacja |

### 8.2. Kod źródłowy w sprawozdaniu

- **Maks. ½ strony** na jeden listing.
- Używaj `...` dla pominiętych fragmentów.
- Pod każdym listingiem: 2–4 zdania **wyjaśnienia**, nie powtórzenia kodu słowami.
- Numeruj listingi: „Listing 1. Funkcja czyszczenia ocen”.

### 8.3. Wykresy i tabele

- Każdy wykres: **podpis** (Rys. 1. Rozkład ocen użytkowników) + **1 akapit interpretacji**.
- Tabele metryk — preferowane nad długie akapity z liczbami.
- Rozdzielczość PNG min. 120 dpi (już macie w kodzie EDA).

### 8.4. Spójność nazewnictwa

Używaj konsekwentnie:
- „filtracja kolaboratywna” (nie raz CF, raz SVD bez wyjaśnienia),
- „rekomendacje oparte na treści” (content-based),
- „JoyBOOkers” / „system rekomendacji książek”.

### 8.5. Strona tytułowa

- Sprawdź daty: **18.05.2026 – 08.06.2026** — czy zgadza się z terminem oddania.
- Numer grupy PS: 1 — OK.

---

## 9. Checklista przed oddaniem

### Błędy i spójność
- [ ] Poprawiono RMSE (0,89, nie 89%)
- [ ] Zmieniono czas przyszły na przeszły we wstępie
- [ ] Usunięto / połączono duplikat klasteryzacji (§6 vs §9)
- [ ] Liczby w tekście = liczby w `reports/*.json`

### Struktura
- [ ] Dodano spis treści
- [ ] Dodano Wprowadzenie
- [ ] Dodano rozdział o zbiorach danych DS1–DS4
- [ ] Rozbudowano §4 Selekcja cech (min. 1,5 strony)
- [ ] Rozbudowano §5 Miary podobieństwa (min. 1 strona)
- [ ] Dodano opis modelu hybrydowego + tabela baseline'ów
- [ ] Dodano rozdział Aplikacja internetowa (min. 3 strony + screenshoty)
- [ ] Dodano Podsumowanie i Wnioski
- [ ] Dodano Bibliografię

### Zawartość merytoryczna
- [ ] Opisano architekturę systemu (diagram)
- [ ] Opisano cold-start i genre priors
- [ ] Opisano Ridge hybrid (najlepszy wynik!)
- [ ] Wspomniano o SQLite i pipeline `setup_all.py`
- [ ] Opisano ograniczenia projektu (szczerze)
- [ ] Podział pracy między autorów

### Kod i wykresy
- [ ] Skrócono długie listingi (max ~25 linii)
- [ ] Wstawiono min. 6 zrzutów ekranu aplikacji
- [ ] Wstawiono wykresy EDA z `reports/eda/`
- [ ] Wstawiono wykres PCA klastrów (jeśli dostępny)
- [ ] Każdy rysunek ma podpis i interpretację

### Formalia
- [ ] Spójny styl akademicki (bez „zrobiliśmy”)
- [ ] Numeracja rozdziałów i listingów
- [ ] Sprawdzona pisownia (polski)
- [ ] PDF czytelny — kod nie wychodzi poza marginesy
- [ ] Spójność z `DOKUMENTACJA_PROJEKTU.md`

---

## Załącznik: szybkie źródła liczb do sprawozdania

| Metryka | Wartość | Plik |
|---------|---------|------|
| Interakcje (clean) | 235 484 | `reports/data_pipeline/preprocess_summary.json` |
| Użytkownicy | 3 980 | jw. |
| Książki z ocenami | 48 920 | jw. |
| Gęstość macierzy | 0,12% | jw. |
| Średnia ocena | 3,80 | `reports/features/features_summary.json` |
| SVD RMSE | 0,890 | `reports/ml/evaluation/evaluate_all.json` |
| SVD MAE | 0,706 | jw. |
| Precision@10 | 0,0056 | jw. |
| Sentyment accuracy | 91,8% | jw. |
| Sentyment F1 macro | 85,0% | jw. |
| K-Means Silhouette (k=3) | 0,470 | `reports/ml/clustering/train_report.json` |
| Użytkownicy w K-Means | 3 311 | jw. |
| Content mean cosine | 0,633 | `evaluate_all.json` |
| Genre overlap@10 | 0,714 | jw. |
| Ridge hybrid RMSE | 0,154 | `reports/ml/evaluation/hybrid/baseline_comparison.json` |
| Książki w macierzy TF-IDF | 149 342 | `evaluate_all.json` |

---

*Plik wygenerowany na podstawie analizy `EDI_projekt_Alishkevich_Kulesza.pdf` oraz repozytorium JoyBOOkers. Ostatnia aktualizacja: czerwiec 2026.*
