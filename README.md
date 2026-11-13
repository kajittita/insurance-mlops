# Insurance charges – MLOps projekt

Féléves beadandó az **MLSecOps** tárgyhoz.
Egy egészségbiztosítási adathalmazon (életkor, nem, BMI, gyerekek száma, dohányzás, régió) gépi tanulási modell
jósolja meg az éves orvosi költséget (`charges`, **regresszió**), a modell köré pedig egy teljes MLOps folyamat épül:
verziókezelés, kísérletkövetés, model registry, REST API, konténerizálás, automatikus ellenőrzések és drift-monitorozás.

| | |
|---|---|
| **Szerző** | Takács Kitti (Neptun: XPLKVE) |
| **Dataset** | [Medical Cost Personal Dataset (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance) – licenc: [ODbL DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/) |
| **Célváltozó / feladat** | `charges` / regresszió |
| **Indítás** | `docker compose up --build` |

## Architektúra

```
                 ┌──────────────┐   dvc repro    ┌────────────────────┐
  data/raw  ───► │  prepare.py  │ ─────────────► │ data/processed/... │
  (DVC)          └──────────────┘                └─────────┬──────────┘
                                                           │
                       ┌───────────────────────────────────┤
                       ▼                                   ▼
               ┌──────────────┐  logs runs,        ┌──────────────┐
               │   trainer    │  registers model   │     api      │  POST /train, POST /predict
               │  (train.py)  │ ─────────────────► │  (FastAPI)   │ ◄──── kliens / simulate.py
               └──────────────┘        │           └──────┬───────┘
                                       ▼                  │ minden kérés + jóslat
                               ┌──────────────┐           ▼
                               │ MLflow server│    logs/predictions.csv  ──►  ┌────────────────┐
                               │ + registry   │                                │   streamlit    │
                               │ (@champion)  │       data/processed  ──────►  │ + Evidently    │
                               └──────────────┘                                │  drift report  │
                                                                               └────────────────┘
```

A `docker compose up` négy szolgáltatást indít:

| Szolgáltatás | Feladat | Cím |
|---|---|---|
| `mlflow` | kísérletkövetés és model registry (adatai egy volume-ban) | http://localhost:5000 |
| `trainer` | induláskor egyszer lefuttat egy tanítást, utána leáll | – |
| `api` | FastAPI: `/train`, `/predict`, `/health` | http://localhost:8000/docs |
| `streamlit` | monitoring dashboard az Evidently drift riporttal | http://localhost:8501 |

## Mappaszerkezet

```
├── src/insurance/          # az üzleti logika (minden itt van, ezt méri a coverage)
│   ├── config.py           # útvonalak, oszlopnevek, MLflow beállítások
│   ├── prepare.py          # előfeldolgozás (DVC stage)
│   ├── train.py            # tanítás, MLflow tracking, champion logika (DVC stage)
│   ├── api.py              # REST API
│   ├── monitoring.py       # Evidently drift riport
│   └── simulate.py         # forgalom küldése az API-nak (normál vagy eltolt)
├── dashboard/app.py        # Streamlit – csak megjelenítés
├── tests/                  # pytest tesztek (100% coverage)
├── data/                   # csak a DVC mutatófájlok vannak a Gitben
├── docs/screenshots/       # a futtatásról készült képek
├── dvc.yaml, dvc.lock      # a pipeline: prepare -> train
├── Dockerfile, compose.yaml
├── .github/workflows/ci.yml
├── .pre-commit-config.yaml
├── pyproject.toml          # pytest, coverage, pyright beállítások
└── requirements*.txt       # minden csomag pontos verzióval
```
