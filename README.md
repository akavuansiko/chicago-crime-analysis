# 🔍 Chicago Crime Analysis

## Problématique
> "Comment les crimes à Chicago se distribuent-ils dans l'espace et dans le temps,
> et peut-on identifier des patterns récurrents pour mieux anticiper leur évolution ?"

---

## 👥 Membres & Rôles

| Membre | Rôle | Tâches principales |
|---|---|---|
| **Angelikia** | Chef de projet & Data Exploration | Setup repo, load_data(), exploration complète, Top 10 crimes (groupby + Plotly) |
| **Ekta** | Pattern Mining & Analyse avancée | Apriori (mlxtend), règles d'association, Sankey diagram |
| **Léora** | Analyse temporelle & Forecasting | Time series, Facebook Prophet, prévision 12 mois |
| **Chrisa** | Analyse spatiale & Clustering | Geopandas, OPTICS/K-means, carte MapBox des hotspots |
| **Flavie** | Dashboard & Présentation | Python Dash, slides PDF, intégration finale |

---

## 📊 Source des données
[Chicago Crime Dataset — City of Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/bridgeport-crime-by-longitude-latitude-location/srg9-g5fv)

Données extraites du système CLEAR (Chicago Police Department) — incidents criminels géolocalisés avec date, type de crime, district, arrestation, latitude/longitude.

---

## 📁 Structure du projet

chicago-crime-analysis/

├── data/

│   └── chicago_crime.csv

├── notebooks/

│   ├── exploration.ipynb

│   ├── pattern_mining.ipynb

│   ├── forecasting.ipynb

│   └── spatial.ipynb

├── dashboard/

│   ├── app.py

│   └── kavuansiko_dashboard.html

├── slides/

│   └── presentation.pdf

└── README.md

---

## 🔬 Requêtes & Analyses

### Requête 1 — Data Exploration (Angelikia)
- Chargement avec `load_data()` + parsing des dates
- Exploration : shape, types, valeurs manquantes, ranges
- **Top 10 des types de crimes** par nombre d'incidents (groupby + Plotly)
- Somme des arrestations par district

### Requête 2 — Pattern Mining (Ekta)
- Discrétisation : heure → matin/après-midi/soir
- **Apriori** (mlxtend) : itemsets fréquents + règles d'association
- Visualisation **Sankey diagram** (Plotly) : antécédents → conséquents
- Analyse de l'impact du support minimal σ

### Requête 3 — Forecasting temporel (Léora)
- Agrégation crimes par mois/année → Pandas time series
- **Facebook Prophet** : prévision sur 12 mois (yhat / yhat_lower / yhat_upper)
- Composantes : tendance, saisonnalité, change points
- Optionnel : themeriver par type de crime

### Requête 4 — Analyse spatiale (Chrisa)
- Conversion lat/long → objets Point (Geopandas)
- **OPTICS / K-means** : détection de hotspots criminels
- Visualisation `scatter_mapbox` (Plotly) colorée par cluster
- Optionnel : enrichissement données météo ou population

---

## 🚀 Lancer le projet

### Prérequis
```bash
pip install pandas plotly geopandas scikit-learn mlxtend prophet dash jupyter
```

### Exécuter les notebooks
```bash
jupyter notebook notebooks/
```

### Lancer le dashboard
```bash
python dashboard/app.py
```

---

## 🗓️ Présentation orale
**17 juin 2025** — Présentation groupe + démo solo (questions sur le code)

Structure des slides : Contexte → Dataset → 4 Requêtes → Conclusions → Démo
