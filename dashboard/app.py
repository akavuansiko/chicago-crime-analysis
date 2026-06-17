"""Dashboard interactif - Chicago Crime Analysis.

Version mise à jour : la source principale n'est plus le CSV local Bridgeport.
Le dashboard charge désormais les données depuis le portail officiel City of Chicago
(Crimes - 2001 to Present) et utilise une agrégation temporelle citywide pour
l'analyse temporelle et le forecasting.

Lancement du dashboard depuis la racine du projet :
    python dashboard/app.py

Création d'une preuve HTML autonome :
    python dashboard/app.py --export-html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, callback, dash_table, dcc, html
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.cluster import KMeans, OPTICS

try:
    from prophet import Prophet
except Exception:  # Prophet peut échouer selon l'environnement local.
    Prophet = None


# -----------------------------------------------------------------------------
# Configuration générale
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Ancien CSV local conservé seulement comme secours si Internet est indisponible.
LOCAL_FALLBACK_PATH = DATA_DIR / "chicago_crime.csv"
CITYWIDE_CACHE_PATH = DATA_DIR / "chicago_crimes_citywide_cache.csv"
MONTHLY_CACHE_PATH = DATA_DIR / "chicago_monthly_citywide_cache.csv"
EXPORT_PATH = BASE_DIR / "Kavuansiko_dashboard.html"

DATASET_NAME = "Chicago Crimes - 2001 to Present"
DATASET_PAGE_URL = "https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2"
SODA_ENDPOINT = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"
DEFAULT_DATA_LIMIT = 50_000
MAX_SPATIAL_POINTS = 5_000

TEAM_MEMBERS = [
    "Angelikia Kavuansiko - exploration des données",
    "Ekta - pattern mining",
    "Léora - analyse temporelle",
    "Chrisa - analyse spatiale",
    "Flavie - dashboard et intégration finale",
]

COLUMN_RENAME = {
    "id": "ID",
    "case_number": "Case Number",
    "date": "Date",
    "block": "Block",
    "iucr": "IUCR",
    "primary_type": "Primary Type",
    "description": "Description",
    "location_description": "Location Description",
    "arrest": "Arrest",
    "domestic": "Domestic",
    "beat": "Beat",
    "district": "District",
    "ward": "Ward",
    "community_area": "Community Area",
    "fbi_code": "FBI Code",
    "x_coordinate": "X Coordinate",
    "y_coordinate": "Y Coordinate",
    "year": "Year",
    "updated_on": "Updated On",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "location": "Location",
}

PAGE_STYLE = {
    "fontFamily": "Arial, Helvetica, sans-serif",
    "backgroundColor": "#f4f6f8",
    "minHeight": "100vh",
    "color": "#17202a",
}
CARD_STYLE = {
    "backgroundColor": "white",
    "borderRadius": "14px",
    "padding": "18px",
    "boxShadow": "0 3px 14px rgba(0,0,0,0.08)",
}


# -----------------------------------------------------------------------------
# Accès aux données City of Chicago
# -----------------------------------------------------------------------------
def build_citywide_api_url(limit: int = DEFAULT_DATA_LIMIT) -> str:
    """Construire l'URL CSV de l'API Socrata pour les incidents récents citywide.

    Input : nombre maximal de lignes à charger.
    Output : URL complète interrogeant le dataset officiel Crimes - 2001 to Present.
    """
    params = {
        "$limit": int(limit),
        "$order": "date DESC",
    }
    return f"{SODA_ENDPOINT}?{urlencode(params)}"


def build_monthly_api_url() -> str:
    """Construire l'URL d'agrégation mensuelle citywide.

    Input : aucun.
    Output : URL SoQL qui compte les crimes par année et par mois pour toute la ville.
    """
    params = {
        "$select": "date_extract_y(date) AS year, date_extract_m(date) AS month, count(*) AS count",
        "$group": "year, month",
        "$order": "year, month",
        "$limit": 5000,
    }
    return f"{SODA_ENDPOINT}?{urlencode(params)}"


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Harmoniser les noms de colonnes entre le CSV local et l'API Socrata.

    Input : dataframe brut.
    Output : dataframe avec les noms attendus par le reste du dashboard.
    """
    result = df.rename(columns={col: COLUMN_RENAME.get(col, col) for col in df.columns})

    required_columns = [
        "ID", "Date", "Primary Type", "Location Description", "Arrest", "Domestic",
        "Latitude", "Longitude", "Year",
    ]
    for column in required_columns:
        if column not in result.columns:
            result[column] = pd.NA

    return result


def _parse_bool_series(series: pd.Series) -> pd.Series:
    """Convertir les booléens Socrata/Pandas en valeurs True/False fiables.

    Input : série contenant True/False, true/false, 1/0 ou valeurs manquantes.
    Output : série booléenne exploitable pour les taux et Apriori.
    """
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False)
        .astype(bool)
    )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyer et enrichir les données criminelles.

    Input : dataframe brut issu de l'API ou du CSV local.
    Output : dataframe standardisé avec Date, coordonnées numériques et variables temporelles.
    """
    result = _standardise_columns(df).copy()

    result["Date"] = pd.to_datetime(result["Date"], errors="coerce", format="mixed")
    result["Updated On"] = pd.to_datetime(result.get("Updated On"), errors="coerce", format="mixed")

    for column in ["Latitude", "Longitude"]:
        result[column] = pd.to_numeric(
            result[column].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    result["Arrest"] = _parse_bool_series(result["Arrest"])
    result["Domestic"] = _parse_bool_series(result["Domestic"])

    result["Primary Type"] = result["Primary Type"].fillna("UNKNOWN")
    result["Location Description"] = result["Location Description"].fillna("UNKNOWN")

    result["Year"] = result["Date"].dt.year.astype("Int64")
    result["Month"] = result["Date"].dt.month.astype("Int64")
    result["Hour"] = result["Date"].dt.hour.astype("Int64")
    result["YearMonth"] = result["Date"].dt.to_period("M").dt.to_timestamp()

    return result


def load_data(limit: int = DEFAULT_DATA_LIMIT, offline: bool = False) -> pd.DataFrame:
    """Charger les incidents citywide depuis l'API officielle, avec cache local.

    Input : nombre de lignes à récupérer et mode hors ligne optionnel.
    Output : dataframe nettoyé pour les analyses du dashboard.
    """
    if offline and CITYWIDE_CACHE_PATH.exists():
        return prepare_dataframe(pd.read_csv(CITYWIDE_CACHE_PATH))

    url = build_citywide_api_url(limit)
    try:
        df = pd.read_csv(url)
        df.to_csv(CITYWIDE_CACHE_PATH, index=False)
        return prepare_dataframe(df)
    except Exception as error:
        print(
            "Avertissement : impossible de charger la source en ligne City of Chicago. "
            f"Détail : {error}"
        )
        if CITYWIDE_CACHE_PATH.exists():
            print(f"Utilisation du cache local : {CITYWIDE_CACHE_PATH}")
            return prepare_dataframe(pd.read_csv(CITYWIDE_CACHE_PATH))
        if LOCAL_FALLBACK_PATH.exists():
            print(
                "Utilisation du CSV local de secours. Attention : ce fichier peut être "
                "plus restreint que la source citywide."
            )
            return prepare_dataframe(pd.read_csv(LOCAL_FALLBACK_PATH))
        raise FileNotFoundError(
            "Aucune source disponible : ni API City of Chicago, ni cache local, ni CSV de secours."
        ) from error


def load_monthly_citywide_counts(offline: bool = False) -> pd.DataFrame:
    """Charger l'agrégation mensuelle citywide depuis l'API Socrata.

    Input : mode hors ligne optionnel.
    Output : dataframe mensuel avec Date et Nombre de crimes couvrant toute la ville.
    """
    if offline and MONTHLY_CACHE_PATH.exists():
        return pd.read_csv(MONTHLY_CACHE_PATH, parse_dates=["Date"])

    try:
        monthly_raw = pd.read_csv(build_monthly_api_url())
        monthly_raw.columns = [str(c).lower() for c in monthly_raw.columns]
        monthly = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    monthly_raw["year"].astype(int).astype(str)
                    + "-"
                    + monthly_raw["month"].astype(int).astype(str).str.zfill(2)
                    + "-01",
                    errors="coerce",
                ),
                "Nombre de crimes": pd.to_numeric(monthly_raw["count"], errors="coerce").fillna(0),
            }
        )
        monthly = monthly.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        monthly.to_csv(MONTHLY_CACHE_PATH, index=False)
        return monthly
    except Exception as error:
        print(
            "Avertissement : l'agrégation mensuelle citywide n'a pas pu être chargée. "
            f"Détail : {error}"
        )
        if MONTHLY_CACHE_PATH.exists():
            return pd.read_csv(MONTHLY_CACHE_PATH, parse_dates=["Date"])
        raise


# -----------------------------------------------------------------------------
# Filtres et KPIs
# -----------------------------------------------------------------------------
def filter_data(df: pd.DataFrame, year: str | int, crime_type: str) -> pd.DataFrame:
    """Appliquer les filtres sélectionnés dans le dashboard.

    Input : dataframe, année et type de crime.
    Output : sous-ensemble filtré du dataframe.
    """
    result = df.copy()
    if year != "ALL":
        result = result[result["Year"] == int(year)]
    if crime_type != "ALL":
        result = result[result["Primary Type"] == crime_type]
    return result


def calculate_kpis(df: pd.DataFrame) -> dict[str, str]:
    """Calculer les indicateurs synthétiques affichés en tête du dashboard.

    Input : dataframe filtré.
    Output : dictionnaire de valeurs formatées.
    """
    if df.empty:
        return {
            "incidents": "0",
            "arrest_rate": "0,0 %",
            "main_crime": "Aucun",
            "period": "Aucune donnée",
        }

    arrest_rate = float(df["Arrest"].mean() * 100)
    main_crime = str(df["Primary Type"].mode().iloc[0])
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    period = f"{min_date:%m/%Y} - {max_date:%m/%Y}" if pd.notna(min_date) else "Aucune date"

    return {
        "incidents": f"{len(df):,}".replace(",", " "),
        "arrest_rate": f"{arrest_rate:.1f} %".replace(".", ","),
        "main_crime": main_crime,
        "period": period,
    }


# -----------------------------------------------------------------------------
# Requête 1 - agrégations descriptives
# -----------------------------------------------------------------------------
def make_top_crimes_figure(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Créer le classement des types de crimes les plus fréquents.

    Input : dataframe et nombre de catégories à afficher.
    Output : figure Plotly en barres horizontales.
    """
    if df.empty:
        return empty_figure("Aucune donnée pour les filtres sélectionnés")

    counts = (
        df["Primary Type"]
        .value_counts()
        .head(top_n)
        .sort_values()
        .rename_axis("Type de crime")
        .reset_index(name="Nombre d'incidents")
    )

    fig = px.bar(
        counts,
        x="Nombre d'incidents",
        y="Type de crime",
        orientation="h",
        text="Nombre d'incidents",
        title=f"Top {min(top_n, len(counts))} des types de crimes",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(l=20, r=30, t=65, b=20), height=460)
    return fig


def make_arrest_rate_figure(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Calculer le taux d'arrestation des crimes les plus représentés.

    Input : dataframe et nombre de catégories à afficher.
    Output : figure Plotly avec un taux exprimé en pourcentage.
    """
    if df.empty:
        return empty_figure("Aucune donnée pour les filtres sélectionnés")

    grouped = (
        df.groupby("Primary Type", observed=True)
        .agg(Incidents=("ID", "count"), Taux_arrestation=("Arrest", "mean"))
        .sort_values("Incidents", ascending=False)
        .head(top_n)
        .reset_index()
    )
    grouped["Taux d'arrestation (%)"] = grouped["Taux_arrestation"] * 100
    grouped = grouped.sort_values("Taux d'arrestation (%)")

    fig = px.bar(
        grouped,
        x="Taux d'arrestation (%)",
        y="Primary Type",
        orientation="h",
        text="Taux d'arrestation (%)",
        hover_data={"Incidents": True, "Taux_arrestation": False},
        title="Taux d'arrestation parmi les crimes les plus fréquents",
        labels={"Primary Type": "Type de crime"},
    )
    fig.update_traces(texttemplate="%{text:.1f} %", textposition="outside")
    fig.update_xaxes(range=[0, max(100, grouped["Taux d'arrestation (%)"].max() * 1.18)])
    fig.update_layout(margin=dict(l=20, r=30, t=65, b=20), height=460)
    return fig


# -----------------------------------------------------------------------------
# Requête 2 - Apriori et règles d'association
# -----------------------------------------------------------------------------
def discretise_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transformer les variables en catégories utilisables par Apriori.

    Input : dataframe préparé.
    Output : dataframe discrétisé avec type, tranche horaire, lieu, arrestation et domesticité.
    """
    result = df.copy()

    def time_slot(hour: int | float) -> str:
        if pd.isna(hour):
            return "Heure_inconnue"
        hour = int(hour)
        if 6 <= hour < 12:
            return "Matin"
        if 12 <= hour < 18:
            return "Après-midi"
        if 18 <= hour < 23:
            return "Soir"
        return "Nuit"

    top_locations = result["Location Description"].value_counts().nlargest(5).index
    result["Heure"] = result["Hour"].apply(time_slot)
    result["Lieu"] = result["Location Description"].apply(
        lambda value: value if value in top_locations else "AUTRE"
    )
    result["Arrestation"] = np.where(result["Arrest"], "Arrestation_OUI", "Arrestation_NON")
    result["Domestique"] = np.where(result["Domestic"], "Domestique_OUI", "Domestique_NON")

    return result[["Primary Type", "Heure", "Lieu", "Arrestation", "Domestique"]]


def encode_transactions(discrete_df: pd.DataFrame) -> pd.DataFrame:
    """Encoder les catégories sous forme booléenne pour Apriori.

    Input : dataframe discrétisé.
    Output : matrice booléenne obtenue avec TransactionEncoder.
    """
    transactions = discrete_df.apply(lambda row: list(row.values), axis=1).tolist()
    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(transactions)
    return pd.DataFrame(encoded_array, columns=encoder.columns_)


def mine_association_rules(
    encoded_df: pd.DataFrame, min_support: float = 0.10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extraire les itemsets fréquents puis les règles d'association.

    Input : matrice booléenne et support minimal.
    Output : itemsets fréquents et règles dont le lift est au moins égal à 1.
    """
    itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True)
    if itemsets.empty:
        return itemsets, pd.DataFrame()

    rules = association_rules(itemsets, metric="lift", min_threshold=1.0)
    if not rules.empty:
        rules = rules.sort_values(["lift", "confidence"], ascending=False)
    return itemsets, rules


def _format_itemset(items: Iterable[str]) -> str:
    """Transformer un ensemble d'items en libellé lisible."""
    return " + ".join(sorted(str(item).replace("_", " ") for item in items))


def make_support_curve(encoded_df: pd.DataFrame) -> go.Figure:
    """Montrer l'effet du support minimal sur le nombre de motifs retenus.

    Input : matrice booléenne utilisée par Apriori.
    Output : courbe support minimal / nombre d'itemsets.
    """
    supports = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    rows = []
    for support in supports:
        count = len(apriori(encoded_df, min_support=support, use_colnames=True))
        rows.append({"Support minimal": support, "Nombre d'itemsets": count})

    support_df = pd.DataFrame(rows)
    fig = px.line(
        support_df,
        x="Support minimal",
        y="Nombre d'itemsets",
        markers=True,
        title="Sensibilité du nombre de motifs au support minimal",
    )
    fig.add_vline(x=0.10, line_dash="dash", annotation_text="σ retenu = 0,10")
    fig.update_layout(margin=dict(l=20, r=20, t=65, b=20), height=430)
    return fig


def make_sankey_rules(rules: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Créer un diagramme de Sankey des principales règles d'association.

    Input : règles Apriori et nombre maximal de règles.
    Output : figure Sankey où l'épaisseur représente la confiance.
    """
    if rules.empty:
        return empty_figure("Aucune règle d'association au seuil choisi")

    top_rules = rules.head(top_n).copy()
    top_rules["Antecedent"] = top_rules["antecedents"].apply(_format_itemset)
    top_rules["Consequent"] = top_rules["consequents"].apply(_format_itemset)

    left_labels = list(dict.fromkeys(top_rules["Antecedent"].tolist()))
    right_labels = list(dict.fromkeys(top_rules["Consequent"].tolist()))
    labels = left_labels + [label for label in right_labels if label not in left_labels]
    index = {label: position for position, label in enumerate(labels)}

    fig = go.Figure(
        go.Sankey(
            node=dict(label=labels, pad=15, thickness=16),
            link=dict(
                source=[index[value] for value in top_rules["Antecedent"]],
                target=[index[value] for value in top_rules["Consequent"]],
                value=top_rules["confidence"].tolist(),
                customdata=np.column_stack(
                    [top_rules["lift"].round(2), top_rules["support"].round(3)]
                ),
                hovertemplate=(
                    "Confiance : %{value:.2f}<br>Lift : %{customdata[0]:.2f}"
                    "<br>Support : %{customdata[1]:.3f}<extra></extra>"
                ),
            ),
        )
    )
    fig.update_layout(
        title="Principales règles d'association - antécédent vers conséquent",
        margin=dict(l=20, r=20, t=70, b=20),
        height=560,
    )
    return fig


def rules_table_dataframe(rules: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Préparer un tableau lisible des meilleures règles Apriori.

    Input : dataframe de règles.
    Output : tableau avec antécédent, conséquent, support, confiance et lift.
    """
    if rules.empty:
        return pd.DataFrame(
            columns=["Antécédent", "Conséquent", "Support", "Confiance", "Lift"]
        )

    table = rules.head(top_n).copy()
    table["Antécédent"] = table["antecedents"].apply(_format_itemset)
    table["Conséquent"] = table["consequents"].apply(_format_itemset)
    table["Support"] = table["support"].round(3)
    table["Confiance"] = table["confidence"].round(3)
    table["Lift"] = table["lift"].round(2)
    return table[["Antécédent", "Conséquent", "Support", "Confiance", "Lift"]]


# -----------------------------------------------------------------------------
# Requête 3 - dimension temporelle citywide et forecasting
# -----------------------------------------------------------------------------
def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Agréger les incidents par mois à partir des données chargées localement.

    Input : dataframe préparé.
    Output : série mensuelle avec Date et Nombre de crimes.
    """
    valid = df.dropna(subset=["Date"]).set_index("Date")
    monthly = valid.resample("MS").size().rename("Nombre de crimes").reset_index()
    return monthly


def make_monthly_figure(monthly: pd.DataFrame) -> go.Figure:
    """Visualiser l'évolution mensuelle citywide du nombre d'incidents.

    Input : agrégation mensuelle.
    Output : courbe temporelle Plotly.
    """
    fig = px.line(
        monthly,
        x="Date",
        y="Nombre de crimes",
        title="Évolution mensuelle des incidents - ville de Chicago",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=65, b=20), height=430)
    return fig


def run_prophet_forecast(
    monthly: pd.DataFrame, periods: int = 12
) -> tuple[object | None, pd.DataFrame]:
    """Entraîner Prophet et prévoir les douze prochains mois.

    Input : série mensuelle citywide et horizon de prévision.
    Output : modèle entraîné et dataframe des valeurs prévues.

    Une régression tendance-saisonnalité est utilisée comme solution de secours si
    Prophet ou Stan n'est pas compatible avec l'environnement local.
    """
    prophet_df = monthly.rename(columns={"Date": "ds", "Nombre de crimes": "y"})
    prophet_df = prophet_df.dropna(subset=["ds", "y"])

    try:
        if Prophet is None:
            raise RuntimeError("Prophet n'est pas disponible dans cet environnement.")
        if sys.version_info >= (3, 13):
            raise RuntimeError(
                "Prophet est désactivé sous Python 3.13 ; utilisez Python 3.10 à 3.12 "
                "ou conservez la prévision de secours."
            )
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.80,
        )
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods, freq="MS")
        forecast = model.predict(future)
        forecast.attrs["model_name"] = "Prophet"
    except Exception as error:
        print(
            "Avertissement : Prophet n'a pas pu être exécuté. "
            "Utilisation de la prévision saisonnière de secours. "
            f"Détail : {error}"
        )
        model = None

        n_observed = len(prophet_df)
        total_length = n_observed + periods
        t_observed = np.arange(n_observed, dtype=float)
        t_all = np.arange(total_length, dtype=float)

        def design_matrix(t_values: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [
                    np.ones(len(t_values)),
                    t_values,
                    np.sin(2 * np.pi * t_values / 12),
                    np.cos(2 * np.pi * t_values / 12),
                    np.sin(4 * np.pi * t_values / 12),
                    np.cos(4 * np.pi * t_values / 12),
                ]
            )

        x_observed = design_matrix(t_observed)
        coefficients, *_ = np.linalg.lstsq(
            x_observed, prophet_df["y"].to_numpy(dtype=float), rcond=None
        )
        fitted = x_observed @ coefficients
        predicted = design_matrix(t_all) @ coefficients
        residual_std = float(np.std(prophet_df["y"].to_numpy(dtype=float) - fitted, ddof=1))
        margin = 1.282 * residual_std

        future_dates = pd.date_range(start=prophet_df["ds"].min(), periods=total_length, freq="MS")
        forecast = pd.DataFrame(
            {
                "ds": future_dates,
                "yhat": predicted,
                "yhat_lower": predicted - margin,
                "yhat_upper": predicted + margin,
            }
        )
        forecast.attrs["model_name"] = "modèle saisonnier de secours"

    for column in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[column] = forecast[column].clip(lower=0)
    return model, forecast


def make_forecast_figure(monthly: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """Comparer les observations et la prévision.

    Input : série observée citywide et dataframe de prévision.
    Output : figure avec prévision et intervalle d'incertitude.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["Date"], y=monthly["Nombre de crimes"], mode="lines", name="Données observées"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", fill="tonexty", name="Intervalle de prévision à 80 %", line=dict(width=0)))
    model_name = forecast.attrs.get("model_name", "Prophet")
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name=f"Prévision - {model_name}"))
    fig.update_layout(
        title=f"Prévision mensuelle des incidents sur 12 mois - Chicago citywide ({model_name})",
        xaxis_title="Date",
        yaxis_title="Nombre de crimes",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=65, b=20),
        height=470,
    )
    return fig


# -----------------------------------------------------------------------------
# Requête 4 - dimension spatiale et clustering
# -----------------------------------------------------------------------------
def spatial_subset(df: pd.DataFrame, max_points: int = MAX_SPATIAL_POINTS) -> pd.DataFrame:
    """Conserver les incidents géolocalisables et échantillonner si nécessaire.

    Input : dataframe préparé et nombre maximal de points pour les cartes.
    Output : dataframe géolocalisable, éventuellement échantillonné pour préserver l'interactivité.
    """
    spatial = df.dropna(subset=["Latitude", "Longitude"]).copy()
    if len(spatial) > max_points:
        spatial = spatial.sample(max_points, random_state=42)
    return spatial


def apply_kmeans(df: pd.DataFrame, clusters: int = 6) -> pd.DataFrame:
    """Regrouper les incidents géolocalisés en zones avec K-means.

    Input : dataframe spatial et nombre maximal de clusters.
    Output : copie du dataframe avec une colonne Cluster.
    """
    result = spatial_subset(df)
    if result.empty:
        result["Cluster"] = pd.Series(dtype=str)
        return result

    n_clusters = min(clusters, len(result))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    result["Cluster"] = model.fit_predict(result[["Latitude", "Longitude"]]).astype(str)
    return result


def _map_center(spatial: pd.DataFrame) -> dict[str, float]:
    """Calculer le centre d'une carte à partir des coordonnées disponibles."""
    return {"lat": float(spatial["Latitude"].mean()), "lon": float(spatial["Longitude"].mean())}


def make_density_map(df: pd.DataFrame) -> go.Figure:
    """Visualiser la concentration géographique des incidents.

    Input : dataframe éventuellement filtré.
    Output : carte de densité Plotly fondée sur OpenStreetMap.
    """
    spatial = spatial_subset(df)
    if spatial.empty:
        return empty_figure("Aucune coordonnée disponible")

    fig = px.density_mapbox(
        spatial,
        lat="Latitude",
        lon="Longitude",
        radius=10,
        zoom=10,
        center=_map_center(spatial),
        mapbox_style="open-street-map",
        hover_name="Primary Type",
        hover_data={"Date": True, "Location Description": True},
        title="Densité géographique des incidents - ville de Chicago",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), height=560)
    return fig


def make_kmeans_map(df: pd.DataFrame, clusters: int = 6) -> go.Figure:
    """Afficher les zones obtenues par K-means.

    Input : dataframe éventuellement filtré et nombre de clusters.
    Output : carte Plotly avec une couleur par cluster.
    """
    clustered = apply_kmeans(df, clusters=clusters)
    if clustered.empty:
        return empty_figure("Aucune coordonnée disponible")

    fig = px.scatter_mapbox(
        clustered,
        lat="Latitude",
        lon="Longitude",
        color="Cluster",
        zoom=10,
        center=_map_center(clustered),
        mapbox_style="open-street-map",
        hover_name="Primary Type",
        hover_data=["Date", "Location Description", "Arrest"],
        title=f"Clustering K-means - {clustered['Cluster'].nunique()} zones citywide",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.70))
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), height=560)
    return fig


def apply_optics(df: pd.DataFrame, min_samples: int = 20) -> pd.DataFrame:
    """Détecter les zones denses et les incidents isolés avec OPTICS.

    Input : dataframe spatial et densité minimale.
    Output : copie du dataframe avec une colonne Cluster_OPTICS ; -1 désigne le bruit.
    """
    result = spatial_subset(df, max_points=3_000)
    if len(result) < 5:
        result["Cluster_OPTICS"] = pd.Series(dtype=str)
        return result

    effective_min_samples = min(min_samples, max(2, len(result) // 5))
    model = OPTICS(min_samples=effective_min_samples, xi=0.05, min_cluster_size=0.05)
    result["Cluster_OPTICS"] = model.fit_predict(result[["Latitude", "Longitude"]]).astype(str)
    return result


def make_optics_map(df: pd.DataFrame) -> go.Figure:
    """Afficher les hotspots OPTICS et les observations isolées.

    Input : dataframe éventuellement filtré.
    Output : carte Plotly ; le label -1 correspond aux incidents considérés comme bruit.
    """
    clustered = apply_optics(df)
    if clustered.empty:
        return empty_figure("Pas assez de coordonnées pour appliquer OPTICS")

    dense_labels = [value for value in clustered["Cluster_OPTICS"].unique() if value != "-1"]
    isolated = int((clustered["Cluster_OPTICS"] == "-1").sum())
    fig = px.scatter_mapbox(
        clustered,
        lat="Latitude",
        lon="Longitude",
        color="Cluster_OPTICS",
        zoom=10,
        center=_map_center(clustered),
        mapbox_style="open-street-map",
        hover_name="Primary Type",
        hover_data=["Date", "Location Description", "Arrest"],
        title=f"Clustering OPTICS - {len(dense_labels)} zones denses, {isolated} incidents isolés",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.70))
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), height=560)
    return fig


# -----------------------------------------------------------------------------
# Composants communs et interface Dash
# -----------------------------------------------------------------------------
def empty_figure(message: str) -> go.Figure:
    """Créer une figure vide accompagnée d'un message explicatif."""
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=18))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def indicator_explanation(title: str, calculation: str, meaning: str, interpretation: str) -> html.Div:
    """Créer le bloc de définition exigé par le sujet pour chaque indicateur."""
    return html.Div(
        [
            html.H4(title, style={"marginTop": "0"}),
            html.P([html.Strong("Calcul : "), calculation]),
            html.P([html.Strong("Ce que l'indicateur représente : "), meaning]),
            html.P([html.Strong("Interprétation : "), interpretation]),
        ],
        style={**CARD_STYLE, "borderLeft": "5px solid #34495e", "marginBottom": "16px", "lineHeight": "1.45"},
    )


def kpi_card(label: str, value_id: str) -> html.Div:
    """Créer une carte KPI dont la valeur sera mise à jour par callback."""
    return html.Div(
        [
            html.Div(label, style={"fontSize": "14px", "color": "#5d6d7e"}),
            html.Div(id=value_id, style={"fontSize": "27px", "fontWeight": "700", "marginTop": "8px"}),
        ],
        style={**CARD_STYLE, "flex": "1", "minWidth": "190px"},
    )


def create_app(limit: int = DEFAULT_DATA_LIMIT, offline: bool = False) -> Dash:
    """Construire l'application Dash et pré-calculer les analyses lourdes.

    Input : nombre de lignes récentes à charger pour les vues interactives et mode hors ligne.
    Output : instance Dash prête à être lancée.
    """
    df = load_data(limit=limit, offline=offline)

    # Les analyses descriptives, spatiales et Apriori utilisent l'échantillon récent citywide.
    discrete_df = discretise_data(df)
    encoded_df = encode_transactions(discrete_df)
    itemsets, rules = mine_association_rules(encoded_df, min_support=0.10)
    support_fig = make_support_curve(encoded_df)
    sankey_fig = make_sankey_rules(rules)
    rules_table = rules_table_dataframe(rules)

    # L'analyse temporelle utilise une agrégation mensuelle citywide indépendante du limit.
    try:
        monthly = load_monthly_citywide_counts(offline=offline)
        temporal_scope = "agrégation mensuelle citywide complète depuis l'API City of Chicago"
    except Exception:
        monthly = aggregate_monthly(df)
        temporal_scope = "agrégation mensuelle calculée depuis les lignes chargées localement"

    _, forecast = run_prophet_forecast(monthly, periods=12)
    monthly_fig = make_monthly_figure(monthly)
    forecast_fig = make_forecast_figure(monthly, forecast)

    years = sorted(int(year) for year in df["Year"].dropna().unique())
    crime_types = sorted(str(value) for value in df["Primary Type"].dropna().unique())

    app = Dash(__name__)
    app.title = "Chicago Crime Analysis"

    app.layout = html.Div(
        [
            html.Header(
                [
                    html.H1("Chicago Crime Analysis", style={"marginBottom": "6px"}),
                    html.P(
                        "Prototype de visualisation des incidents criminels enregistrés dans la ville de Chicago",
                        style={"marginTop": "0", "fontSize": "17px"},
                    ),
                    html.P(
                        f"Dataset : {DATASET_NAME} | {len(df):,} lignes récentes chargées pour les vues interactives | {len(df.columns)} variables".replace(",", " "),
                        style={"marginBottom": "8px"},
                    ),
                    html.P(f"Source : {DATASET_PAGE_URL}", style={"fontSize": "13px", "marginBottom": "8px"}),
                    html.P("Équipe : " + " • ".join(TEAM_MEMBERS), style={"fontSize": "13px"}),
                ],
                style={"padding": "28px 5%", "backgroundColor": "#17202a", "color": "white"},
            ),
            html.Main(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Année", style={"fontWeight": "700"}),
                                    dcc.Dropdown(
                                        id="year-filter",
                                        options=[{"label": "Toutes les années", "value": "ALL"}]
                                        + [{"label": str(year), "value": year} for year in years],
                                        value="ALL",
                                        clearable=False,
                                    ),
                                ],
                                style={"flex": "1", "minWidth": "220px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Type de crime", style={"fontWeight": "700"}),
                                    dcc.Dropdown(
                                        id="crime-filter",
                                        options=[{"label": "Tous les types", "value": "ALL"}]
                                        + [{"label": value, "value": value} for value in crime_types],
                                        value="ALL",
                                        clearable=False,
                                    ),
                                ],
                                style={"flex": "2", "minWidth": "280px"},
                            ),
                        ],
                        style={**CARD_STYLE, "display": "flex", "gap": "20px", "flexWrap": "wrap"},
                    ),
                    html.Div(
                        [
                            kpi_card("Nombre d'incidents", "kpi-incidents"),
                            kpi_card("Taux d'arrestation", "kpi-arrest"),
                            kpi_card("Crime dominant", "kpi-main-crime"),
                            kpi_card("Période observée", "kpi-period"),
                        ],
                        style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "margin": "18px 0"},
                    ),
                    html.Div(
                        [
                            html.Strong("Note méthodologique : "),
                            "les onglets exploration, pattern mining et spatial s'appuient sur les lignes récentes chargées depuis l'API pour conserver un dashboard fluide. ",
                            "L'onglet temporel utilise une agrégation mensuelle citywide issue directement de l'API, afin de ne plus limiter la série temporelle au secteur Bridgeport.",
                        ],
                        style={**CARD_STYLE, "marginBottom": "18px", "lineHeight": "1.45"},
                    ),
                    dcc.Tabs(
                        [
                            dcc.Tab(
                                label="1. Exploration",
                                children=[
                                    indicator_explanation(
                                        "Fréquence des types de crimes",
                                        "Comptage des incidents après regroupement par Primary Type.",
                                        "Le volume observé pour chaque catégorie de crime dans les données chargées depuis l'API.",
                                        "Une barre plus longue correspond à un type de crime plus fréquent dans l'échantillon citywide chargé.",
                                    ),
                                    html.Div(
                                        [dcc.Graph(id="top-crimes-graph", style={"flex": "1"}), dcc.Graph(id="arrest-rate-graph", style={"flex": "1"})],
                                        style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="2. Pattern mining",
                                children=[
                                    indicator_explanation(
                                        "Règles d'association Apriori",
                                        f"Encodage booléen des incidents puis Apriori avec un support minimal de 0,10. {len(itemsets)} itemsets et {len(rules)} règles ont été obtenus.",
                                        "Les associations récurrentes entre tranche horaire, lieu, type de crime, arrestation et contexte domestique.",
                                        "La confiance mesure la fiabilité de la règle ; un lift supérieur à 1 indique une association plus fréquente que le hasard.",
                                    ),
                                    html.Div([dcc.Graph(figure=support_fig), dcc.Graph(figure=sankey_fig)], style={**CARD_STYLE, "marginTop": "16px"}),
                                    html.H3("Principales règles", style={"marginTop": "24px"}),
                                    dash_table.DataTable(
                                        data=rules_table.to_dict("records"),
                                        columns=[{"name": column, "id": column} for column in rules_table.columns],
                                        page_size=10,
                                        style_table={"overflowX": "auto"},
                                        style_cell={"textAlign": "left", "padding": "9px", "whiteSpace": "normal", "height": "auto"},
                                        style_header={"fontWeight": "bold", "backgroundColor": "#eaf2f8"},
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="3. Temporalité",
                                children=[
                                    indicator_explanation(
                                        "Série temporelle mensuelle citywide",
                                        "Agrégation directe depuis l'API City of Chicago : nombre d'incidents groupés par année et par mois.",
                                        f"L'évolution temporelle des crimes enregistrés dans toute la ville de Chicago ({temporal_scope}).",
                                        "La courbe permet d'observer les tendances, les variations saisonnières et les périodes de rupture. La prévision reste exploratoire.",
                                    ),
                                    dcc.Graph(figure=monthly_fig),
                                    dcc.Graph(figure=forecast_fig),
                                ],
                            ),
                            dcc.Tab(
                                label="4. Spatial",
                                children=[
                                    indicator_explanation(
                                        "Concentration spatiale et clustering",
                                        "Nettoyage des coordonnées latitude/longitude, puis cartes de densité, K-means et OPTICS.",
                                        "La répartition géographique des incidents dans la ville de Chicago.",
                                        f"Les cartes sont échantillonnées à {MAX_SPATIAL_POINTS:,} points maximum pour rester interactives. Les clusters sont statistiques et ne prouvent pas une causalité.".replace(",", " "),
                                    ),
                                    dcc.Graph(id="density-map"),
                                    dcc.Graph(id="kmeans-map"),
                                    dcc.Graph(id="optics-map"),
                                ],
                            ),
                            dcc.Tab(
                                label="Méthode & limites",
                                children=[
                                    html.Div(
                                        [
                                            html.H3("Processus KDD"),
                                            html.Ol(
                                                [
                                                    html.Li("Sélection d'un dataset multidimensionnel citywide."),
                                                    html.Li("Préparation : conversion des dates, booléens et coordonnées."),
                                                    html.Li("Transformation : variables temporelles, discrétisation et encodage transactionnel."),
                                                    html.Li("Data mining : groupby, Apriori, forecasting, K-means et OPTICS."),
                                                    html.Li("Interprétation : indicateurs, graphiques et dashboard interactif."),
                                                ]
                                            ),
                                            html.H3("Limites"),
                                            html.Ul(
                                                [
                                                    html.Li("Les vues interactives chargent un nombre limité de lignes récentes pour éviter un dashboard trop lourd."),
                                                    html.Li("La série temporelle utilise une agrégation citywide complète, distincte du chargement interactif."),
                                                    html.Li("Les règles Apriori indiquent des associations, pas des causalités."),
                                                    html.Li("Les clusters spatiaux dépendent des paramètres choisis et ne sont pas des quartiers administratifs."),
                                                    html.Li("Les données policières reflètent les crimes enregistrés, pas nécessairement tous les faits réellement commis."),
                                                ]
                                            ),
                                        ],
                                        style={**CARD_STYLE, "lineHeight": "1.55"},
                                    )
                                ],
                            ),
                        ]
                    ),
                ],
                style={"padding": "22px 5%"},
            ),
        ],
        style=PAGE_STYLE,
    )

    @callback(
        Output("kpi-incidents", "children"),
        Output("kpi-arrest", "children"),
        Output("kpi-main-crime", "children"),
        Output("kpi-period", "children"),
        Output("top-crimes-graph", "figure"),
        Output("arrest-rate-graph", "figure"),
        Output("density-map", "figure"),
        Output("kmeans-map", "figure"),
        Output("optics-map", "figure"),
        Input("year-filter", "value"),
        Input("crime-filter", "value"),
    )
    def update_dashboard(year: str | int, crime_type: str):
        filtered = filter_data(df, year, crime_type)
        kpis = calculate_kpis(filtered)
        return (
            kpis["incidents"],
            kpis["arrest_rate"],
            kpis["main_crime"],
            kpis["period"],
            make_top_crimes_figure(filtered),
            make_arrest_rate_figure(filtered),
            make_density_map(filtered),
            make_kmeans_map(filtered),
            make_optics_map(filtered),
        )

    return app


def export_dashboard_html(app: Dash, output_path: Path = EXPORT_PATH) -> None:
    """Exporter une preuve HTML statique du dashboard.

    Input : application Dash initialisée et chemin de sortie.
    Output : fichier HTML contenant la structure de la page.
    """
    html_content = pio.to_html(go.Figure(), include_plotlyjs="cdn", full_html=True)
    html_content = html_content.replace(
        "</body>",
        "<div style='font-family:Arial;padding:40px'>"
        "<h1>Chicago Crime Analysis - Dashboard</h1>"
        "<p>Le dashboard interactif se lance avec la commande : <code>python dashboard/app.py</code>.</p>"
        "<p>Source : City of Chicago Data Portal - Crimes 2001 to Present.</p>"
        "<p>Cette page constitue une preuve HTML de génération du dashboard.</p>"
        "</div></body>",
    )
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Dashboard HTML exporté : {output_path}")


def parse_args() -> argparse.Namespace:
    """Lire les arguments de lancement du dashboard."""
    parser = argparse.ArgumentParser(description="Dashboard Chicago Crime Analysis")
    parser.add_argument("--port", type=int, default=8050, help="Port Dash à utiliser")
    parser.add_argument("--limit", type=int, default=DEFAULT_DATA_LIMIT, help="Nombre de lignes récentes à charger depuis l'API")
    parser.add_argument("--offline", action="store_true", help="Utiliser le cache local si disponible")
    parser.add_argument("--export-html", action="store_true", help="Exporter une preuve HTML statique")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(limit=args.limit, offline=args.offline)
    if args.export_html:
        export_dashboard_html(app)
    else:
        app.run(debug=True, port=args.port)
