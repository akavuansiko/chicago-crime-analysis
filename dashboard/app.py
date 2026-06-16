"""Dashboard interactif - Chicago Crime Analysis.

Ce fichier rassemble les quatre analyses réalisées dans les notebooks du projet :
1. exploration et agrégations ;
2. extraction de motifs fréquents avec Apriori ;
3. analyse temporelle et prévision Prophet ;
4. analyse spatiale et clustering K-means.

Lancement du dashboard :
    python app.py

Création d'une preuve HTML autonome :
    python app.py --export-html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, callback, dash_table, dcc, html
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from prophet import Prophet
from sklearn.cluster import KMeans, OPTICS


# -----------------------------------------------------------------------------
# Configuration générale
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "chicago_crime.csv"
EXPORT_PATH = BASE_DIR / "Kavuansiko_dashboard.html"

DATASET_NAME = "Chicago Crime Dataset - Bridgeport"
TEAM_MEMBERS = [
    "Angelikia Kavuansiko - exploration des données",
    "Ekta - pattern mining",
    "Léora - analyse temporelle",
    "Chrisa - analyse spatiale",
    "Flavie - dashboard et intégration finale",
]

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
# Chargement et préparation des données
# -----------------------------------------------------------------------------
def load_data(file_path: str | Path) -> pd.DataFrame:
    """Charger et préparer le fichier CSV.

    Input : chemin du fichier CSV.
    Output : dataframe nettoyé et enrichi de variables temporelles.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. Placez app.py dans le dossier dashboard/ "
            "et le CSV dans data/chicago_crime.csv."
        )

    df = pd.read_csv(path)

    # Conversion explicite des dates américaines : mois/jour/année.
    df["Date"] = pd.to_datetime(
        df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )

    # Les coordonnées utilisent une virgule décimale dans le fichier fourni.
    for column in ["Latitude", "Longitude"]:
        df[column] = pd.to_numeric(
            df[column].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    # Variables utiles aux analyses temporelles et aux filtres du dashboard.
    df["Year"] = df["Date"].dt.year.astype("Int64")
    df["Month"] = df["Date"].dt.month.astype("Int64")
    df["Hour"] = df["Date"].dt.hour.astype("Int64")
    df["YearMonth"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    return df


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
    period = f"{min_date:%m/%Y} - {max_date:%m/%Y}"

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
    Output : dataframe discrétisé identique à celui du notebook pattern_mining.
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

    # Même règle que dans le notebook : cinq lieux conservés, tous les autres
    # sont regroupés dans la catégorie AUTRE.
    top_locations = result["Location Description"].value_counts().nlargest(5).index
    result["Heure"] = result["Hour"].apply(time_slot)
    result["Lieu"] = result["Location Description"].apply(
        lambda value: value if value in top_locations else "AUTRE"
    )
    result["Arrestation"] = np.where(
        result["Arrest"], "Arrestation_OUI", "Arrestation_NON"
    )
    result["Domestique"] = np.where(
        result["Domestic"], "Domestique_OUI", "Domestique_NON"
    )

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
# Requête 3 - dimension temporelle et forecasting
# -----------------------------------------------------------------------------
def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Agréger les incidents par mois en réintroduisant les mois sans incident.

    Input : dataframe préparé.
    Output : série mensuelle complète avec Date et Nombre de crimes.
    """
    valid = df.dropna(subset=["Date"]).set_index("Date")
    monthly = valid.resample("MS").size().rename("Nombre de crimes").reset_index()
    return monthly


def make_monthly_figure(monthly: pd.DataFrame) -> go.Figure:
    """Visualiser l'évolution mensuelle du nombre d'incidents.

    Input : agrégation mensuelle.
    Output : courbe temporelle Plotly.
    """
    fig = px.line(
        monthly,
        x="Date",
        y="Nombre de crimes",
        title="Évolution mensuelle des incidents",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=65, b=20), height=430)
    return fig


def run_prophet_forecast(
    monthly: pd.DataFrame, periods: int = 12
) -> tuple[Prophet | None, pd.DataFrame]:
    """Entraîner Prophet et prévoir les douze prochains mois.

    Input : série mensuelle et horizon de prévision.
    Output : modèle entraîné et dataframe des valeurs prévues.

    Une régression tendance-saisonnalité est utilisée comme solution de secours si
    le moteur Stan de Prophet n'est pas compatible avec l'environnement local.
    Cette sécurité évite qu'un problème d'installation bloque toute la démo.
    """
    prophet_df = monthly.rename(columns={"Date": "ds", "Nombre de crimes": "y"})

    try:
        # Prophet/Stan n'est pas toujours stable sous les versions Python très récentes.
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

        # Modèle de secours : tendance linéaire + saisonnalité annuelle sinusoïdale.
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
        residual_std = float(
            np.std(prophet_df["y"].to_numpy(dtype=float) - fitted, ddof=1)
        )
        margin = 1.282 * residual_std  # intervalle approximatif à 80 %

        future_dates = pd.date_range(
            start=prophet_df["ds"].min(), periods=total_length, freq="MS"
        )
        forecast = pd.DataFrame(
            {
                "ds": future_dates,
                "yhat": predicted,
                "yhat_lower": predicted - margin,
                "yhat_upper": predicted + margin,
            }
        )
        forecast.attrs["model_name"] = "modèle saisonnier de secours"

    # Un nombre d'incidents ne peut pas être négatif.
    for column in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[column] = forecast[column].clip(lower=0)
    return model, forecast


def make_forecast_figure(monthly: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """Comparer les observations et la prévision Prophet.

    Input : série observée et dataframe de prévision.
    Output : figure avec prévision et intervalle d'incertitude.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["Date"],
            y=monthly["Nombre de crimes"],
            mode="lines",
            name="Données observées",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode="lines",
            fill="tonexty",
            name="Intervalle de prévision à 80 %",
            line=dict(width=0),
        )
    )
    model_name = forecast.attrs.get("model_name", "Prophet")
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name=f"Prévision - {model_name}",
        )
    )
    fig.update_layout(
        title=f"Prévision mensuelle des incidents sur 12 mois ({model_name})",
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
def spatial_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Conserver uniquement les incidents possédant des coordonnées valides.

    Input : dataframe préparé.
    Output : dataframe géolocalisable.
    """
    return df.dropna(subset=["Latitude", "Longitude"]).copy()


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
        radius=18,
        zoom=14,
        mapbox_style="open-street-map",
        hover_name="Primary Type",
        hover_data={"Date": True, "Location Description": True},
        title="Densité géographique des incidents - Bridgeport",
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
        zoom=14,
        mapbox_style="open-street-map",
        hover_name="Primary Type",
        hover_data=["Date", "Location Description", "Arrest"],
        title=f"Clustering K-means - {clustered['Cluster'].nunique()} zones",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.75))
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0), height=560)
    return fig


def apply_optics(df: pd.DataFrame, min_samples: int = 20) -> pd.DataFrame:
    """Détecter les zones denses et les incidents isolés avec OPTICS.

    Input : dataframe spatial et densité minimale.
    Output : copie du dataframe avec une colonne Cluster_OPTICS ; -1 désigne le bruit.
    """
    result = spatial_subset(df)
    if len(result) < 5:
        result["Cluster_OPTICS"] = pd.Series(dtype=str)
        return result

    effective_min_samples = min(min_samples, max(2, len(result) // 5))
    model = OPTICS(
        min_samples=effective_min_samples,
        xi=0.05,
        min_cluster_size=0.05,
    )
    result["Cluster_OPTICS"] = model.fit_predict(
        result[["Latitude", "Longitude"]]
    ).astype(str)
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
        zoom=14,
        mapbox_style="open-street-map",
        hover_name="Primary Type",
        hover_data=["Date", "Location Description", "Arrest"],
        title=(
            f"Clustering OPTICS - {len(dense_labels)} zones denses, "
            f"{isolated} incidents isolés"
        ),
    )
    fig.update_traces(marker=dict(size=8, opacity=0.75))
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
        style={
            **CARD_STYLE,
            "borderLeft": "5px solid #34495e",
            "marginBottom": "16px",
            "lineHeight": "1.45",
        },
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


def create_app(data_path: str | Path = DATA_PATH) -> Dash:
    """Construire l'application Dash et pré-calculer les analyses lourdes.

    Input : chemin du CSV.
    Output : instance Dash prête à être lancée.
    """
    df = load_data(data_path)

    # Analyses calculées une seule fois au démarrage.
    discrete_df = discretise_data(df)
    encoded_df = encode_transactions(discrete_df)
    itemsets, rules = mine_association_rules(encoded_df, min_support=0.10)
    support_fig = make_support_curve(encoded_df)
    sankey_fig = make_sankey_rules(rules)
    rules_table = rules_table_dataframe(rules)

    monthly = aggregate_monthly(df)
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
                        "Prototype de visualisation des incidents criminels du secteur de Bridgeport, Chicago",
                        style={"marginTop": "0", "fontSize": "17px"},
                    ),
                    html.P(
                        f"Dataset : {DATASET_NAME} | {len(df)} incidents | {len(df.columns)} variables",
                        style={"marginBottom": "8px"},
                    ),
                    html.P("Équipe : " + " • ".join(TEAM_MEMBERS), style={"fontSize": "13px"}),
                ],
                style={
                    "padding": "28px 5%",
                    "backgroundColor": "#17202a",
                    "color": "white",
                },
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
                    dcc.Tabs(
                        [
                            dcc.Tab(
                                label="1. Exploration",
                                children=[
                                    indicator_explanation(
                                        "Fréquence des types de crimes",
                                        "Comptage des incidents après regroupement par Primary Type.",
                                        "Le volume observé pour chaque catégorie de crime.",
                                        "Une barre plus longue correspond à un type de crime plus fréquent dans l'échantillon.",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(id="top-crimes-graph", style={"flex": "1"}),
                                            dcc.Graph(id="arrest-rate-graph", style={"flex": "1"}),
                                        ],
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
                                    html.Div(
                                        [dcc.Graph(figure=support_fig), dcc.Graph(figure=sankey_fig)],
                                        style={**CARD_STYLE, "marginTop": "16px"},
                                    ),
                                    html.H3("Principales règles", style={"marginTop": "24px"}),
                                    dash_table.DataTable(
                                        data=rules_table.to_dict("records"),
                                        columns=[{"name": column, "id": column} for column in rules_table.columns],
                                        page_size=10,
                                        style_table={"overflowX": "auto"},
                                        style_cell={"textAlign": "left", "padding": "9px", "whiteSpace": "normal", "height": "auto"},
                                        style_header={"fontWeight": "700", "backgroundColor": "#eaecee"},
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="3. Analyse temporelle",
                                children=[
                                    indicator_explanation(
                                        "Prévision Prophet",
                                        "Agrégation mensuelle des incidents, entraînement d'un modèle Prophet et projection sur douze mois.",
                                        "La trajectoire attendue du nombre mensuel d'incidents et son intervalle d'incertitude.",
                                        "La prévision doit être lue comme une tendance statistique, non comme une certitude, en raison du faible volume et du périmètre limité à Bridgeport.",
                                    ),
                                    html.Div(
                                        [dcc.Graph(figure=monthly_fig), dcc.Graph(figure=forecast_fig)],
                                        style={**CARD_STYLE, "marginTop": "16px"},
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="4. Analyse spatiale",
                                children=[
                                    indicator_explanation(
                                        "Densité et clustering géographique",
                                        "Nettoyage des coordonnées, carte de densité et regroupement K-means en six zones.",
                                        "Les secteurs où les incidents du jeu de données se concentrent.",
                                        "Les zones denses sont des hotspots de l'échantillon Bridgeport ; elles ne permettent pas de conclure sur l'ensemble de Chicago.",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Graph(id="density-map", style={"flex": "1", "minWidth": "430px"}),
                                            dcc.Graph(id="kmeans-map", style={"flex": "1", "minWidth": "430px"}),
                                            dcc.Graph(id="optics-map", style={"flex": "1", "minWidth": "430px"}),
                                        ],
                                        style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginTop": "16px"},
                                    ),
                                ],
                            ),
                            dcc.Tab(
                                label="Méthode et limites",
                                children=[
                                    html.Div(
                                        [
                                            html.H3("Processus KDD appliqué"),
                                            html.Ol(
                                                [
                                                    html.Li("Sélection : choix d'un dataset criminel multidimensionnel."),
                                                    html.Li("Prétraitement : dates, coordonnées, valeurs manquantes et catégories."),
                                                    html.Li("Transformation : agrégations, discrétisation et encodage booléen."),
                                                    html.Li("Data mining : Apriori, Prophet et K-means."),
                                                    html.Li("Interprétation : indicateurs, visualisations et limites méthodologiques."),
                                                ]
                                            ),
                                            html.H3("Limites à annoncer pendant la soutenance"),
                                            html.Ul(
                                                [
                                                    html.Li("Le jeu de données ne couvre que Bridgeport et non l'intégralité de Chicago."),
                                                    html.Li("949 observations constituent un volume modeste pour le forecasting et le clustering."),
                                                    html.Li("L'année 2026 est incomplète, donc son niveau ne doit pas être comparé directement aux années complètes."),
                                                    html.Li("Une association Apriori n'établit pas de relation de causalité."),
                                                    html.Li("K-means impose un nombre de zones et des formes de clusters approximativement sphériques."),
                                                ]
                                            ),
                                        ],
                                        style={**CARD_STYLE, "marginTop": "16px", "lineHeight": "1.6"},
                                    )
                                ],
                            ),
                        ]
                    ),
                ],
                style={"padding": "24px 5% 50px"},
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
        """Mettre à jour les KPI et graphiques filtrables."""
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


# -----------------------------------------------------------------------------
# Export HTML autonome : preuve exigée dans le sujet
# -----------------------------------------------------------------------------
def export_static_dashboard(data_path: str | Path, output_path: str | Path) -> Path:
    """Exporter une version HTML autonome regroupant les quatre requêtes.

    Input : chemin du CSV et chemin du fichier HTML de sortie.
    Output : chemin du fichier HTML créé.
    """
    df = load_data(data_path)
    kpis = calculate_kpis(df)

    discrete_df = discretise_data(df)
    encoded_df = encode_transactions(discrete_df)
    itemsets, rules = mine_association_rules(encoded_df, min_support=0.10)
    rules_table = rules_table_dataframe(rules).to_html(index=False, classes="rules-table")

    monthly = aggregate_monthly(df)
    _, forecast = run_prophet_forecast(monthly, periods=12)

    figures = [
        make_top_crimes_figure(df),
        make_arrest_rate_figure(df),
        make_support_curve(encoded_df),
        make_sankey_rules(rules),
        make_monthly_figure(monthly),
        make_forecast_figure(monthly, forecast),
        make_density_map(df),
        make_kmeans_map(df),
        make_optics_map(df),
    ]

    figure_html = []
    for index, figure in enumerate(figures):
        figure_html.append(
            pio.to_html(
                figure,
                include_plotlyjs=True if index == 0 else False,
                full_html=False,
                config={"responsive": True, "displaylogo": False},
            )
        )

    team_html = " • ".join(TEAM_MEMBERS)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    html_document = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chicago Crime Analysis - Dashboard</title>
<style>
body {{ margin:0; font-family:Arial,Helvetica,sans-serif; background:#f4f6f8; color:#17202a; }}
header {{ background:#17202a; color:white; padding:32px 6%; }}
main {{ padding:24px 6% 60px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:18px; }}
.card {{ background:white; border-radius:14px; padding:18px; margin-bottom:18px; box-shadow:0 3px 14px rgba(0,0,0,.08); }}
.kpi {{ font-size:30px; font-weight:700; margin-top:8px; }}
.section-title {{ margin-top:38px; border-bottom:3px solid #34495e; padding-bottom:8px; }}
.note {{ border-left:5px solid #34495e; line-height:1.5; }}
.rules-table {{ width:100%; border-collapse:collapse; font-size:14px; }}
.rules-table th,.rules-table td {{ border:1px solid #d5d8dc; padding:9px; text-align:left; }}
.rules-table th {{ background:#eaecee; }}
.warning {{ border-left:5px solid #b9770e; }}
</style>
</head>
<body>
<header>
<h1>Chicago Crime Analysis</h1>
<p>Prototype de visualisation des incidents criminels du secteur de Bridgeport, Chicago</p>
<p><strong>Dataset :</strong> {DATASET_NAME} | {len(df)} incidents | {len(df.columns)} variables</p>
<p><strong>Équipe :</strong> {team_html}</p>
</header>
<main>
<div class="grid">
<div class="card"><div>Nombre d'incidents</div><div class="kpi">{kpis['incidents']}</div></div>
<div class="card"><div>Taux d'arrestation</div><div class="kpi">{kpis['arrest_rate']}</div></div>
<div class="card"><div>Crime dominant</div><div class="kpi">{kpis['main_crime']}</div></div>
<div class="card"><div>Période observée</div><div class="kpi">{kpis['period']}</div></div>
</div>

<h2 class="section-title">1. Exploration et agrégations</h2>
<div class="card note"><strong>Calcul :</strong> regroupement par type de crime et moyenne de la variable Arrest.<br>
<strong>Représentation :</strong> fréquence des catégories et proportion d'incidents suivis d'une arrestation.<br>
<strong>Interprétation :</strong> les volumes décrivent l'échantillon ; le taux d'arrestation ne mesure pas à lui seul l'efficacité policière.</div>
<div class="grid"><div class="card">{figure_html[0]}</div><div class="card">{figure_html[1]}</div></div>

<h2 class="section-title">2. Pattern mining - Apriori</h2>
<div class="card note"><strong>Calcul :</strong> discrétisation puis Apriori avec support minimal 0,10 ; {len(itemsets)} itemsets et {len(rules)} règles.<br>
<strong>Représentation :</strong> associations entre heure, lieu, type de crime, arrestation et contexte domestique.<br>
<strong>Interprétation :</strong> confiance = fiabilité conditionnelle ; lift &gt; 1 = association plus fréquente que le hasard. Une association ne prouve pas une causalité.</div>
<div class="grid"><div class="card">{figure_html[2]}</div><div class="card">{figure_html[3]}</div></div>
<div class="card"><h3>Principales règles</h3>{rules_table}</div>

<h2 class="section-title">3. Analyse temporelle et forecasting</h2>
<div class="card note"><strong>Calcul :</strong> agrégation mensuelle, entraînement Prophet et prévision sur douze mois.<br>
<strong>Représentation :</strong> tendance attendue et intervalle de prévision à 80 %.<br>
<strong>Interprétation :</strong> la projection reste fragile compte tenu du faible volume mensuel et de l'année 2026 incomplète.</div>
<div class="grid"><div class="card">{figure_html[4]}</div><div class="card">{figure_html[5]}</div></div>

<h2 class="section-title">4. Analyse spatiale et clustering</h2>
<div class="card note"><strong>Calcul :</strong> carte de densité et K-means sur les coordonnées valides.<br>
<strong>Représentation :</strong> concentrations d'incidents et partition en six zones.<br>
<strong>Interprétation :</strong> les hotspots concernent uniquement Bridgeport et ne doivent pas être généralisés à toute la ville.</div>
<div class="grid"><div class="card">{figure_html[6]}</div><div class="card">{figure_html[7]}</div><div class="card">{figure_html[8]}</div></div>

<h2 class="section-title">Méthode et limites</h2>
<div class="card warning">
<ul>
<li>Le dataset porte sur Bridgeport, pas sur toute la ville de Chicago.</li>
<li>Le volume de 949 observations est limité pour certaines techniques prédictives.</li>
<li>L'année 2026 est incomplète.</li>
<li>Les règles Apriori décrivent des cooccurrences, non des causalités.</li>
<li>K-means impose à l'avance le nombre de clusters.</li>
</ul>
</div>
</main>
</body>
</html>"""

    output.write_text(html_document, encoding="utf-8")
    return output


def main() -> None:
    """Point d'entrée : exporter le HTML ou lancer le serveur Dash."""
    parser = argparse.ArgumentParser(description="Chicago Crime Analysis Dashboard")
    parser.add_argument(
        "--export-html",
        action="store_true",
        help="génère Kavuansiko_dashboard.html au lieu de lancer le serveur",
    )
    parser.add_argument("--host", default="127.0.0.1", help="adresse du serveur Dash")
    parser.add_argument("--port", type=int, default=8050, help="port du serveur Dash")
    args = parser.parse_args()

    if args.export_html:
        created = export_static_dashboard(DATA_PATH, EXPORT_PATH)
        print(f"Dashboard HTML créé : {created}")
        return

    app = create_app(DATA_PATH)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
