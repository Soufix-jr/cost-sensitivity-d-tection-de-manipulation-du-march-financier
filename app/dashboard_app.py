import pandas as pd
import numpy as np
from pathlib import Path

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "reports" / "results"

LABELED_FILE = DATA_DIR / "market_labeled.csv"
FINAL_COMPARISON_FILE = RESULTS_DIR / "final_extended_model_comparison.csv"
THRESHOLD_FILE = RESULTS_DIR / "threshold_optimization_results.csv"
CALIBRATION_FILE = RESULTS_DIR / "calibrated_models_results.csv"


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(LABELED_FILE)
df["Date"] = pd.to_datetime(df["Date"])

comparison_df = pd.read_csv(FINAL_COMPARISON_FILE)
threshold_df = pd.read_csv(THRESHOLD_FILE)
calibration_df = pd.read_csv(CALIBRATION_FILE)


# =====================================================
# BUSINESS CONSTANTS
# =====================================================

C_FN = 15_000_000
C_FP = 7_000


# =====================================================
# APP INIT
# =====================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True
)

server = app.server


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def euro_format(x):
    return f"{x:,.0f} €".replace(",", " ")


def kpi_card(title, value, subtitle, color="primary"):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="text-uppercase text-muted"),
            html.H2(value, className=f"text-{color}"),
            html.P(subtitle, className="text-muted mb-0")
        ]),
        className="shadow-sm border-0 h-100"
    )
def plot_drift_psi_ranking():
    temp = drift_feature_df.sort_values("psi", ascending=False).head(15).copy()

    fig = px.bar(
        temp,
        x="psi",
        y="feature",
        color="severity",
        orientation="h",
        title="Top Feature Drift — PSI Ranking",
        template="plotly_dark",
        color_discrete_map={
            "critical": "#ff3b3b",
            "warning": "#ffb000",
            "none": "#00ff88"
        }
    )

    fig.add_vline(
        x=0.2,
        line_dash="dash",
        line_color="#ff3b3b",
        annotation_text="Critical PSI = 0.2"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,22,0.7)",
        height=520,
        xaxis_title="PSI",
        yaxis_title="Feature"
    )

    return fig


def plot_label_drift_timeline():
    temp = drift_label_df.copy()
    temp["window_end"] = pd.to_datetime(temp["window_end"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=temp["window_end"],
        y=temp["positive_rate"],
        mode="lines",
        name="Window positive rate",
        line=dict(color="#00d4ff", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=temp["window_end"],
        y=temp["baseline_positive_rate"],
        mode="lines",
        name="Baseline positive rate",
        line=dict(color="#ffb000", dash="dash")
    ))

    drift_points = temp[temp["label_drift_detected"] == True]

    fig.add_trace(go.Scatter(
        x=drift_points["window_end"],
        y=drift_points["positive_rate"],
        mode="markers",
        name="Detected label drift",
        marker=dict(color="#ff3b3b", size=8)
    ))

    fig.update_layout(
        title="Label Drift Timeline — Positive Rate by Rolling Window",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,22,0.7)",
        height=500,
        xaxis_title="Window end",
        yaxis_title="Positive rate"
    )

    return fig


def plot_cost_degradation_monitor():
    temp = drift_cost_df.copy()
    temp["reference_cost_m"] = temp["reference_cost"] / 1_000_000
    temp["current_cost_m"] = temp["current_cost"] / 1_000_000

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=temp["model"],
        y=temp["reference_cost_m"],
        name="Reference cost",
        marker_color="#1f4e79"
    ))

    fig.add_trace(go.Bar(
        x=temp["model"],
        y=temp["current_cost_m"],
        name="Current cost",
        marker_color="#00d4ff"
    ))

    fig.update_layout(
        title="Cost Degradation Monitoring",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,22,0.7)",
        barmode="group",
        height=480,
        xaxis_tickangle=-25,
        yaxis_title="Cost in million euros"
    )

    return fig


def plot_threshold_recalibration():
    temp = drift_recalibration_df.copy()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=temp["model"],
        y=temp["old_threshold"],
        name="Old threshold",
        marker_color="#ffb000"
    ))

    fig.add_trace(go.Bar(
        x=temp["model"],
        y=temp["new_threshold"],
        name="New threshold",
        marker_color="#00ff88"
    ))

    fig.update_layout(
        title="Automatic Threshold Recalibration after Drift",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,22,0.7)",
        barmode="group",
        height=480,
        xaxis_tickangle=-25,
        yaxis_title="Threshold"
    )

    return fig

def create_label_distribution_fig():
    counts = df["label"].value_counts().sort_index()
    temp = pd.DataFrame({
        "Classe": ["Normal", "Manipulation suspecte"],
        "Nombre": counts.values
    })

    fig = px.bar(
        temp,
        x="Classe",
        y="Nombre",
        text="Nombre",
        title="Distribution globale des classes",
        template="plotly_dark"
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(height=430)

    return fig


def create_ticker_distribution_fig():
    table = pd.crosstab(df["ticker"], df["label"]).reset_index()
    table.columns = ["ticker", "Normal", "Manipulation suspecte"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=table["ticker"],
        y=table["Normal"],
        name="Normal"
    ))

    fig.add_trace(go.Bar(
        x=table["ticker"],
        y=table["Manipulation suspecte"],
        name="Manipulation suspecte"
    ))

    fig.update_layout(
        title="Distribution des classes par actif",
        barmode="stack",
        template="plotly_dark",
        height=450
    )

    return fig


def create_temporal_fig():
    monthly = (
        df.set_index("Date")
        .groupby("ticker")["label"]
        .resample("ME")
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="Date",
        y="label",
        color="ticker",
        markers=True,
        title="Évolution temporelle des signaux suspects",
        template="plotly_dark"
    )

    fig.update_layout(
        yaxis_title="Nombre de signaux suspects",
        height=500
    )

    return fig


def create_model_comparison_fig():
    temp = comparison_df.copy()
    temp["cost_million"] = temp["cost"] / 1_000_000

    fig = px.bar(
        temp,
        x="model",
        y="cost_million",
        color="method",
        text=temp["cost"].apply(euro_format),
        title="Comparaison finale des coûts par modèle",
        template="plotly_dark"
    )

    fig.update_layout(
        xaxis_title="Modèle",
        yaxis_title="Coût total en millions d'euros",
        height=520,
        xaxis_tickangle=-25
    )

    fig.update_traces(textposition="outside")

    return fig


def create_fp_fn_fig():
    temp = comparison_df.copy()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=temp["model"],
        y=temp["FP"],
        name="Faux positifs"
    ))

    fig.add_trace(go.Bar(
        x=temp["model"],
        y=temp["FN"],
        name="Faux négatifs"
    ))

    fig.update_layout(
        title="Comparaison FP / FN par modèle",
        template="plotly_dark",
        barmode="group",
        height=500,
        xaxis_tickangle=-25
    )

    return fig


def create_calibration_fig():
    fig = px.bar(
        calibration_df,
        x="model",
        y="brier_score",
        color="calibration",
        text="brier_score",
        title="Comparaison des Brier Scores après calibration",
        template="plotly_dark"
    )

    fig.update_traces(texttemplate="%{text:.5f}", textposition="outside")
    fig.update_layout(
        yaxis_title="Brier score",
        height=450,
        xaxis_tickangle=-25
    )

    return fig


def create_scatter_precision_recall():
    temp = comparison_df.copy()

    fig = px.scatter(
        temp,
        x="recall",
        y="precision",
        size="cost",
        color="method",
        hover_name="model",
        title="Compromis Precision / Recall / Coût",
        template="plotly_dark"
    )

    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=500
    )

    return fig


# =====================================================
# KPIS
# =====================================================

total_obs = len(df)
n_normal = int((df["label"] == 0).sum())
n_suspicious = int((df["label"] == 1).sum())
positive_rate = n_suspicious / total_obs

best_model = comparison_df.sort_values("cost").iloc[0]
best_model_name = best_model["model"]
best_method = best_model["method"]
best_cost = best_model["cost"]
best_fp = int(best_model["FP"])
best_fn = int(best_model["FN"])
best_recall = best_model["recall"]
best_precision = best_model["precision"]


# =====================================================
# LAYOUT
# =====================================================

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row([
            dbc.Col([
                html.H1(
                    "Market Manipulation Cost-Sensitive Dashboard",
                    className="text-info fw-bold mt-4"
                ),
                html.P(
                    "Dashboard interactif pour analyser les données, comparer les modèles "
                    "et interpréter les décisions cost-sensitive.",
                    className="text-muted"
                )
            ])
        ]),

        html.Hr(),

        dbc.Row([
            dbc.Col(
                kpi_card(
                    "Observations",
                    f"{total_obs:,}".replace(",", " "),
                    "Nombre total d'observations",
                    "info"
                ),
                md=3
            ),
            dbc.Col(
                kpi_card(
                    "Manipulations suspectes",
                    f"{n_suspicious}",
                    f"Taux positif : {positive_rate:.2%}",
                    "warning"
                ),
                md=3
            ),
            dbc.Col(
                kpi_card(
                    "Meilleur modèle",
                    best_model_name,
                    best_method,
                    "success"
                ),
                md=3
            ),
            dbc.Col(
                kpi_card(
                    "Coût minimal",
                    euro_format(best_cost),
                    f"FP={best_fp} | FN={best_fn}",
                    "danger"
                ),
                md=3
            )
        ], className="g-3 mb-4"),

        dbc.Tabs([
            dbc.Tab(label="Vue globale", tab_id="tab-overview"),
            dbc.Tab(label="Exploration des données", tab_id="tab-data"),
            dbc.Tab(label="Modèles", tab_id="tab-models"),
            dbc.Tab(label="Cost-sensitive analysis", tab_id="tab-cost"),
            dbc.Tab(label="Simulation seuil", tab_id="tab-simulator"),
            dbc.Tab(label="Conclusion business", tab_id="tab-business"),
            dbc.Tab(label="Risk Command Center", tab_id="risk"),
            dbc.Tab(label="Live Timeline", tab_id="timeline"),
            dbc.Tab(label="Model Arena", tab_id="arena"),
            dbc.Tab(label="Cost Simulator", tab_id="simulator"),
            dbc.Tab(label="Alert Investigation Room", tab_id="alerts"),
            dbc.Tab(label="Drift Monitoring Center", tab_id="drift"),
            dbc.Tab(label="Executive Story", tab_id="story"),
        ], id="tabs", active_tab="tab-overview"),

        html.Div(id="tab-content", className="mt-4")
    ]
)


# =====================================================
# TABS CALLBACK
# =====================================================

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab(active_tab):

    if active_tab == "tab-overview":
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Résumé exécutif"),
                        dbc.CardBody([
                            html.P(
                                "Le projet vise à détecter les manipulations de marché en minimisant "
                                "le coût économique des erreurs. Le faux négatif est l'erreur critique : "
                                "il correspond à une manipulation réelle non détectée."
                            ),
                            html.P(
                                "Le meilleur modèle empirique est Random Forest + SMOTE, car il obtient "
                                "zéro faux négatif sur le test set. Threshold Moving reste cependant "
                                "plus justifiable théoriquement, car son seuil est directement relié "
                                "à la matrice de coûts."
                            ),
                            html.H5("Meilleur résultat", className="text-info"),
                            html.Ul([
                                html.Li(f"Modèle : {best_model_name}"),
                                html.Li(f"Méthode : {best_method}"),
                                html.Li(f"Coût total : {euro_format(best_cost)}"),
                                html.Li(f"Precision : {best_precision:.4f}"),
                                html.Li(f"Recall : {best_recall:.4f}"),
                                html.Li(f"FP : {best_fp}"),
                                html.Li(f"FN : {best_fn}")
                            ])
                        ])
                    ], className="shadow-sm border-0")
                ], md=5),

                dbc.Col([
                    dcc.Graph(figure=create_model_comparison_fig())
                ], md=7)
            ]),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=create_fp_fn_fig())
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=create_scatter_precision_recall())
                ], md=6)
            ], className="mt-4")
        ], fluid=True)

    elif active_tab == "tab-data":
        return dbc.Container([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=create_label_distribution_fig()), md=6),
                dbc.Col(dcc.Graph(figure=create_ticker_distribution_fig()), md=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=create_temporal_fig()), md=12)
            ], className="mt-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Interprétation"),
                        dbc.CardBody([
                            html.P(
                                "La classe normale domine largement le dataset. Cela confirme que "
                                "le problème est déséquilibré et que l'accuracy seule ne suffit pas."
                            ),
                            html.P(
                                "GME et AMC présentent davantage de signaux suspects, ce qui est cohérent "
                                "avec leur historique de forte volatilité et de tensions de marché."
                            )
                        ])
                    ])
                ])
            ], className="mt-4")
        ], fluid=True)

    elif active_tab == "tab-models":
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=create_calibration_fig())
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=create_scatter_precision_recall())
                ], md=6)
            ]),

            dbc.Row([
                dbc.Col([
                    html.H4("Table finale des modèles", className="text-info mt-4"),
                    dbc.Table.from_dataframe(
                        comparison_df.round(4),
                        striped=True,
                        bordered=True,
                        hover=True,
                        responsive=True,
                        className="table-dark"
                    )
                ])
            ])
        ], fluid=True)

    elif active_tab == "tab-cost":
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Matrice de coûts"),
                        dbc.CardBody([
                            html.H4("Coût métier utilisé", className="text-info"),
                            html.P(f"Coût faux négatif : {euro_format(C_FN)}"),
                            html.P(f"Coût faux positif : {euro_format(C_FP)}"),
                            html.Hr(),
                            html.P(
                                "Le coût total est calculé par : "
                                "Cost = c_FN × FN + c_FP × FP."
                            ),
                            html.P(
                                "Une seule manipulation ratée peut coûter plus cher que plusieurs fausses alertes."
                            )
                        ])
                    ])
                ], md=4),

                dbc.Col([
                    dcc.Graph(figure=create_model_comparison_fig())
                ], md=8)
            ]),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=create_fp_fn_fig())
                ])
            ], className="mt-4")
        ], fluid=True)

    elif active_tab == "tab-simulator":
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Simulateur de seuil"),
                        dbc.CardBody([
                            html.P(
                                "Ce simulateur montre comment le seuil théorique de Bayes change "
                                "lorsqu'on modifie les coûts métier."
                            ),

                            html.Label("Coût faux négatif c_FN"),
                            dcc.Slider(
                                id="slider-cfn",
                                min=1_000_000,
                                max=50_000_000,
                                step=1_000_000,
                                value=C_FN,
                                marks={
                                    1_000_000: "1M",
                                    15_000_000: "15M",
                                    30_000_000: "30M",
                                    50_000_000: "50M"
                                }
                            ),

                            html.Br(),

                            html.Label("Coût faux positif c_FP"),
                            dcc.Slider(
                                id="slider-cfp",
                                min=1_000,
                                max=50_000,
                                step=1_000,
                                value=C_FP,
                                marks={
                                    1_000: "1k",
                                    7_000: "7k",
                                    20_000: "20k",
                                    50_000: "50k"
                                }
                            ),

                            html.Div(id="threshold-output", className="mt-4")
                        ])
                    ])
                ], md=5),

                dbc.Col([
                    dcc.Graph(id="threshold-gauge")
                ], md=7)
            ])
        ], fluid=True)

    elif active_tab == "tab-business":
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Conclusion business"),
                        dbc.CardBody([
                            html.H4("Lecture finale", className="text-info"),
                            html.P(
                                "Le dashboard montre que le meilleur résultat empirique est obtenu "
                                "avec Random Forest + SMOTE. Cette méthode atteint zéro faux négatif, "
                                "ce qui minimise fortement le coût total."
                            ),
                            html.P(
                                "Cependant, Threshold Moving reste central dans le projet. Sa force n'est pas "
                                "seulement empirique : elle est décisionnelle. Le seuil est directement relié "
                                "aux coûts métier, ce qui rend la méthode plus explicable et plus défendable."
                            ),
                            html.H5("Message clé", className="text-warning"),
                            html.P(
                                "SMOTE traite le déséquilibre statistique. "
                                "Threshold Moving traite la décision économique."
                            ),
                            html.P(
                                "Dans un contexte financier réel, le bon modèle doit donc être choisi "
                                "en tenant compte à la fois de la performance, de l'explicabilité, "
                                "du coût des erreurs et de la capacité d'enquête des analystes."
                            )
                        ])
                    ])
                ])
            ])
        ], fluid=True)

    return html.Div("Tab inconnue.")


# =====================================================
# SIMULATOR CALLBACK
# =====================================================

@app.callback(
    Output("threshold-output", "children"),
    Output("threshold-gauge", "figure"),
    Input("slider-cfn", "value"),
    Input("slider-cfp", "value")
)
def update_threshold_simulator(c_fn, c_fp):
    threshold = c_fp / (c_fn + c_fp)

    output = dbc.Alert(
        [
            html.H5("Seuil théorique de Bayes"),
            html.P(f"c_FN = {euro_format(c_fn)}"),
            html.P(f"c_FP = {euro_format(c_fp)}"),
            html.H3(f"t* = {threshold:.6f}", className="text-warning"),
            html.P(
                "Plus le coût du faux négatif augmente, plus le seuil baisse. "
                "Le système devient alors plus prudent et déclenche plus facilement une alerte."
            )
        ],
        color="dark"
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=threshold,
        number={"valueformat": ".6f"},
        title={"text": "Seuil optimal théorique"},
        gauge={
            "axis": {"range": [0, 0.01]},
            "bar": {"color": "#00CC96"},
            "steps": [
                {"range": [0, 0.001], "color": "#1F4E79"},
                {"range": [0.001, 0.005], "color": "#D68910"},
                {"range": [0.005, 0.01], "color": "#B03A2E"},
            ],
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        height=430
    )

    return output, fig


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)