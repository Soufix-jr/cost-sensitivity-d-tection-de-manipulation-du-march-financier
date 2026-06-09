import pandas as pd
import numpy as np
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, dash_table
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

DRIFT_FEATURE_FILE = RESULTS_DIR / "drift_feature_report.csv"
DRIFT_LABEL_FILE = RESULTS_DIR / "drift_label_report.csv"
DRIFT_COST_FILE = RESULTS_DIR / "drift_cost_degradation_report.csv"
DRIFT_RECALIBRATION_FILE = RESULTS_DIR / "drift_threshold_recalibration.csv"
DRIFT_LOG_FILE = RESULTS_DIR / "drift_events_log.csv"


# =====================================================
# CONSTANTS
# =====================================================

C_FN = 15_000_000
C_FP = 7_000


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(LABELED_FILE)
df["Date"] = pd.to_datetime(df["Date"])

comparison_df = pd.read_csv(FINAL_COMPARISON_FILE)
threshold_df = pd.read_csv(THRESHOLD_FILE)
calibration_df = pd.read_csv(CALIBRATION_FILE)

drift_feature_df = pd.read_csv(DRIFT_FEATURE_FILE)
drift_label_df = pd.read_csv(DRIFT_LABEL_FILE)
drift_cost_df = pd.read_csv(DRIFT_COST_FILE)
drift_recalibration_df = pd.read_csv(DRIFT_RECALIBRATION_FILE)
drift_log_df = pd.read_csv(DRIFT_LOG_FILE)


# =====================================================
# PREPARE DATA
# =====================================================

total_obs = len(df)
n_suspicious = int(df["label"].sum())
n_normal = total_obs - n_suspicious
positive_rate = n_suspicious / total_obs

ticker_risk = (
    df.groupby("ticker")
    .agg(
        total=("label", "count"),
        suspicious=("label", "sum"),
        avg_score=("suspicion_score", "mean")
    )
    .reset_index()
)

ticker_risk["risk_rate"] = ticker_risk["suspicious"] / ticker_risk["total"]
most_risky_ticker = ticker_risk.sort_values("risk_rate", ascending=False).iloc[0]

best_model = comparison_df.sort_values("cost").iloc[0]


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
# HELPERS
# =====================================================

def euro(x):
    return f"{x:,.0f} €".replace(",", " ")


def card(title, value, subtitle, color_class="neon-blue"):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="kpi-title"),
            html.Div(value, className=f"kpi-value {color_class}"),
            html.Div(subtitle, className="kpi-subtitle"),
        ]),
        className="glass-card h-100"
    )


def risk_status():
    if positive_rate >= 0.03:
        return "HIGH ATTENTION", "risk-badge-high"
    return "MODERATE SURVEILLANCE", "risk-badge-medium"


def transparent_layout(fig, height=450):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,22,0.7)",
        height=height
    )
    return fig


# =====================================================
# MAIN PLOTS
# =====================================================

def plot_risk_gauge():
    risk_score = min(100, positive_rate * 2500)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Global Market Risk Score"},
        number={"suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#00d4ff"},
            "steps": [
                {"range": [0, 35], "color": "#0b3d2e"},
                {"range": [35, 70], "color": "#4d3b00"},
                {"range": [70, 100], "color": "#4d0000"},
            ],
            "threshold": {
                "line": {"color": "#ff3b3b", "width": 4},
                "thickness": 0.75,
                "value": 70,
            },
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=330,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def plot_ticker_risk_map():
    temp = ticker_risk.copy()
    temp["risk_rate_pct"] = temp["risk_rate"] * 100

    fig = px.treemap(
        temp,
        path=["ticker"],
        values="total",
        color="risk_rate_pct",
        color_continuous_scale=["#00ff88", "#ffb000", "#ff3b3b"],
        title="Risk Map by Asset",
        hover_data=["suspicious", "risk_rate_pct", "avg_score"],
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=430,
        margin=dict(l=10, r=10, t=60, b=10)
    )

    return fig


def plot_live_timeline():
    suspicious = df[df["label"] == 1].copy()

    fig = px.scatter(
        suspicious,
        x="Date",
        y="ticker",
        size="suspicion_score",
        color="ticker",
        hover_data=[
            "suspicion_score",
            "abnormal_return",
            "abnormal_volume",
            "volatility_ratio",
            "price_zscore"
        ],
        title="Live Market Abuse Timeline",
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,22,0.7)",
        height=520,
        xaxis_title="Date",
        yaxis_title="Asset"
    )

    return fig


def plot_model_arena():
    temp = comparison_df.copy()
    temp["cost_million"] = temp["cost"] / 1_000_000

    fig = px.scatter(
        temp,
        x="recall",
        y="precision",
        size="cost_million",
        color="method",
        hover_name="model",
        hover_data=["FP", "FN", "cost"],
        title="Model Arena — Precision vs Recall vs Cost",
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,22,0.7)",
        height=500,
        xaxis_title="Recall",
        yaxis_title="Precision"
    )

    return fig


def plot_cost_ranking():
    temp = comparison_df.sort_values("cost").copy()
    temp["cost_million"] = temp["cost"] / 1_000_000

    fig = px.bar(
        temp,
        x="model",
        y="cost_million",
        color="method",
        text=temp["cost"].apply(euro),
        title="Cost Ranking by Strategy",
        template="plotly_dark"
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,22,0.7)",
        height=520,
        xaxis_tickangle=-25,
        yaxis_title="Cost in million euros"
    )

    return fig


def plot_fp_fn():
    temp = comparison_df.sort_values("cost").copy()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=temp["model"],
        y=temp["FP"],
        name="False Positives",
        marker_color="#ffb000"
    ))

    fig.add_trace(go.Bar(
        x=temp["model"],
        y=temp["FN"],
        name="False Negatives",
        marker_color="#ff3b3b"
    ))

    fig.update_layout(
        title="False Positives vs False Negatives",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,22,0.7)",
        barmode="group",
        height=480,
        xaxis_tickangle=-25
    )

    return fig


def plot_calibration():
    fig = px.bar(
        calibration_df,
        x="model",
        y="brier_score",
        color="calibration",
        text="brier_score",
        title="Calibration Quality — Brier Score",
        template="plotly_dark"
    )

    fig.update_traces(texttemplate="%{text:.5f}", textposition="outside")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,22,0.7)",
        height=430,
        xaxis_tickangle=-25
    )

    return fig


def alert_table_data():
    cols = [
        "Date",
        "ticker",
        "suspicion_score",
        "abnormal_return",
        "abnormal_volume",
        "volatility_ratio",
        "price_zscore",
        "return_volume_interaction"
    ]

    temp = df[df["label"] == 1][cols].copy()
    temp = temp.sort_values(["suspicion_score", "return_volume_interaction"], ascending=False)

    temp["Date"] = temp["Date"].astype(str)

    for col in temp.columns:
        if col not in ["Date", "ticker"]:
            temp[col] = temp[col].round(4)

    return temp.head(100)


# =====================================================
# DRIFT PLOTS
# =====================================================

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


# =====================================================
# LAYOUT
# =====================================================

status_text, status_class = risk_status()

app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row([
            dbc.Col([
                html.H1("Market Abuse Command Center", className="main-title mt-4"),
                html.P(
                    "Immersive decision interface for cost-sensitive market manipulation surveillance.",
                    className="subtitle"
                ),
            ], md=9),
            dbc.Col([
                html.Div(status_text, className=f"{status_class} text-center mt-5")
            ], md=3)
        ]),

        html.Hr(),

        dbc.Row([
            dbc.Col(
                card(
                    "Total observations",
                    f"{total_obs:,}".replace(",", " "),
                    "Market windows analysed",
                    "neon-blue"
                ),
                md=3
            ),
            dbc.Col(
                card(
                    "Suspicious events",
                    f"{n_suspicious}",
                    f"Rate: {positive_rate:.2%}",
                    "neon-orange"
                ),
                md=3
            ),
            dbc.Col(
                card(
                    "Riskiest asset",
                    most_risky_ticker["ticker"],
                    f"Risk rate: {most_risky_ticker['risk_rate']:.2%}",
                    "neon-red"
                ),
                md=3
            ),
            dbc.Col(
                card(
                    "Best empirical strategy",
                    best_model["model"],
                    euro(best_model["cost"]),
                    "neon-green"
                ),
                md=3
            ),
        ], className="g-3 mb-4"),

        dbc.Tabs(
            [
                dbc.Tab(label="Risk Command Center", tab_id="risk"),
                dbc.Tab(label="Live Timeline", tab_id="timeline"),
                dbc.Tab(label="Model Arena", tab_id="arena"),
                dbc.Tab(label="Cost Simulator", tab_id="simulator"),
                dbc.Tab(label="Alert Investigation Room", tab_id="alerts"),
                dbc.Tab(label="Drift Monitoring Center", tab_id="drift"),
                dbc.Tab(label="Executive Story", tab_id="story"),
            ],
            id="tabs",
            active_tab="risk",
            className="command-tabs"
        ),

        html.Div(id="content", className="mt-4")
    ]
)


# =====================================================
# TAB RENDERING
# =====================================================

@app.callback(
    Output("content", "children"),
    Input("tabs", "active_tab")
)
def render_tab(tab):

    if tab == "risk":
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Surveillance Narrative", className="section-title"),
                            html.P(
                                "This command center does not only display model scores. "
                                "It supports a decision process: detect suspicious market behavior, "
                                "understand when and where alerts appear, compare detection strategies, "
                                "and simulate the economic impact of false positives and false negatives."
                            ),
                            html.P(
                                "The central business risk is the false negative: a real manipulation that remains undetected. "
                                "The cost-sensitive approach therefore prioritizes recall and total economic cost over accuracy alone."
                            ),
                            html.H5("Core decision equation", className="neon-blue mt-3"),
                            html.Code("Cost = c_FN × FN + c_FP × FP"),
                        ])
                    ], className="glass-card")
                ], md=5),

                dbc.Col([
                    dcc.Graph(figure=plot_risk_gauge())
                ], md=3),

                dbc.Col([
                    dcc.Graph(figure=plot_ticker_risk_map())
                ], md=4),
            ]),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_cost_ranking()), md=7),
                dbc.Col(dcc.Graph(figure=plot_fp_fn()), md=5),
            ], className="mt-4")
        ], fluid=True)

    if tab == "timeline":
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Live Market Timeline", className="section-title"),
                            html.P(
                                "This view transforms suspicious observations into a temporal investigation map. "
                                "Each point represents a suspicious event. The size reflects the suspicion score. "
                                "The tooltip shows the financial signals that triggered the alert."
                            )
                        ])
                    ], className="glass-card mb-3")
                ])
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_live_timeline()), md=12)
            ])
        ], fluid=True)

    if tab == "arena":
        temp = comparison_df.sort_values("cost").copy()
        temp_display = temp.copy()
        temp_display["cost"] = temp_display["cost"].apply(euro)
        temp_display = temp_display.round(4)

        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Model Arena", className="section-title"),
                    html.P(
                        "Models are compared as competing decision strategies. "
                        "The winner is not necessarily the model with the highest accuracy, "
                        "but the model with the lowest economic cost."
                    )
                ])
            ]),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_model_arena()), md=6),
                dbc.Col(dcc.Graph(figure=plot_calibration()), md=6),
            ]),

            dbc.Row([
                dbc.Col([
                    html.H4("Final Ranking", className="section-title mt-4"),
                    dash_table.DataTable(
                        data=temp_display.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in temp_display.columns],
                        page_size=10,
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": "#0b1020",
                            "color": "#00d4ff",
                            "fontWeight": "bold",
                            "border": "1px solid #1f4e79"
                        },
                        style_cell={
                            "backgroundColor": "#050816",
                            "color": "#e6edf3",
                            "border": "1px solid #1f4e79",
                            "fontSize": "13px",
                            "padding": "8px"
                        },
                    )
                ])
            ])
        ], fluid=True)

    if tab == "simulator":
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Cost Simulator", className="section-title"),
                            html.P(
                                "This simulator shows how the Bayes decision threshold reacts when the business costs change."
                            ),

                            html.Label("False Negative Cost c_FN"),
                            dcc.Slider(
                                id="cfn-slider",
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

                            html.Label("False Positive Cost c_FP"),
                            dcc.Slider(
                                id="cfp-slider",
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

                            html.Div(id="simulator-text", className="mt-4")
                        ])
                    ], className="glass-card")
                ], md=5),

                dbc.Col([
                    dcc.Graph(id="threshold-gauge")
                ], md=7)
            ])
        ], fluid=True)

    if tab == "alerts":
        alerts = alert_table_data()

        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Alert Investigation Room", className="section-title"),
                    html.P(
                        "This room is designed for analysts. It lists the most suspicious observations "
                        "and exposes the signals that contributed to the alert."
                    )
                ])
            ]),

            dbc.Row([
                dbc.Col([
                    dash_table.DataTable(
                        id="alerts-table",
                        data=alerts.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in alerts.columns],
                        page_size=12,
                        sort_action="native",
                        filter_action="native",
                        row_selectable="single",
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": "#0b1020",
                            "color": "#00d4ff",
                            "fontWeight": "bold",
                            "border": "1px solid #1f4e79"
                        },
                        style_cell={
                            "backgroundColor": "#050816",
                            "color": "#e6edf3",
                            "border": "1px solid #1f4e79",
                            "fontSize": "13px",
                            "padding": "8px"
                        },
                    )
                ], md=8),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Selected Alert Explanation", className="section-title"),
                            html.Div(id="alert-explanation", className="alert-panel")
                        ])
                    ], className="glass-card")
                ], md=4)
            ])
        ], fluid=True)

    if tab == "drift":
        top_drift = drift_feature_df.sort_values("psi", ascending=False).iloc[0]
        n_critical = int((drift_feature_df["severity"] == "critical").sum())
        n_label_drift = int(drift_label_df["label_drift_detected"].sum())
        n_recalibrated = len(drift_recalibration_df)

        drift_log_display = drift_log_df.tail(50).copy()

        return dbc.Container([
            dbc.Row([
                dbc.Col(card(
                    "Critical feature drift",
                    str(n_critical),
                    f"Top variable: {top_drift['feature']}",
                    "neon-red"
                ), md=3),

                dbc.Col(card(
                    "Highest PSI",
                    f"{top_drift['psi']:.3f}",
                    "PSI threshold: 0.2",
                    "neon-orange"
                ), md=3),

                dbc.Col(card(
                    "Label drift windows",
                    str(n_label_drift),
                    "Rolling positive-rate changes",
                    "neon-blue"
                ), md=3),

                dbc.Col(card(
                    "Recalibrated thresholds",
                    str(n_recalibrated),
                    "Triggered by feature drift",
                    "neon-green"
                ), md=3),
            ], className="g-3 mb-4"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Drift Monitoring Narrative", className="section-title"),
                            html.P(
                                "This module monitors whether the recent market distribution remains compatible "
                                "with the training distribution. It combines statistical drift detection, label drift, "
                                "business cost degradation, and automatic threshold recalibration."
                            ),
                            html.P(
                                "In the current run, the strongest drift appears on raw price and volume level variables. "
                                "However, the cost degradation monitor does not detect economic degradation yet. "
                                "The system still recalibrates thresholds because the feature drift is critical."
                            ),
                            html.Code("Reaction rule: PSI > 0.2 OR cost degradation > 20% → recalibrate threshold")
                        ])
                    ], className="glass-card")
                ])
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_drift_psi_ranking()), md=6),
                dbc.Col(dcc.Graph(figure=plot_label_drift_timeline()), md=6),
            ]),

            dbc.Row([
                dbc.Col(dcc.Graph(figure=plot_cost_degradation_monitor()), md=6),
                dbc.Col(dcc.Graph(figure=plot_threshold_recalibration()), md=6),
            ], className="mt-4"),

            dbc.Row([
                dbc.Col([
                    html.H4("Drift Event Log", className="section-title mt-4"),
                    dash_table.DataTable(
                        data=drift_log_display.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in drift_log_display.columns],
                        page_size=10,
                        sort_action="native",
                        filter_action="native",
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": "#0b1020",
                            "color": "#00d4ff",
                            "fontWeight": "bold",
                            "border": "1px solid #1f4e79"
                        },
                        style_cell={
                            "backgroundColor": "#050816",
                            "color": "#e6edf3",
                            "border": "1px solid #1f4e79",
                            "fontSize": "12px",
                            "padding": "8px",
                            "whiteSpace": "normal",
                            "height": "auto"
                        },
                    )
                ])
            ])
        ], fluid=True)

    if tab == "story":
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3("Executive Story", className="section-title"),
                            html.P(
                                "The project starts from a practical problem: market manipulation is rare, "
                                "but missing it is expensive. A standard classifier optimizes statistical errors. "
                                "This command center reframes the task as a business decision problem."
                            ),
                            html.P(
                                "The empirical winner is Random Forest + SMOTE because it detects all suspicious cases "
                                "in the test set. Its total cost is therefore limited to the investigation cost of false alerts."
                            ),
                            html.P(
                                "However, Threshold Moving remains central because it connects the decision threshold "
                                "to the cost matrix through Bayes decision theory. This makes the method more explainable "
                                "and more aligned with regulatory reasoning."
                            ),
                            html.H4("Final message", className="neon-green mt-3"),
                            html.P(
                                "SMOTE treats class imbalance. Threshold Moving treats economic decision-making. "
                                "The drift monitoring layer then transforms the model into a living ML system that can detect "
                                "distribution changes and react through threshold recalibration."
                            )
                        ])
                    ], className="glass-card")
                ], md=12)
            ])
        ], fluid=True)

    return html.Div("Unknown tab.")


# =====================================================
# SIMULATOR CALLBACK
# =====================================================

@app.callback(
    Output("simulator-text", "children"),
    Output("threshold-gauge", "figure"),
    Input("cfn-slider", "value"),
    Input("cfp-slider", "value")
)
def update_simulator(c_fn, c_fp):
    threshold = c_fp / (c_fn + c_fp)

    if threshold < 0.001:
        mode = "EXTREME SURVEILLANCE"
        color = "danger"
        explanation = "The system becomes very aggressive because missing manipulation is extremely expensive."
    elif threshold < 0.005:
        mode = "HIGH SURVEILLANCE"
        color = "warning"
        explanation = "The system remains cautious and triggers alerts at low probabilities."
    else:
        mode = "MODERATE SURVEILLANCE"
        color = "info"
        explanation = "The system becomes more selective because investigation cost is relatively higher."

    text = dbc.Alert([
        html.H4(mode),
        html.P(f"c_FN = {euro(c_fn)}"),
        html.P(f"c_FP = {euro(c_fp)}"),
        html.H3(f"Bayes threshold = {threshold:.6f}"),
        html.P(explanation)
    ], color=color)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=threshold,
        number={"valueformat": ".6f"},
        title={"text": "Bayes Decision Threshold"},
        gauge={
            "axis": {"range": [0, 0.01]},
            "bar": {"color": "#00d4ff"},
            "steps": [
                {"range": [0, 0.001], "color": "#4d0000"},
                {"range": [0.001, 0.005], "color": "#4d3b00"},
                {"range": [0.005, 0.01], "color": "#0b3d2e"},
            ],
        }
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=420
    )

    return text, fig


# =====================================================
# ALERT EXPLANATION CALLBACK
# =====================================================

@app.callback(
    Output("alert-explanation", "children"),
    Input("alerts-table", "selected_rows"),
)
def explain_alert(selected_rows):
    alerts = alert_table_data()

    if not selected_rows:
        row = alerts.iloc[0]
    else:
        row = alerts.iloc[selected_rows[0]]

    signals = []

    if abs(row["abnormal_return"]) > 2:
        signals.append("abnormal return")
    if row["abnormal_volume"] > 2:
        signals.append("abnormal volume")
    if row["volatility_ratio"] > 1.5:
        signals.append("volatility shock")
    if abs(row["price_zscore"]) > 2:
        signals.append("price deviation")

    if not signals:
        signals.append("combined weak signals")

    return [
        html.P([html.Strong("Date: "), str(row["Date"])]),
        html.P([html.Strong("Ticker: "), str(row["ticker"])]),
        html.P([html.Strong("Suspicion score: "), str(row["suspicion_score"])]),
        html.P([html.Strong("Triggered signals: "), ", ".join(signals)]),
        html.Hr(),
        html.P(
            "Business interpretation: this alert combines abnormal market behavior signals. "
            "If it is a real manipulation and the system misses it, the estimated false negative cost "
            f"is {euro(C_FN)}."
        )
    ]


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)