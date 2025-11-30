# interactive_dashboard.py
"""
Interactive Student Performance Dashboard
- Tabs UI
- Plotly visualizations
- Workflow/Sankey pipeline diagram
- Model comparison & best-model analysis
- Uses files:
    - processed_student_data.csv
    - model_results_summary.json
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_dataset(csv_path: str):
    p = Path(csv_path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df

@st.cache_data
def load_json(json_path: str):
    p = Path(json_path)
    if not p.exists():
        return None
    with open(p, 'r') as f:
        return json.load(f)

def safe_get_metrics(results_json):
    """
    Normalize possible JSON structures to a dict like:
    {
      "RQ1": {...models...},
      "RQ2": {...},
      "RQ3": {...},
      "Best Models": {...}
    }
    """
    # Try known keys first
    mapping = {}
    # common variants from earlier runs
    if not results_json:
        return {}

    # If user saved keys 'rq1_results' etc.
    if 'rq1_results' in results_json or 'rq2_results' in results_json or 'rq3_results' in results_json:
        mapping['RQ1'] = results_json.get('rq1_results', results_json.get('RQ1', {}))
        mapping['RQ2'] = results_json.get('rq2_results', results_json.get('RQ2', {}))
        mapping['RQ3'] = results_json.get('rq3_results', results_json.get('RQ3', {}))
        mapping['Best Models'] = results_json.get('best_models', results_json.get('Best Models', results_json.get('best_models', {})))
        mapping['Comparison Table'] = pd.DataFrame(results_json.get('comparison_table', {})) if results_json.get('comparison_table') else None
        return mapping

    # If top-level keys look like 'RQ1 - Midterm I' in a comparison table
    if 'RQ1 - Midterm I' in results_json or any(k.startswith('RQ') for k in results_json.keys()):
        # best-effort pass-through
        for k, v in results_json.items():
            mapping[k] = v
        return mapping

    # If file is the structure used earlier in canvas (RQ1, RQ2, RQ3)
    if 'RQ1' in results_json:
        mapping['RQ1'] = results_json['RQ1']
        mapping['RQ2'] = results_json['RQ2']
        mapping['RQ3'] = results_json['RQ3']
        mapping['Best Models'] = results_json.get('Best Models', results_json.get('best_models', {}))
        return mapping

    # fallback: return everything
    return results_json

def flatten_comparison_table(comp_df):
    """If comparison table saved as dict-of-lists convert to DataFrame"""
    if comp_df is None:
        return None
    if isinstance(comp_df, dict):
        try:
            return pd.DataFrame(comp_df)
        except:
            return None
    if isinstance(comp_df, list):
        return pd.DataFrame(comp_df)
    if isinstance(comp_df, pd.DataFrame):
        return comp_df
    return None

# -------------------------
# Load files (sidebar)
# -------------------------
st.sidebar.header("Files & Settings")
csv_default = "processed_student_data.csv"
json_default = "model_results_summary.json"

csv_path = st.sidebar.text_input("Processed CSV path", csv_default)
json_path = st.sidebar.text_input("Model results JSON path", json_default)

uploaded_csv = st.sidebar.file_uploader("Or upload processed_student_data.csv", type=['csv'])
uploaded_json = st.sidebar.file_uploader("Or upload model_results_summary.json", type=['json'])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
else:
    df = load_dataset(csv_path)

if uploaded_json:
    results_json_raw = json.load(uploaded_json)
else:
    results_json_raw = load_json(json_path)

if df is None:
    st.sidebar.error(f"Could not find dataset at '{csv_path}'. Upload file or correct path.")
if results_json_raw is None:
    st.sidebar.warning(f"Could not find JSON results at '{json_path}'. Upload file or correct path.")

results_meta = safe_get_metrics(results_json_raw)

# -------------------------
# Layout: Tabs
# -------------------------
tabs = st.tabs(["Overview", "Dataset Explorer", "EDA", "Feature Relationships", "Model Comparison", "Best Models", "Workflow"])

# ---------- Overview ----------
with tabs[0]:
    st.header("📌 Overview")
    st.markdown("""
    This dashboard visualizes your student-performance pipeline and model results.
    Use the left sidebar to upload `processed_student_data.csv` and `model_results_summary.json`
    if they are not in the current working directory.
    """)
    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Students (rows)", f"{df.shape[0]}")
        col2.metric("Numeric features", f"{len(df.select_dtypes(include=[np.number]).columns)}")
        col3.metric("Columns", f"{len(df.columns)}")
        st.markdown("**Top 10 columns:**")
        st.write(list(df.columns[:10]))

    st.markdown("---")
    st.markdown("### Quick instructions")
    st.markdown("""
    - Navigate tabs for detail views.  
    - `Model Comparison` uses the JSON model results.  
    - `Workflow` visualizes the processing pipeline.  
    """)

# ---------- Dataset Explorer ----------
with tabs[1]:
    st.header("📄 Dataset Explorer")
    if df is None:
        st.warning("Dataset not loaded yet. Upload or provide the CSV file path in the sidebar.")
    else:
        st.subheader("Preview & Download")
        st.dataframe(df.head(50), use_container_width=True)
        csv_download = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download processed_student_data.csv", csv_download, "processed_student_data.csv", "text/csv")

        st.subheader("Column selector")
        cols = st.multiselect("Select columns to show", df.columns.tolist(), default=df.columns[:8].tolist())
        st.dataframe(df[cols].head(200), use_container_width=True)

        st.subheader("Summary statistics (numeric)")
        st.dataframe(df.describe().T)

# ---------- EDA ----------
with tabs[2]:
    st.header("🔎 Exploratory Data Analysis (interactive)")
    if df is None:
        st.warning("Dataset not loaded yet.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col1, col2 = st.columns([2,1])
        with col1:
            feature = st.selectbox("Choose numeric feature for histogram/box", num_cols, index=0)
            nbins = st.slider("Bins", 10, 100, 30)
            fig_hist = px.histogram(df, x=feature, nbins=nbins, title=f"Distribution of {feature}")
            st.plotly_chart(fig_hist, use_container_width=True)

            fig_box = px.box(df, y=feature, points="all", title=f"Boxplot of {feature}")
            st.plotly_chart(fig_box, use_container_width=True)

        with col2:
            st.subheader("Correlation with targets")
            targets = [c for c in ['Midterm_I', 'Midterm_II', 'Final_Exam'] if c in df.columns]
            if targets:
                target = st.selectbox("Select target", targets)
                corrs = df.corr()[target].sort_values(ascending=False)
                st.dataframe(corrs.reset_index().rename(columns={'index':'feature', target:'corr'}) )
            else:
                st.info("No common target columns found (Midterm_I/Midterm_II/Final_Exam).")

        st.markdown("---")
        st.subheader("Correlation heatmap (numeric columns)")
        corr_df = df.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(corr_df, text_auto=True, aspect="auto", title="Correlation matrix (numeric)")
        st.plotly_chart(fig_corr, use_container_width=True)

# ---------- Feature Relationships ----------
with tabs[3]:
    st.header("🔗 Feature Relationships")
    if df is None:
        st.warning("Dataset not loaded.")
    else:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        left, right = st.columns(2)
        with left:
            xcol = st.selectbox("X feature", num_cols, index=0)
            color_col = st.selectbox("Color by (optional)", [None] + df.columns.tolist(), index=0)
        with right:
            ycol = st.selectbox("Y / Target", [c for c in ['Final_Exam','Midterm_II','Midterm_I'] if c in num_cols] + [c for c in num_cols if c not in ['Final_Exam','Midterm_II','Midterm_I']], index=0)

        fig_scatter = px.scatter(df, x=xcol, y=ycol, color=(None if color_col is None else df[color_col]), trendline="ols", title=f"{xcol} vs {ycol}")
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### Pairplot (selected subset)")
        sample = df[num_cols].sample(n=min(200, len(df)), random_state=1)
        fig_pairs = px.scatter_matrix(sample, dimensions=sample.columns[:6], title="Scatter matrix (first 6 numeric features)")
        st.plotly_chart(fig_pairs, use_container_width=True)

# ---------- Model Comparison ----------
with tabs[4]:
    st.header("📊 Model Comparison (Plotly)")
    if not results_meta:
        st.warning("Model results JSON not loaded. Upload or give correct path in sidebar.")
    else:
        # Attempt to build a comparison DataFrame
        # First look for 'comparison_table'
        comp_df = None
        if isinstance(results_meta.get('Comparison Table', None), pd.DataFrame):
            comp_df = results_meta['Comparison Table']
        else:
            comp_df = flatten_comparison_table(results_meta.get('Comparison Table'))

        # Fallback: try to construct from RQ dictionaries
        if comp_df is None:
            rows = []
            # For RQ1/2/3 keys
            for rq_key, rq_val in [('RQ1', results_meta.get('RQ1')), ('RQ2', results_meta.get('RQ2')), ('RQ3', results_meta.get('RQ3'))]:
                if not rq_val:
                    continue
                for model_name, metrics in rq_val.items():
                    # metrics may contain train/test_mae/test_rmse/test_r2 depending on format
                    # Try multiple key names
                    test_mae = metrics.get('test_mae') or metrics.get('Test MAE') or metrics.get('Test_MAE') or metrics.get('MAE')
                    test_rmse = metrics.get('test_rmse') or metrics.get('Test RMSE') or metrics.get('Test_RMSE') or metrics.get('RMSE')
                    test_r2 = metrics.get('test_r2') or metrics.get('Test R²') or metrics.get('Test R2') or metrics.get('R2')
                    rows.append({
                        'Research Question': rq_key,
                        'Model': model_name,
                        'Test MAE': test_mae,
                        'Test RMSE': test_rmse,
                        'Test R2': test_r2
                    })
            if rows:
                comp_df = pd.DataFrame(rows)

        if comp_df is None or comp_df.empty:
            st.info("Could not auto-build a comparison table from the JSON. You can still explore raw JSON in the 'Best Models' tab.")
            st.write(results_json_raw)
        else:
            st.subheader("Comparison table")
            st.dataframe(comp_df.style.format({"Test MAE":"{:.3f}", "Test RMSE":"{:.3f}", "Test R2":"{:.3f}"}), use_container_width=True)

            metric = st.selectbox("Select metric to visualize", ["Test MAE", "Test RMSE", "Test R2"])
            fig_bar = px.bar(comp_df, x='Model', y=metric, color='Research Question', barmode='group', title=f"Models comparison by {metric}")
            st.plotly_chart(fig_bar, use_container_width=True)

            # Show CI ribbons if available in raw JSON for best models
            st.markdown("#### Confidence Intervals for MAE (if available)")
            ci_rows = []
            for rq_key in ['RQ1','RQ2','RQ3']:
                rq = results_meta.get(rq_key)
                if not rq:
                    continue
                for mname,m in rq.items():
                    lower = m.get('mae_ci_lower') or m.get('MAE CI Lower') or m.get('mae_ci_low')
                    upper = m.get('mae_ci_upper') or m.get('MAE CI Upper') or m.get('mae_ci_up')
                    if lower is not None and upper is not None:
                        ci_rows.append({'RQ': rq_key, 'Model': mname, 'CI Lower': lower, 'CI Upper': upper, 'Test MAE': m.get('test_mae')})
            if ci_rows:
                ci_df = pd.DataFrame(ci_rows)
                st.dataframe(ci_df)
                fig_ci = go.Figure()
                for _, r in ci_df.iterrows():
                    fig_ci.add_trace(go.Scatter(
                        x=[r['RQ']],
                        y=[r['Test MAE']],
                        error_y=dict(type='data', symmetric=False, array=[r['CI Upper'] - r['Test MAE']], arrayminus=[r['Test MAE'] - r['CI Lower']]),
                        mode='markers+lines',
                        name=f"{r['Model']} ({r['RQ']})"
                    ))
                fig_ci.update_layout(title="MAE with 95% CI (if available)")
                st.plotly_chart(fig_ci, use_container_width=True)

# ---------- Best Models ----------
with tabs[5]:
    st.header("🏆 Best Models & Details")
    if not results_meta:
        st.warning("No model results available.")
    else:
        st.subheader("Raw JSON (click to expand)")
        st.write(results_json_raw)

        st.markdown("---")
        st.subheader("Best models detected in JSON")
        best_models = results_meta.get('Best Models') or results_meta.get('best_models') or results_meta.get('best_models', {})
        if not best_models:
            st.info("No explicit 'best_models' entry in JSON. The comparison table can help identify best models.")
        else:
            try:
                st.json(best_models)
            except:
                st.write(best_models)

        st.markdown("#### Train vs Test metrics for chosen model")
        # Allow user to pick RQ and model to inspect metrics
        rq_choice = st.selectbox("Select RQ", [k for k in ['RQ1','RQ2','RQ3'] if k in results_meta.keys()] or list(results_meta.keys()))
        model_choice = st.selectbox("Select Model", list(results_meta.get(rq_choice, {}).keys()) if results_meta.get(rq_choice) else [])
        if rq_choice and model_choice:
            metrics = results_meta.get(rq_choice, {}).get(model_choice, {})
            if metrics:
                # Print train/test metrics if available
                train_mae = metrics.get('train_mae') or metrics.get('Train MAE') or metrics.get('train_MAE')
                test_mae = metrics.get('test_mae') or metrics.get('Test MAE') or metrics.get('test_MAE')
                train_rmse = metrics.get('train_rmse') or metrics.get('Train RMSE')
                test_rmse = metrics.get('test_rmse') or metrics.get('Test RMSE')
                train_r2 = metrics.get('train_r2') or metrics.get('Train R²') or metrics.get('Train R2')
                test_r2 = metrics.get('test_r2') or metrics.get('Test R²') or metrics.get('Test R2')
                st.write(f"**Train MAE:** {train_mae}  •  **Test MAE:** {test_mae}")
                st.write(f"**Train RMSE:** {train_rmse}  •  **Test RMSE:** {test_rmse}")
                st.write(f"**Train R²:** {train_r2}  •  **Test R²:** {test_r2}")
            else:
                st.info("No metrics found for this selection.")

# ---------- Workflow ----------
with tabs[6]:
    st.header("🔁 Workflow / Pipeline Diagram")
    st.markdown("This Sankey diagram shows the high-level steps of the data-processing & modeling pipeline.")
    steps = [
        "Load Excel Sheets",
        "Combine Sheets",
        "Drop metadata Rows",
        "Impute Missing Values",
        "Feature Engineering",
        "Train/Test Split",
        "Modeling (SLR / MLR / Dummy)",
        "Bootstrap CI",
        "Evaluate & Compare",
        "Save Outputs (CSV/JSON)"
    ]
    # Create a simple flow: each step -> next step
    nodes = steps
    node_indices = list(range(len(nodes)))
    source = node_indices[:-1]
    target = node_indices[1:]
    value = [1]*len(source)
    fig_sankey = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = nodes,
        ),
        link = dict(
          source = source,
          target = target,
          value = value
      ))])
    fig_sankey.update_layout(title_text="Pipeline Sankey Diagram", font_size=12)
    st.plotly_chart(fig_sankey, use_container_width=True)

    st.markdown("**Legend / Notes:**\n- Imputation: median for numeric features in your pipeline.\n- Feature selection: Score_1..Score_14 used for Midterm I, etc.\n- Bootstrapping: 500 samples on train set to estimate MAE 95% CI.")

# -------------------------
# End of app
# -------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Need help running this in Colab?\n- Install: `pip install streamlit pyngrok`\n- Create `dashboard.py` and run with ngrok or localtunnel as described in the instructions provided earlier.")

