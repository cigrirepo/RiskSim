import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import triang, norm, lognorm, uniform
import plotly.express as px
import plotly.graph_objects as go
import json
# Optional: import numba for speed
# from numba import njit

# AI/NLP
import openai

# Exports
from fpdf import FPDF
import xlsxwriter

st.set_page_config(page_title="RiskSim360", layout="wide")

# ------------------------
# Helper functions
# ------------------------
def parse_assumptions(df):
    """
    Expect columns: ['Driver','Distribution','Param1','Param2','Param3']
    e.g. Distribution: 'triangular','normal','lognormal','uniform'
    Param: for triangular: (min, mode, max)
           for normal: (mean, std)
    """
    assumptions = []
    for _, row in df.iterrows():
        driver = row['Driver']
        dist = row['Distribution'].lower()
        params = [row['Param1'], row['Param2'], row['Param3']]
        assumptions.append({'driver': driver, 'dist': dist, 'params': params})
    return assumptions

# @njit  # Optional speed-up
def run_monte_carlo(assumptions, n_sims, correlation=None):
    """
    Run Monte Carlo sims. Return DataFrame of shape (n_sims, len(assumptions)).
    """
    sims = np.zeros((n_sims, len(assumptions)))
    for i, a in enumerate(assumptions):
        if a['dist']=='triangular':
            c = (a['params'][1] - a['params'][0])/(a['params'][2]-a['params'][0])
            sims[:,i] = triang(c, loc=a['params'][0], scale=(a['params'][2]-a['params'][0])).rvs(n_sims)
        elif a['dist']=='normal':
            sims[:,i] = norm(loc=a['params'][0], scale=a['params'][1]).rvs(n_sims)
        elif a['dist']=='lognormal':
            sims[:,i] = lognorm(s=a['params'][1], scale=np.exp(a['params'][0])).rvs(n_sims)
        elif a['dist']=='uniform':
            sims[:,i] = uniform(loc=a['params'][0], scale=(a['params'][1]-a['params'][0])).rvs(n_sims)
        else:
            sims[:,i] = np.nan
    # TODO: handle correlation via Cholesky if provided
    return pd.DataFrame(sims, columns=[a['driver'] for a in assumptions])


def calculate_npv(sim_df, cashflow_cols, discount_rate):
    """
    cashflow_cols: list of column names in sim_df representing CF t=0..T
    discount_rate: scalar
    """
    # Present Value of each period
    pv = sim_df[cashflow_cols].values / ((1+discount_rate) ** np.arange(len(cashflow_cols)))
    return pv.sum(axis=1)


def tornado_chart(impact_df):
    """
    Plot tornado chart given impact_df with columns ['Driver','Impact']
    """
    fig = px.bar(impact_df.sort_values('Impact'), x='Impact', y='Driver', orientation='h')
    return fig


def generate_risk_mermaid(drivers):
    """
    Build Mermaid syntax for risk workflow to send to Flowwmaid.
    e.g. drivers = ['Revenue Growth','COGS %','Discount Rate']
    """
    nodes = '\n'.join([f"    {i+1}[{d}]" for i,d in enumerate(drivers)])
    edges = '\n'.join([f"    {i+1} --> Outcome" for i in range(len(drivers))])
    mermaid = f"```mermaid\nflowchart LR\n{nodes}\n    Outcome((Net Present Value))\n{edges}\n```"
    return mermaid


def generate_narrative(findings):
    """
    Call OpenAI GPT to generate a report summary from findings dict.
    """
    prompt = (
        "Write a concise executive summary of the following risk analysis results:\n"
        + json.dumps(findings)
    )
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=250,
        temperature=0.5
    )
    return response.choices[0].text.strip()


def export_pdf(charts, narrative):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, narrative)
    # TODO: embed charts as images
    pdf.output("RiskSim360_Report.pdf")
    st.success("PDF report generated: RiskSim360_Report.pdf")


def main():
    st.title("RiskSim360: Monte Carlo Risk Simulator")

    # Sidebar inputs
    st.sidebar.header("Inputs & Settings")
    uploaded = st.sidebar.file_uploader("Upload Assumptions (CSV)", type=["csv","xlsx"])
    n_sims = st.sidebar.number_input("# Simulations", min_value=1000, max_value=100000, value=20000, step=1000)
    discount_rate = st.sidebar.number_input("Discount Rate", min_value=0.0, max_value=1.0, value=0.1)
    run_button = st.sidebar.button("Run Simulation")

    if uploaded and run_button:
        # Load data
        if uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        assumptions = parse_assumptions(df)
        sim_df = run_monte_carlo(assumptions, int(n_sims))

        # Calculate NPV
        # Assume cashflow drivers are named 'CF0','CF1',... etc
        cf_cols = [d['driver'] for d in assumptions]
        npv_series = calculate_npv(sim_df, cf_cols, discount_rate)

        # Display histogram
        fig = px.histogram(npv_series, nbins=50, title="NPV Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Tornado chart: compute driver impacts
        impacts = []
        base_npv = npv_series.mean()
        for driver in cf_cols:
            # perturb each driver by +/- 1 std dev
            pert = sim_df.copy()
            pert[driver] += sim_df[driver].std()
            pert_npv = calculate_npv(pert, cf_cols, discount_rate)
            impacts.append({'Driver': driver, 'Impact': (pert_npv.mean() - base_npv)})
        impact_df = pd.DataFrame(impacts)
        tornado_fig = tornado_chart(impact_df)
        st.plotly_chart(tornado_fig, use_container_width=True)

        # Mermaid risk workflow text
        mermaid_md = generate_risk_mermaid(cf_cols)
        st.markdown("### Risk Workflow Diagram")
        st.code(mermaid_md, language='markdown')

        # Optional: send mermaid to Flowwmaid via API (not implemented)
        if st.checkbox("Send to Flowwmaid for visualization"): 
            st.info("Feature coming soon: integration with Flowwmaid API...")

        # AI narrative
        if st.checkbox("Generate AI Narrative"): 
            findings = {
                'Base NPV': round(base_npv,2),
                'Mean NPV': round(npv_series.mean(),2),
                'Std NPV': round(npv_series.std(),2),
                'Probability NPV < 0': round((npv_series<0).mean()*100,2)
            }
            narrative = generate_narrative(findings)
            st.subheader("Executive Summary")
            st.write(narrative)

        # Export
        if st.button("Export Report as PDF"): 
            export_pdf([fig, tornado_fig], narrative if 'narrative' in locals() else "")

if __name__ == "__main__":
    main()

