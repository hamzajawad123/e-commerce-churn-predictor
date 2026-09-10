"""High-risk cohort page."""
import html
from datetime import datetime

import pandas as pd
import streamlit as st

from ui_theme import fetch_high_risk, inject_theme

st.set_page_config(page_title="High-Risk Cohort · Churn & LTV", page_icon="◎", layout="wide")
inject_theme()

st.markdown('<div class="hero-head">High-Risk Cohort</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Latest unique customers scored as High churn risk (one row per Customer ID).</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-banner">
      <b>Note:</b> This list comes from predictions you already ran.
      Open <b>Customer Lookup</b> to rescore someone.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p style="font-size:1.15rem;font-weight:800;color:#ffffff;margin:0.35rem 0 0.35rem;">Max customers to show</p>',
    unsafe_allow_html=True,
)
limit = st.slider(
    "Max customers to show",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    label_visibility="collapsed",
)
if st.button("Refresh list", type="primary"):
    with st.spinner("Loading…"):
        rows, err = fetch_high_risk(limit)
    if err:
        st.error(err)
        st.stop()
    st.session_state["high_risk_rows"] = rows

rows = st.session_state.get("high_risk_rows")
if rows is None:
    st.markdown(
        '<div class="empty-state">Click <b>Refresh list</b> to load high-risk predictions.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if not rows:
    st.info("No high-risk predictions logged yet — score some customers on Customer Lookup first.")
    st.stop()

df = pd.DataFrame(rows)
df = df.sort_values("predicted_at", ascending=False).drop_duplicates(subset=["customer_id"], keep="first")


def _risk_badge(tier: str) -> str:
    cls = {
        "High": "tbl-badge risk-high",
        "Medium": "tbl-badge risk-medium",
        "Low": "tbl-badge risk-low",
    }.get(str(tier), "tbl-badge risk-low")
    return f'<span class="{cls}">{html.escape(str(tier))}</span>'


def _ltv_badge(tier: str) -> str:
    cls = "tbl-badge ltv-high" if str(tier) == "High" else "tbl-badge ltv-low"
    return f'<span class="{cls}">{html.escape(str(tier))}</span>'


def _fmt_when(value) -> str:
    text = str(value)
    try:
        # Format API timestamps for display
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return text[:19]


body_rows = []
for _, r in df.iterrows():
    churn_pct = f"{float(r['churn_probability']) * 100:.1f}%"
    ltv = f"${float(r['predicted_ltv']):,.2f}"
    body_rows.append(
        "<tr>"
        f"<td class='id'>{int(r['customer_id'])}</td>"
        f"<td class='pct'><div class='pct-wrap'><span>{churn_pct}</span>"
        f"<div class='pct-bar'><i style='width:"
        f"{min(100.0, float(r['churn_probability']) * 100):.1f}%'></i></div></div></td>"
        f"<td>{_risk_badge(r['risk_tier'])}</td>"
        f"<td class='money'>{ltv}</td>"
        f"<td>{_ltv_badge(r['ltv_tier'])}</td>"
        f"<td class='strategy'>{html.escape(str(r['retention_strategy']))}</td>"
        f"<td class='when'>{html.escape(_fmt_when(r['predicted_at']))}</td>"
        "</tr>"
    )

table_html = f"""
<style>
.cool-table-wrap {{
  margin-top: 0.85rem;
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid #334155;
  box-shadow: 0 14px 30px rgba(15, 23, 42, 0.18);
}}
.cool-table {{
  width: 100%;
  border-collapse: collapse;
  font-family: "Source Sans 3", sans-serif;
  background: #0f172a;
  color: #e2e8f0;
}}
.cool-table thead th {{
  background: linear-gradient(90deg, #0f766e 0%, #0369a1 55%, #1d4ed8 100%);
  color: #f8fafc;
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  text-align: left;
  padding: 0.95rem 0.9rem;
  border: none;
}}
.cool-table tbody tr {{
  background: #111827;
  border-bottom: 1px solid #1f2937;
  transition: background 0.15s ease;
}}
.cool-table tbody tr:nth-child(even) {{ background: #0b1220; }}
.cool-table tbody tr:hover {{ background: #1e293b; }}
.cool-table td {{
  padding: 0.9rem;
  vertical-align: middle;
  font-size: 0.95rem;
  border: none;
}}
.cool-table td.id {{
  font-weight: 800;
  color: #7dd3fc;
  font-variant-numeric: tabular-nums;
}}
.cool-table td.money {{
  font-weight: 700;
  color: #6ee7b7;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}}
.cool-table td.strategy {{ color: #cbd5e1; max-width: 280px; }}
.cool-table td.when {{ color: #94a3b8; white-space: nowrap; font-size: 0.85rem; }}
.pct-wrap {{ min-width: 110px; }}
.pct-wrap span {{ font-weight: 800; color: #fda4af; }}
.pct-bar {{
  margin-top: 0.35rem;
  height: 7px;
  background: #334155;
  border-radius: 999px;
  overflow: hidden;
}}
.pct-bar i {{
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #f97316, #ef4444);
  border-radius: 999px;
}}
.tbl-badge {{
  display: inline-block;
  padding: 0.28rem 0.7rem;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 800;
}}
.tbl-badge.risk-high {{ background: #fecaca; color: #991b1b; }}
.tbl-badge.risk-medium {{ background: #fed7aa; color: #9a3412; }}
.tbl-badge.risk-low {{ background: #bbf7d0; color: #166534; }}
.tbl-badge.ltv-high {{ background: #99f6e4; color: #115e59; }}
.tbl-badge.ltv-low {{ background: #e2e8f0; color: #334155; }}
</style>
<div class="cool-table-wrap">
<table class="cool-table">
  <thead>
    <tr>
      <th>Customer ID</th>
      <th>Churn Probability</th>
      <th>Risk Tier</th>
      <th>Predicted LTV</th>
      <th>LTV Tier</th>
      <th>Retention Strategy</th>
      <th>Predicted At</th>
    </tr>
  </thead>
  <tbody>
    {''.join(body_rows)}
  </tbody>
</table>
</div>
"""

st.markdown(table_html, unsafe_allow_html=True)
