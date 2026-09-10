"""Streamlit home page."""
import streamlit as st

from ui_theme import inject_theme

st.set_page_config(
    page_title="Home · Churn & LTV",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()

st.markdown(
    """
<style>
.page-hero {
  margin: 0 0 1.25rem 0;
}
.page-hero-head {
  font-family: "Outfit", sans-serif;
  font-size: 1.75rem;
  font-weight: 700;
  color: #ffffff;
  padding: 0.95rem 1.2rem;
  background: linear-gradient(90deg, #0f766e 0%, #0369a1 55%, #1d4ed8 100%);
  border-radius: 14px;
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
}
.page-hero-body {
  margin-top: 0.75rem;
  background: #f1f5f9;
  color: #334155;
  font-size: 1.02rem;
  font-weight: 500;
  line-height: 1.45;
  padding: 0.9rem 1.2rem;
  border-radius: 14px;
  border-left: 4px solid #14b8a6;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
}
.home-callout {
  background: #f0f9ff;
  border: 1px solid #bae6fd;
  border-left: 4px solid #0284c7;
  border-radius: 12px;
  padding: 0.95rem 1.1rem;
  color: #0c4a6e;
  font-size: 0.95rem;
  line-height: 1.5;
  margin-bottom: 1.35rem;
}
.home-callout b { color: #075985; }
.home-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}
.home-tile {
  border-radius: 14px;
  padding: 1.2rem 1.25rem;
  border: 1px solid #e2e8f0;
  background: #ffffff;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
  min-height: 130px;
}
.home-tile .kicker {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #64748b;
  margin-bottom: 0.4rem;
}
.home-tile .name {
  font-family: "Outfit", sans-serif;
  font-size: 1.25rem;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 0.35rem;
}
.home-tile .desc {
  color: #64748b;
  font-size: 0.92rem;
  line-height: 1.4;
}
.home-tile.lookup { border-top: 4px solid #e11d48; }
.home-tile.cohort { border-top: 4px solid #0d9488; }
@media (max-width: 800px) {
  .home-grid { grid-template-columns: 1fr; }
}
</style>

<div class="page-hero">
  <div class="page-hero-head">Home</div>
  <div class="page-hero-body">
    Agent console for churn risk scoring, LTV estimates, and retention playbooks.
  </div>
</div>

<div class="home-callout">
  <b>How to use:</b> open <b>Customer Lookup</b> in the left sidebar to score one customer,
  or <b>High-Risk Cohort</b> to review unique high-risk IDs from recent predictions.
  LTV values are model estimates for segmentation — not accounting revenue.
</div>

<div class="home-grid">
  <div class="home-tile lookup">
    <div class="kicker">Page</div>
    <div class="name">Customer Lookup</div>
    <div class="desc">Score churn + LTV and get a retention action.</div>
  </div>
  <div class="home-tile cohort">
    <div class="kicker">Page</div>
    <div class="name">High-Risk Cohort</div>
    <div class="desc">Browse latest unique high-risk customers.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
