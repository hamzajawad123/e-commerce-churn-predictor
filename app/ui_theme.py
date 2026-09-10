"""Shared Streamlit theme and API helpers."""
import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
  font-family: "Source Sans 3", sans-serif;
}
.block-container {
  padding-top: 1.35rem;
  padding-bottom: 2.5rem;
  max-width: 1120px;
}

[data-testid="stSidebar"] {
  background: linear-gradient(185deg, #0b1220 0%, #15233b 55%, #1a3354 100%);
  border-right: 1px solid #2a3f5f;
}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label {
  color: #dbe7f5 !important;
}
[data-testid="stSidebarNav"] a {
  border-radius: 10px !important;
  margin: 0.2rem 0.35rem !important;
  font-weight: 600 !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: linear-gradient(90deg, #0d9488 0%, #0891b2 100%) !important;
  color: #fff !important;
}

.brand {
  font-family: "Outfit", sans-serif;
  font-size: 1.45rem;
  font-weight: 700;
  color: #f0f9ff;
  letter-spacing: -0.02em;
}
.brand-sub {
  color: #93c5fd;
  font-size: 0.82rem;
  margin: 0.2rem 0 1rem;
}

.hero-head {
  font-family: "Outfit", sans-serif;
  font-size: 1.85rem;
  font-weight: 700;
  color: #ecfeff;
  background: linear-gradient(120deg, #0f766e 0%, #0369a1 55%, #1d4ed8 100%);
  padding: 0.95rem 1.2rem;
  border-radius: 14px;
  margin-bottom: 0.55rem;
  box-shadow: 0 10px 24px rgba(3, 105, 161, 0.28);
}
.hero-sub {
  color: #1e293b !important;
  font-size: 1.05rem !important;
  font-weight: 600 !important;
  margin: 0.55rem 0 1.15rem !important;
  padding: 0.65rem 0.9rem !important;
  background: #f1f5f9 !important;
  border-radius: 10px !important;
  border-left: 4px solid #0d9488 !important;
  display: block !important;
}

.home-sub-card {
  margin: 0.7rem 0 1rem;
  padding: 1rem 1.15rem;
  border-radius: 14px;
  font-size: 1.05rem;
  font-weight: 600;
  color: #083344;
  background: linear-gradient(120deg, #a5f3fc 0%, #67e8f9 45%, #5eead4 100%);
  border: 1px solid #22d3ee;
  box-shadow: 0 8px 18px rgba(6, 182, 212, 0.22);
}
.home-info-card {
  margin-bottom: 1.25rem;
  padding: 1.05rem 1.2rem;
  border-radius: 14px;
  font-size: 0.98rem;
  line-height: 1.5;
  color: #1e3a8a;
  background: linear-gradient(125deg, #dbeafe 0%, #bfdbfe 40%, #c7d2fe 100%);
  border: 1px solid #60a5fa;
  border-left: 6px solid #2563eb;
  box-shadow: 0 8px 18px rgba(37, 99, 235, 0.16);
}
.home-info-card b { color: #1e40af; }
.home-nav-card {
  border-radius: 16px;
  padding: 1.25rem 1.3rem 1.4rem;
  min-height: 150px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
  border: 1px solid transparent;
}
.home-nav-card .label {
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.45rem;
}
.home-nav-card .value {
  font-family: "Outfit", sans-serif;
  font-size: 1.4rem;
  font-weight: 700;
  margin-bottom: 0.45rem;
}
.home-nav-card .desc { font-size: 0.95rem; line-height: 1.4; }
.home-nav-card.lookup {
  background: linear-gradient(155deg, #fecdd3 0%, #fda4af 35%, #fff1f2 100%);
  border-color: #fb7185;
  color: #881337;
}
.home-nav-card.lookup .label { color: #9f1239; }
.home-nav-card.lookup .desc { color: #9f1239; }
.home-nav-card.cohort {
  background: linear-gradient(155deg, #6ee7b7 0%, #34d399 35%, #ecfdf5 100%);
  border-color: #34d399;
  color: #064e3b;
}
.home-nav-card.cohort .label { color: #047857; }
.home-nav-card.cohort .desc { color: #065f46; }


.info-banner {
  background: linear-gradient(90deg, #ecfeff 0%, #e0f2fe 100%);
  border: 1px solid #67e8f9;
  border-left: 5px solid #0891b2;
  border-radius: 12px;
  padding: 0.95rem 1.1rem;
  color: #0e7490;
  font-size: 0.95rem;
  margin-bottom: 1.25rem;
}
.info-banner b { color: #155e75; }

.id-panel {
  background: linear-gradient(145deg, #0f172a 0%, #1e3a5f 100%);
  border: 1px solid #334155;
  border-radius: 16px;
  padding: 1.2rem 1.25rem 1.35rem;
  margin-bottom: 1.25rem;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18);
}
.id-panel-title {
  font-family: "Outfit", sans-serif;
  color: #7dd3fc;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 0.35rem;
}
.id-panel-hint {
  color: #cbd5e1;
  font-size: 0.9rem;
  margin-bottom: 0.85rem;
}

.metric-card {
  border-radius: 16px;
  padding: 1.2rem 1.25rem 1.3rem;
  min-height: 155px;
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
  border: 1px solid transparent;
}
.metric-card.churn {
  background: linear-gradient(160deg, #fff1f2 0%, #ffffff 55%);
  border-color: #fecdd3;
}
.metric-card.ltv {
  background: linear-gradient(160deg, #ecfdf5 0%, #ffffff 55%);
  border-color: #a7f3d0;
}
.metric-card .label {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #64748b;
  margin-bottom: 0.5rem;
}
.metric-card .value {
  font-family: "Outfit", sans-serif;
  font-size: 2.15rem;
  font-weight: 700;
  color: #0f172a;
  line-height: 1.1;
  margin-bottom: 0.75rem;
}
.badge {
  display: inline-block;
  padding: 0.3rem 0.75rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 700;
}
.badge-risk-high { background: #fecaca; color: #991b1b; }
.badge-risk-medium { background: #fed7aa; color: #9a3412; }
.badge-risk-low { background: #bbf7d0; color: #166534; }
.badge-ltv-high { background: #99f6e4; color: #115e59; }
.badge-ltv-low { background: #e2e8f0; color: #334155; }

.meter-track {
  height: 9px;
  background: #e2e8f0;
  border-radius: 999px;
  overflow: hidden;
  margin-top: 0.9rem;
}
.meter-fill { height: 100%; border-radius: 999px; }
.meter-high { background: linear-gradient(90deg, #f97316, #ef4444); }
.meter-medium { background: linear-gradient(90deg, #fbbf24, #f97316); }
.meter-low { background: linear-gradient(90deg, #34d399, #10b981); }

.rec-card {
  margin-top: 1.1rem;
  border-radius: 16px;
  padding: 1.25rem 1.4rem;
  border: 1px solid #e2e8f0;
}
.rec-card.urgency-high {
  border-left: 6px solid #ef4444;
  background: linear-gradient(100deg, #fff1f2 0%, #fff7ed 40%, #ffffff 100%);
}
.rec-card.urgency-medium {
  border-left: 6px solid #f97316;
  background: linear-gradient(100deg, #fff7ed 0%, #fefce8 45%, #ffffff 100%);
}
.rec-card.urgency-low {
  border-left: 6px solid #0d9488;
  background: linear-gradient(100deg, #f0fdfa 0%, #ecfeff 45%, #ffffff 100%);
}
.rec-kicker {
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: #64748b;
  margin-bottom: 0.35rem;
}
.rec-title {
  font-family: "Outfit", sans-serif;
  font-size: 1.25rem;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 0.45rem;
}
.rec-body { color: #334155; font-size: 0.95rem; line-height: 1.45; }

.q4-note {
  margin-top: 0.95rem;
  padding: 0.9rem 1.1rem;
  border-radius: 12px;
  background: #fef08a;
  border: 1px solid #eab308;
  color: #713f12;
  font-size: 0.95rem;
  font-weight: 700;
}

.empty-state {
  padding: 2rem;
  text-align: center;
  color: #475569;
  border: 1px dashed #94a3b8;
  border-radius: 14px;
  background: linear-gradient(180deg, #f8fafc, #eef2ff);
}
</style>
"""


def inject_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def sidebar_brand():
    with st.sidebar:
        st.markdown('<div class="brand">RetainDesk</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="brand-sub">Churn &amp; LTV agent console</div>',
            unsafe_allow_html=True,
        )


def risk_badge_class(tier: str) -> str:
    return {
        "High": "badge-risk-high",
        "Medium": "badge-risk-medium",
        "Low": "badge-risk-low",
    }.get(tier, "badge-risk-low")


def ltv_badge_class(tier: str) -> str:
    return "badge-ltv-high" if tier == "High" else "badge-ltv-low"


def risk_meter_class(tier: str) -> str:
    return {
        "High": "meter-high",
        "Medium": "meter-medium",
        "Low": "meter-low",
    }.get(tier, "meter-low")


def rec_urgency_class(tier: str) -> str:
    return {
        "High": "urgency-high",
        "Medium": "urgency-medium",
        "Low": "urgency-low",
    }.get(tier, "urgency-low")


def recommendation_context(risk: str, ltv: str, strategy: str, customer_id: int) -> str:
    """Customer-specific explanation — not a one-size disclaimer."""
    mapping = {
        ("High", "High"): (
            f"Customer #{customer_id} is both high-churn and high-value. "
            f"Prioritize a personal save path now: <b>{strategy}</b>."
        ),
        ("High", "Low"): (
            f"Customer #{customer_id} is high-churn but lower LTV. "
            f"Use a light-touch, low-cost save: <b>{strategy}</b>."
        ),
        ("Medium", "High"): (
            f"Customer #{customer_id} shows rising risk while remaining valuable. "
            f"Act early with: <b>{strategy}</b>."
        ),
        ("Medium", "Low"): (
            f"Customer #{customer_id} has moderate risk and lower LTV. "
            f"A gentle nudge is enough: <b>{strategy}</b>."
        ),
        ("Low", "High"): (
            f"Customer #{customer_id} looks stable and high-value. "
            f"Recommended stance: <b>{strategy}</b> — protect the relationship without over-discounting."
        ),
        ("Low", "Low"): (
            f"Customer #{customer_id} is low risk with lower LTV. "
            f"Recommended stance: <b>{strategy}</b> — no urgent retention spend."
        ),
    }
    return mapping.get(
        (risk, ltv),
        f"For customer #{customer_id} ({risk} risk / {ltv} LTV): <b>{strategy}</b>.",
    )


def fetch_prediction(customer_id: int):
    try:
        resp = requests.get(f"{API_BASE_URL}/predict_full/{customer_id}", timeout=120)
    except requests.exceptions.ConnectionError:
        return None, "Can't reach the API. Start uvicorn on port 8000 first."
    except requests.exceptions.ReadTimeout:
        return None, "API timed out. First prediction may still be loading models — wait and retry."
    if resp.status_code == 404:
        return None, resp.json().get("detail", "Customer not found.")
    if resp.status_code != 200:
        return None, f"API error ({resp.status_code}): {resp.text}"
    return resp.json(), None


def fetch_high_risk(limit: int):
    try:
        resp = requests.get(
            f"{API_BASE_URL}/high_risk_customers",
            params={"limit": limit},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        return None, "Can't reach the API. Start uvicorn on port 8000 first."
    if resp.status_code != 200:
        return None, f"API error ({resp.status_code}): {resp.text}"
    return resp.json(), None
