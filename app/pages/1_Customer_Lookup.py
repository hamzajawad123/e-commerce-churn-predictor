"""Customer lookup page."""
import streamlit as st

from ui_theme import (
    fetch_prediction,
    inject_theme,
    ltv_badge_class,
    recommendation_context,
    rec_urgency_class,
    risk_badge_class,
    risk_meter_class,
)

st.set_page_config(page_title="Customer Lookup · Churn & LTV", page_icon="◎", layout="wide")
inject_theme()

st.markdown('<div class="hero-head">Customer Lookup</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Score one customer for churn probability, predicted LTV, and a retention action.</p>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-banner">
      <b>Before you score:</b> LTV is a model estimate for segmenting value (not verified revenue).
      Retention text is an <b>agent suggestion</b> based on risk × LTV — not an automated charge, ban, or discount send.
      Prefer reviewing high-churn + high-LTV customers first.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="id-panel">
      <div class="id-panel-title">Select customer</div>
      <div class="id-panel-hint">
        Enter any customer id from 0-49999 in the box below.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "customer_id_input" not in st.session_state:
    st.session_state["customer_id_input"] = 0

customer_id = st.number_input(
    "Customer ID",
    min_value=0,
    max_value=49999,
    step=1,
    key="customer_id_input",
    help="IDs match Customer_ID in the Feast online store / features.parquet",
)

run = st.button("Get prediction", type="primary", use_container_width=False)

if run:
    with st.spinner("Scoring customer…"):
        data, err = fetch_prediction(int(customer_id))
    if err:
        st.error(err)
    else:
        st.session_state["last_prediction"] = data

data = st.session_state.get("last_prediction")
if not data:
    st.markdown(
        '<div class="empty-state">Choose a customer ID above, then click <b>Get prediction</b>.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

risk = data["risk_tier"]
ltv_t = data["ltv_tier"]
proba = float(data["churn_probability"])
pct = max(0.0, min(100.0, proba * 100.0))
ctx = recommendation_context(risk, ltv_t, data["retention_strategy"], data["customer_id"])

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"""
        <div class="metric-card churn">
          <div class="label">Churn probability</div>
          <div class="value">{proba:.1%}</div>
          <span class="badge {risk_badge_class(risk)}">Risk · {risk}</span>
          <div class="meter-track"><div class="meter-fill {risk_meter_class(risk)}" style="width:{pct:.1f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    conf_note = " · lower confidence (Q4)" if data.get("ltv_confidence") == "low" else ""
    st.markdown(
        f"""
        <div class="metric-card ltv">
          <div class="label">Predicted LTV{conf_note}</div>
          <div class="value">${data['predicted_ltv']:,.2f}</div>
          <span class="badge {ltv_badge_class(ltv_t)}">LTV · {ltv_t}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="rec-card {rec_urgency_class(risk)}">
      <div class="rec-kicker">Recommended retention action</div>
      <div class="rec-title">{data['retention_strategy']}</div>
      <div class="rec-body">{ctx}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if data.get("ltv_confidence") == "low":
    st.markdown(
        """
        <div class="q4-note">
          This customer signed up in Q4. Lifetime-value estimates for Q4 signups are less
          precise — treat the dollar amount as a rough guide, not an exact figure.
        </div>
        """,
        unsafe_allow_html=True,
    )
