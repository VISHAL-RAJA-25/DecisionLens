"""dashboard.py - Streamlit dashboard for DecisionLens."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
SEV_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}


@st.cache_data
def load_csv(name):
    return pd.read_csv(PROCESSED / f"{name}.csv")


@st.cache_data
def load_json(name):
    return json.loads((PROCESSED / f"{name}.json").read_text())


def page_overview():
    k = load_json("kpi_summary")
    st.subheader("Business overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Delivered orders", f"{int(k['total_orders']):,}")
    c2.metric("Revenue", f"R$ {k['total_revenue'] / 1e6:.2f}M")
    c3.metric("Avg order value", f"R$ {k['avg_order_value']:.2f}")
    c4.metric("Freight share", f"{k['freight_share_pct']}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Late deliveries", f"{k['late_delivery_pct']}%")
    c6.metric("Avg delivery time", f"{k['avg_delivery_days']} days")
    c7.metric("Avg review score", f"{k['avg_review_score']} / 5")
    c8.metric("Repeat customers", f"{k['repeat_customer_pct']}%")

    left, right = st.columns(2)
    lvr = load_csv("late_vs_review")
    fig = px.bar(lvr, x="delivery", y="avg_review", text="avg_review",
                 color="delivery", title="Late delivery vs review score",
                 color_discrete_map={"Late": "#d9534f", "On time": "#5cb85c"})
    fig.update_layout(showlegend=False, yaxis_range=[0, 5])
    left.plotly_chart(fig, use_container_width=True)

    status = load_csv("order_status_mix")
    big = status[status["pct"] >= 1][["order_status", "orders"]]
    rest = status.loc[status["pct"] < 1, "orders"].sum()
    if rest:
        big = pd.concat([big, pd.DataFrame([{"order_status": "other", "orders": rest}])])
    fig2 = px.pie(big, names="order_status", values="orders",
                  title="Order status mix", hole=0.45)
    fig2.update_traces(textinfo="percent")
    right.plotly_chart(fig2, use_container_width=True)
    st.caption("Late orders score far lower than on-time orders: delivery "
               "performance is the main driver of customer satisfaction.")


def page_trends():
    st.subheader("Monthly trends (full months only)")
    trend = load_csv("monthly_trend")
    rev = load_csv("review_by_month")
    df = trend.merge(rev, on="month")

    st.plotly_chart(px.line(df, x="month", y="revenue", markers=True,
                            title="Monthly revenue (R$)"),
                    use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(px.bar(df, x="month", y="orders", title="Orders per month"),
                   use_container_width=True)
    b.plotly_chart(px.line(df, x="month", y="aov", markers=True,
                           title="Average order value (R$)"),
                   use_container_width=True)
    c, d = st.columns(2)
    c.plotly_chart(px.line(df, x="month", y="late_pct", markers=True,
                           title="Late delivery rate (%)"),
                   use_container_width=True)
    d.plotly_chart(px.line(df, x="month", y="avg_review", markers=True,
                           title="Average review score"),
                   use_container_width=True)


def page_geography():
    st.subheader("Performance by state")
    s = load_csv("state_performance")
    metric = st.selectbox("Metric", ["late_pct", "avg_review", "avg_delivery_days",
                                     "freight_pct", "revenue", "orders"])
    ordered = s.sort_values(metric, ascending=False)
    st.plotly_chart(px.bar(ordered, x="state", y=metric,
                           title=f"{metric} by state", color=metric,
                           color_continuous_scale="RdYlGn_r"),
                    use_container_width=True)
    st.plotly_chart(px.scatter(s, x="late_pct", y="avg_review", size="orders",
                               text="state", title="Late rate vs review score "
                               "(bubble size = orders)"),
                    use_container_width=True)
    st.dataframe(s, use_container_width=True, hide_index=True)


def page_sellers_categories():
    st.subheader("Categories")
    cat = load_csv("category_performance")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(cat.head(10), x="revenue", y="category",
                          orientation="h", title="Top 10 categories by revenue")
                   .update_yaxes(autorange="reversed"), use_container_width=True)
    low = cat.sort_values("avg_review").head(10)
    b.plotly_chart(px.bar(low, x="avg_review", y="category", orientation="h",
                          title="10 lowest-rated categories")
                   .update_yaxes(autorange="reversed"), use_container_width=True)

    st.subheader("Sellers")
    sellers = load_csv("seller_performance")
    min_orders = st.slider("Minimum orders", 30, int(sellers["orders"].max()), 30)
    view = sellers[sellers["orders"] >= min_orders]
    st.plotly_chart(px.scatter(view, x="late_pct", y="avg_review", size="revenue",
                               hover_name="seller_id", color="state",
                               title="Sellers: late rate vs review score"),
                    use_container_width=True)
    st.dataframe(view.sort_values("avg_review").head(25),
                 use_container_width=True, hide_index=True)
    st.caption("Table shows the 25 lowest-rated sellers for the chosen filter.")


def page_problems_ai():
    st.subheader("Detected problems")
    p = load_csv("problems")
    c1, c2 = st.columns(2)
    sev = c1.multiselect("Severity", ["High", "Medium", "Low"], default=["High"])
    types = c2.multiselect("Type", sorted(p["type"].unique()),
                           default=sorted(p["type"].unique()))
    view = p[p["severity"].isin(sev) & p["type"].isin(types)]
    st.write(f"{len(view)} of {len(p)} problems shown")
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.subheader("AI insights and recommendations")
    ai = load_json("ai_insights")
    st.info(ai["executive_summary"])
    for x in ai["insights"]:
        icon = SEV_ICON.get(x["priority"], "⚪")
        with st.expander(f"{icon} [{x['priority']}] {x['problem']}"):
            st.markdown(f"**Root cause:** {x['root_cause']}")
            st.markdown(f"**Recommendation:** {x['recommendation']}")
            st.markdown(f"**Expected impact:** {x['expected_impact']}")
    st.caption(f"Insights source: {ai.get('source', 'unknown')}. Numbers come from "
               "the detection rules; the AI only explains them.")


PAGES = {
    "Overview": page_overview,
    "Trends": page_trends,
    "Geography": page_geography,
    "Sellers & Categories": page_sellers_categories,
    "Problems & AI": page_problems_ai,
}


def main():
    st.set_page_config(page_title="DecisionLens", page_icon="📊", layout="wide")
    st.title("📊 DecisionLens")
    st.caption("Business analytics, problem detection and AI recommendations "
               "on the Olist Brazilian e-commerce dataset")

    page = st.sidebar.radio("Navigate", list(PAGES))
    st.sidebar.markdown("---")
    st.sidebar.caption("Data: Olist Brazilian E-Commerce (Kaggle), 2016-2018")

    try:
        PAGES[page]()
    except FileNotFoundError as e:
        st.error(f"Missing data file: {e.filename}\n\nRun the pipeline first: "
                 "cleaning.py, build_db.py, analytics.py, problem_detection.py, "
                 "ai_engine.py")