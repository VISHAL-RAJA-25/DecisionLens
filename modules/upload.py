"""upload.py - analyze any company's data through column mapping."""
import io
import json

import pandas as pd
import plotly.express as px
import streamlit as st

NONE = "— not available —"
SEV_ORDER = {"High": 0, "Medium": 1, "Low": 2}
SEV_ICON = {"High": "🔴", "Medium": "🟠", "Low": "🟡"}

# field: (label, required, keywords used to guess the column)
FIELDS = {
    "order_id": ("Order ID *", True,
                 ["order_id", "order id", "orderid", "order no", "invoice no",
                  "invoiceno", "invoice", "transaction"]),
    "order_date": ("Order date *", True,
                   ["order_date", "order date", "purchase", "invoicedate",
                    "date", "timestamp"]),
    "amount": ("Order amount (or map quantity + unit price)", False,
               ["amount", "total", "sales", "revenue", "payment"]),
    "quantity": ("Quantity (only if no amount)", False,
                 ["quantity", "qty", "units"]),
    "unit_price": ("Unit price (only if no amount)", False,
                   ["unit_price", "unit price", "unitprice", "price"]),
    "customer_id": ("Customer ID", False,
                    ["customer_id", "customer id", "customerid",
                     "customer unique", "customer", "client", "user"]),
    "category": ("Product category", False,
                 ["category", "segment", "department", "product"]),
    "region": ("Region / state / city", False,
               ["region", "state", "province", "city", "country"]),
    "delivered_date": ("Delivered date", False,
                       ["delivered_date", "delivered", "delivery", "ship date",
                        "shipped"]),
    "promised_date": ("Promised / estimated delivery date", False,
                      ["promised", "estimated", "expected"]),
    "rating": ("Customer rating / review score", False,
               ["rating", "review_score", "review", "stars", "score"]),
}


# ---------------------------------------------------------------- reading
@st.cache_data(show_spinner="Reading file...")
def read_upload(name, data):
    buf = io.BytesIO(data)
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)  # first sheet only
    try:
        return pd.read_csv(buf)
    except UnicodeDecodeError:
        buf.seek(0)
        return pd.read_csv(buf, encoding="latin-1")


def guess_mapping(columns):
    mapping, used = {}, set()
    for field, (_, _, keys) in FIELDS.items():
        found = None
        for key in keys:
            for col in columns:
                if col not in used and key in str(col).lower():
                    found = col
                    break
            if found:
                break
        mapping[field] = found
        if found:
            used.add(found)
    return mapping


# --------------------------------------------------------------- cleaning
def _num(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    cleaned = s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def standardize(raw, mapping):
    """Rename the user's columns to the standard schema and clean them."""
    df = pd.DataFrame({f: raw[c].values for f, c in mapping.items() if c})

    df["order_id"] = df["order_id"].astype(str).str.strip()
    df.loc[df["order_id"].str.lower().isin(["nan", "none", ""]), "order_id"] = pd.NA
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    for c in ("delivered_date", "promised_date"):
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    if "amount" in df:
        df["amount"] = _num(df["amount"])
    else:
        df["amount"] = _num(df["quantity"]) * _num(df["unit_price"])
    if "rating" in df:
        df["rating"] = _num(df["rating"])

    for c in ("category", "region"):
        if c in df:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].str.lower().isin(["nan", "none", ""]), c] = "Unknown"
    if "customer_id" in df:
        df["customer_id"] = df["customer_id"].astype(str).str.strip()
        df.loc[df["customer_id"].str.lower().isin(["nan", "none", ""]),
               "customer_id"] = pd.NA

    n0 = len(df)
    df = df.drop_duplicates()
    n1 = len(df)
    df = df.dropna(subset=["order_id", "order_date", "amount"])
    df = df[df["amount"] > 0]
    note = (f"{n0:,} rows read, {n0 - n1:,} duplicate rows removed, "
            f"{n1 - len(df):,} rows dropped (missing ID/date/amount, or amount "
            f"<= 0 such as returns).")
    return df.reset_index(drop=True), note


# --------------------------------------------------------------- analysis
def analyze(df):
    df = df.copy()
    df["month"] = df["order_date"].dt.to_period("M").astype(str)
    orders = df.drop_duplicates("order_id")

    n_orders = int(df["order_id"].nunique())
    revenue = float(df["amount"].sum())
    k = {"total_orders": n_orders, "total_revenue": round(revenue, 2),
         "avg_order_value": round(revenue / n_orders, 2)}

    if "customer_id" in orders:
        cust = (orders.dropna(subset=["customer_id"])
                .groupby("customer_id")["order_id"].nunique())
        if len(cust):
            k["unique_customers"] = int(len(cust))
            k["repeat_customer_pct"] = round(float((cust > 1).mean() * 100), 2)

    if "delivered_date" in orders:
        d = orders.dropna(subset=["delivered_date"])
        if len(d):
            days = (d["delivered_date"] - d["order_date"]).dt.total_seconds() / 86400
            k["avg_delivery_days"] = round(float(days.mean()), 1)

    both = None
    if "delivered_date" in orders and "promised_date" in orders:
        both = orders.dropna(subset=["delivered_date", "promised_date"]).copy()
        if len(both):
            both["is_late"] = (both["delivered_date"] > both["promised_date"]).astype(int)
            k["late_delivery_pct"] = round(float(both["is_late"].mean() * 100), 2)
        else:
            both = None

    if "rating" in orders and orders["rating"].notna().any():
        k["avg_rating"] = round(float(orders["rating"].mean()), 2)

    # monthly trend; months with very few orders are treated as partial
    monthly = (df.groupby("month")
               .agg(orders=("order_id", "nunique"), revenue=("amount", "sum"))
               .reset_index())
    monthly["revenue"] = monthly["revenue"].round(2)
    monthly["aov"] = (monthly["revenue"] / monthly["orders"]).round(2)
    monthly["partial"] = monthly["orders"] < 0.2 * monthly["orders"].median()
    if both is not None:
        late_m = (both.groupby("month")["is_late"].mean().mul(100).round(1)
                  .rename("late_pct").reset_index())
        monthly = monthly.merge(late_m, on="month", how="left")
    full = monthly[~monthly["partial"]].copy()
    full["revenue_growth_pct"] = (full["revenue"].pct_change() * 100).round(1)

    cat = None
    if "category" in df:
        g = df.groupby("category")
        cat = g.agg(orders=("order_id", "nunique"),
                    revenue=("amount", "sum")).reset_index()
        if "rating" in df:
            r = g["rating"].mean().round(2).rename("avg_rating").reset_index()
            cat = cat.merge(r, on="category", how="left")
        cat = cat.sort_values("revenue", ascending=False).reset_index(drop=True)

    reg = None
    if "region" in df:
        reg = (df.groupby("region")
               .agg(orders=("order_id", "nunique"), revenue=("amount", "sum"))
               .reset_index())
        if both is not None:
            late_r = (both.groupby("region")["is_late"].mean().mul(100).round(1)
                      .rename("late_pct").reset_index())
            reg = reg.merge(late_r, on="region", how="left")
        if "rating" in orders:
            rr = (orders.groupby("region")["rating"].mean().round(2)
                  .rename("avg_rating").reset_index())
            reg = reg.merge(rr, on="region", how="left")
        reg = reg.sort_values("revenue", ascending=False).reset_index(drop=True)

    problems = detect(k, full, cat, reg)
    return {"kpis": k, "monthly": monthly, "full": full, "cat": cat,
            "reg": reg, "problems": problems}


def _p(ptype, entity, metric, value, benchmark, severity, detail):
    return {"type": ptype, "entity": str(entity), "metric": metric,
            "value": round(float(value), 2), "benchmark": round(float(benchmark), 2),
            "severity": severity, "detail": detail}


def detect(k, full, cat, reg):
    out = []
    min_n = max(10, 0.002 * k["total_orders"])

    for _, r in full[full["revenue_growth_pct"] <= -15].iterrows():
        g = r["revenue_growth_pct"]
        out.append(_p("Revenue drop", r["month"], "revenue_growth_pct", g, 0,
                      "Medium" if g <= -30 else "Low",
                      f"Revenue fell {abs(g)}% vs the previous month "
                      f"(check seasonality)."))

    if "late_pct" in full and full["late_pct"].notna().sum() >= 4:
        m, s = full["late_pct"].mean(), full["late_pct"].std()
        if s > 0:
            for _, r in full[full["late_pct"] > m + 1.5 * s].iterrows():
                out.append(_p("Late delivery spike", r["month"], "late_pct",
                              r["late_pct"], m, "High",
                              f"Late rate hit {r['late_pct']}% vs a monthly "
                              f"average of {m:.1f}%."))

    if reg is not None and "late_pct" in reg and "late_delivery_pct" in k:
        o = k["late_delivery_pct"]
        hit = reg[(reg["orders"] >= min_n) & (reg["late_pct"] >= max(o * 1.5, o + 5))]
        for _, r in hit.iterrows():
            sev = "High" if r["late_pct"] >= o * 2 else "Medium"
            out.append(_p("Late delivery by region", r["region"], "late_pct",
                          r["late_pct"], o, sev,
                          f"{r['late_pct']}% of orders late vs {o}% overall "
                          f"({int(r['orders'])} orders)."))

    if cat is not None and "avg_rating" in cat and "avg_rating" in k:
        o = k["avg_rating"]
        hit = cat[(cat["orders"] >= min_n) & (cat["avg_rating"] <= o - 0.35)]
        for _, r in hit.iterrows():
            sev = "High" if r["avg_rating"] <= o - 0.5 else "Medium"
            out.append(_p("Low-rated category", r["category"], "avg_rating",
                          r["avg_rating"], o, sev,
                          f"Average rating {r['avg_rating']} vs {o} overall "
                          f"({int(r['orders'])} orders)."))

    if "repeat_customer_pct" in k and k["repeat_customer_pct"] < 10:
        rate = k["repeat_customer_pct"]
        out.append(_p("Low customer retention", "All customers",
                      "repeat_customer_pct", rate, 10,
                      "High" if rate < 5 else "Medium",
                      f"Only {rate}% of {k['unique_customers']:,} customers "
                      f"ordered more than once."))

    for label, tbl, col in (("category", cat, "category"), ("region", reg, "region")):
        if tbl is not None and len(tbl) > 1:
            top = tbl.iloc[0]
            share = 100 * top["revenue"] / k["total_revenue"]
            if share > 50:
                out.append(_p("Revenue concentration", top[col], "revenue_share_pct",
                              share, 50, "Medium",
                              f"One {label} ({top[col]}) brings {share:.0f}% of revenue."))

    return sorted(out, key=lambda p: SEV_ORDER[p["severity"]])


# ------------------------------------------------------------- AI insights
TEMPLATES = {
    "Revenue drop": (
        "Revenue fell against the previous month, possibly due to seasonality.",
        "Compare with the same month last year before reacting.",
        "Avoids a wrong reaction to normal seasonal change."),
    "Late delivery spike": (
        "A logistics problem pushed late deliveries far above normal.",
        "Investigate carrier capacity and handling time in this month.",
        "A lower late rate in peak periods."),
    "Late delivery by region": (
        "Deliveries to this region regularly miss the promised date.",
        "Review carrier performance there and set realistic delivery estimates.",
        "Fewer late orders and better ratings."),
    "Low-rated category": (
        "Products in this category are rated below average.",
        "Check product quality and descriptions, and review its top sellers.",
        "A higher category rating."),
    "Low customer retention": (
        "Most customers buy only once.",
        "Launch a repeat-purchase campaign, such as discounts or follow-up emails.",
        "A higher repeat-customer rate."),
    "Revenue concentration": (
        "Revenue depends heavily on one category or region.",
        "Grow the next-largest segments to reduce the risk.",
        "A more balanced revenue base."),
}


def fallback_insights(k, problems):
    insights, seen = [], set()
    for p in problems:
        if p["type"] in seen or p["type"] not in TEMPLATES:
            continue
        seen.add(p["type"])
        cause, action, impact = TEMPLATES[p["type"]]
        insights.append({"problem": f"{p['type']}: {p['entity']}",
                         "root_cause": f"{cause} Data: {p['detail']}",
                         "recommendation": action, "priority": p["severity"],
                         "expected_impact": impact})
    parts = [f"The data contains {k['total_orders']:,} orders and revenue of "
             f"{k['total_revenue']:,.0f}."]
    if "late_delivery_pct" in k:
        parts.append(f"{k['late_delivery_pct']}% of deliveries are late.")
    if "avg_rating" in k:
        parts.append(f"The average rating is {k['avg_rating']}.")
    if "repeat_customer_pct" in k:
        parts.append(f"{k['repeat_customer_pct']}% of customers buy again.")
    return {"executive_summary": " ".join(parts), "insights": insights[:7]}


def ai_insights(k, problems):
    try:
        from modules import ai_engine
        prompt = ai_engine.build_prompt(k, problems[:12], "not available")
        prompt = prompt.replace("e-commerce marketplace", "business")
        text = ai_engine.call_llm(prompt).strip()
        text = text.replace("```json", "").replace("```", "").strip()
        res = json.loads(text)
        res["source"] = "gemini"
    except Exception:
        res = fallback_insights(k, problems)
        res["source"] = "rule-based"
    return res


# -------------------------------------------------------------------- UI
def _fmt(x):
    return f"{x / 1e6:.2f}M" if abs(x) >= 1e6 else f"{x:,.0f}"


def render_results(res):
    k, full, cat, reg = res["kpis"], res["full"], res["cat"], res["reg"]

    st.subheader("Key metrics")
    cards = [("Orders", f"{k['total_orders']:,}"),
             ("Revenue", _fmt(k["total_revenue"])),
             ("Avg order value", f"{k['avg_order_value']:,.2f}")]
    if "unique_customers" in k:
        cards += [("Customers", f"{k['unique_customers']:,}"),
                  ("Repeat customers", f"{k['repeat_customer_pct']}%")]
    if "late_delivery_pct" in k:
        cards.append(("Late deliveries", f"{k['late_delivery_pct']}%"))
    if "avg_delivery_days" in k:
        cards.append(("Avg delivery time", f"{k['avg_delivery_days']} days"))
    if "avg_rating" in k:
        cards.append(("Avg rating", f"{k['avg_rating']}"))
    for i in range(0, len(cards), 4):
        for col, (label, val) in zip(st.columns(4), cards[i:i + 4]):
            col.metric(label, val)

    st.subheader("Trends")
    a, b = st.columns(2)
    a.plotly_chart(px.line(full, x="month", y="revenue", markers=True,
                           title="Monthly revenue"), use_container_width=True)
    b.plotly_chart(px.bar(full, x="month", y="orders", title="Orders per month"),
                   use_container_width=True)
    if "late_pct" in full:
        st.plotly_chart(px.line(full, x="month", y="late_pct", markers=True,
                                title="Late delivery rate (%)"),
                        use_container_width=True)
    st.caption("Months with very few orders are treated as partial and excluded.")

    if cat is not None:
        st.subheader("Categories")
        st.plotly_chart(px.bar(cat.head(10), x="revenue", y="category",
                               orientation="h", title="Top 10 categories by revenue")
                        .update_yaxes(autorange="reversed"),
                        use_container_width=True)
    if reg is not None:
        st.subheader("Regions")
        metric = "late_pct" if "late_pct" in reg else "revenue"
        st.plotly_chart(px.bar(reg.sort_values(metric, ascending=False),
                               x="region", y=metric, title=f"{metric} by region"),
                        use_container_width=True)
        st.dataframe(reg, use_container_width=True, hide_index=True)

    st.subheader("Detected problems")
    if res["problems"]:
        st.dataframe(pd.DataFrame(res["problems"]), use_container_width=True,
                     hide_index=True)
    else:
        st.success("No problems detected with the current rules.")


def render_ai(ai):
    st.info(ai["executive_summary"])
    for x in ai["insights"]:
        icon = SEV_ICON.get(x["priority"], "⚪")
        with st.expander(f"{icon} [{x['priority']}] {x['problem']}"):
            st.markdown(f"**Root cause:** {x['root_cause']}")
            st.markdown(f"**Recommendation:** {x['recommendation']}")
            st.markdown(f"**Expected impact:** {x['expected_impact']}")
    st.caption(f"Insights source: {ai.get('source', 'unknown')}")


def page_upload():
    st.subheader("Analyze your own data")
    st.write("Upload a CSV or Excel file of orders or sales. Match your columns "
             "to the fields below. Only order ID, order date and an amount are "
             "required, and each extra field unlocks more analysis.")

    f = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
    if f is None:
        st.info("Waiting for a file. Tip: run `python3 utils/make_test_file.py` "
                "to create a test file from the Olist data.")
        return

    if st.session_state.get("up_name") != f.name:
        st.session_state.update(up_name=f.name, up_result=None, up_ai=None)

    raw = read_upload(f.name, f.getvalue())
    st.write(f"**{f.name}**: {len(raw):,} rows, {raw.shape[1]} columns")
    st.dataframe(raw.head(8), use_container_width=True, hide_index=True)

    st.markdown("#### Match your columns")
    guess = guess_mapping(list(raw.columns))
    options = [NONE] + list(raw.columns)
    mapping = {}
    cols = st.columns(3)
    for i, (field, (label, _, _)) in enumerate(FIELDS.items()):
        default = options.index(guess[field]) if guess[field] in options else 0
        choice = cols[i % 3].selectbox(label, options, index=default,
                                       key=f"map_{f.name}_{field}")
        mapping[field] = None if choice == NONE else choice

    problems = []
    if not mapping["order_id"]:
        problems.append("Order ID")
    if not mapping["order_date"]:
        problems.append("Order date")
    if not mapping["amount"] and not (mapping["quantity"] and mapping["unit_price"]):
        problems.append("Order amount (or quantity and unit price)")
    if problems:
        st.error("Please map: " + ", ".join(problems))
        return

    skipped = []
    if not mapping["customer_id"]:
        skipped.append("retention (needs customer ID)")
    if not mapping["category"]:
        skipped.append("category analysis (needs category)")
    if not mapping["region"]:
        skipped.append("region analysis (needs region)")
    if not (mapping["delivered_date"] and mapping["promised_date"]):
        skipped.append("late-delivery analysis (needs delivered and promised dates)")
    if not mapping["rating"]:
        skipped.append("rating analysis (needs rating)")
    if skipped:
        st.caption("Not available with this mapping: " + "; ".join(skipped))

    if st.button("Analyze", type="primary"):
        df, note = standardize(raw, mapping)
        if df.empty:
            st.error("No usable rows after cleaning. Check the date and amount columns.")
            return
        st.session_state.up_result = {"note": note, **analyze(df)}
        st.session_state.up_ai = None

    res = st.session_state.get("up_result")
    if res:
        st.success(res["note"])
        render_results(res)

        st.subheader("AI insights and recommendations")
        if st.button("Generate AI insights"):
            with st.spinner("Asking the AI. This can take up to a minute..."):
                st.session_state.up_ai = ai_insights(res["kpis"], res["problems"])
        if st.session_state.get("up_ai"):
            render_ai(st.session_state.up_ai)