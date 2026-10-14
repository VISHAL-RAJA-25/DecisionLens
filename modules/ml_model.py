"""ml_model.py - predict which orders will be delivered late."""
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "database" / "decisionlens.db"
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "late_model.joblib"
METRICS_PATH = PROCESSED / "model_metrics.json"
IMPORTANCE_PATH = PROCESSED / "model_importance.csv"

# Only information known at purchase time (no delivery dates: that would leak)
NUM = ["price", "freight", "freight_ratio", "n_items", "n_sellers", "weight_g",
       "volume_cm3", "estimated_days", "purchase_dow", "purchase_hour",
       "purchase_month_num"]
CAT = ["customer_state", "seller_state", "category", "same_state"]


def load_data():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("""
            SELECT o.order_id,
                   o.order_purchase_timestamp      AS purchase_ts,
                   o.order_estimated_delivery_date AS estimated_ts,
                   o.is_late,
                   c.customer_state,
                   MIN(s.seller_state)                  AS seller_state,
                   MIN(p.product_category_name_english) AS category,
                   COUNT(*)                             AS n_items,
                   COUNT(DISTINCT i.seller_id)          AS n_sellers,
                   SUM(i.price)                         AS price,
                   SUM(i.freight_value)                 AS freight,
                   SUM(p.product_weight_g)              AS weight_g,
                   SUM(p.product_length_cm * p.product_height_cm
                       * p.product_width_cm)            AS volume_cm3
            FROM orders o
            JOIN customers c   ON o.customer_id = c.customer_id
            JOIN order_items i ON o.order_id = i.order_id
            JOIN sellers s     ON i.seller_id = s.seller_id
            JOIN products p    ON i.product_id = p.product_id
            WHERE o.order_status = 'delivered' AND o.is_late IS NOT NULL
            GROUP BY o.order_id
        """, conn)

    df["purchase_ts"] = pd.to_datetime(df["purchase_ts"])
    df["estimated_ts"] = pd.to_datetime(df["estimated_ts"])
    df["estimated_days"] = (
        (df["estimated_ts"] - df["purchase_ts"]).dt.total_seconds() / 86400)
    df["purchase_dow"] = df["purchase_ts"].dt.dayofweek
    df["purchase_hour"] = df["purchase_ts"].dt.hour
    df["purchase_month_num"] = df["purchase_ts"].dt.month
    df["freight_ratio"] = (df["freight"] / df["price"].replace(0, np.nan)).fillna(0)
    df["same_state"] = np.where(df["customer_state"] == df["seller_state"],
                                "yes", "no")
    df["category"] = df["category"].fillna("unknown")
    for c in ("weight_g", "volume_cm3"):
        df[c] = df[c].fillna(df[c].median())
    df["is_late"] = df["is_late"].astype(int)
    return df


def logit_pipeline():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM)])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=1000))])


def boosting_pipeline():
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT),
        ("num", "passthrough", NUM)])
    clf = HistGradientBoostingClassifier(
        categorical_features=list(range(len(CAT))),
        max_iter=200, learning_rate=0.08, random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(model, X, y):
    """Rank-based metrics: only ~8% of orders are late, so accuracy misleads."""
    y = y.to_numpy()
    p = model.predict_proba(X)[:, 1]
    top = np.argsort(-p)[: int(len(y) * 0.10)]
    precision_top = float(y[top].mean())
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 3),
        "pr_auc": round(float(average_precision_score(y, p)), 3),
        "top10_recall": round(float(y[top].sum() / y.sum()), 3),
        "top10_precision": round(precision_top, 3),
        "top10_lift": round(precision_top / float(y.mean()), 2),
        "threshold": float(np.quantile(p, 0.90)),
    }


def train():
    df = load_data()
    df = df.sort_values("purchase_ts").reset_index(drop=True)
    cut = int(len(df) * 0.8)
    X, y = df[NUM + CAT], df["is_late"]
    X_tr, y_tr = X.iloc[:cut], y.iloc[:cut]
    X_te, y_te = X.iloc[cut:], y.iloc[cut:]

    models = {"Logistic regression (baseline)": logit_pipeline(),
              "Gradient boosting": boosting_pipeline()}
    results = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        results[name] = evaluate(m, X_te, y_te)
        r = results[name]
        print(f"{name:<32} AUC {r['roc_auc']}  PR-AUC {r['pr_auc']}  "
              f"top-10% catches {r['top10_recall'] * 100:.0f}% of late orders")

    best = models["Gradient boosting"]
    sample = X_te.sample(min(5000, len(X_te)), random_state=42)
    imp = permutation_importance(best, sample, y_te.loc[sample.index],
                                 scoring="roc_auc", n_repeats=3,
                                 random_state=42, n_jobs=1)
    importance = (pd.DataFrame({"feature": NUM + CAT,
                                "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    metrics = {"rows": int(len(df)), "test_rows": int(len(y_te)),
               "late_rate_pct": round(float(y.mean() * 100), 2),
               "models": results}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    joblib.dump({
        "model": best, "num": NUM, "cat": CAT,
        "threshold": results["Gradient boosting"]["threshold"],
        "base_rate": float(y.mean()),
        "defaults": {c: float(df[c].median()) for c in NUM},
        "choices": {"customer_state": sorted(df["customer_state"].unique()),
                    "seller_state": sorted(df["seller_state"].unique()),
                    "category": sorted(df["category"].unique())},
    }, MODEL_PATH)

    print("\nTop drivers of late delivery:")
    print(importance.head(8).to_string(index=False))
    print(f"\nSaved model to {MODEL_PATH}")


# ------------------------------------------------------------------ dashboard
def page_model():
    import plotly.express as px
    import streamlit as st

    st.subheader("Late-delivery prediction")
    if not METRICS_PATH.exists() or not MODEL_PATH.exists():
        st.warning("Model not trained yet. Run `python3 modules/ml_model.py`.")
        return

    m = json.loads(METRICS_PATH.read_text())
    best = m["models"]["Gradient boosting"]
    base = m["models"]["Logistic regression (baseline)"]
    st.write(f"Trained on {m['rows']:,} delivered orders ({m['late_rate_pct']}% "
             f"late), tested on {m['test_rows']:,} unseen orders. Uses only "
             "information known at purchase time.")

    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", best["roc_auc"], f"{best['roc_auc'] - base['roc_auc']:+.3f} vs baseline")
    c2.metric("Late orders caught in top 10% risk", f"{best['top10_recall'] * 100:.0f}%")
    c3.metric("Lift in top 10%", f"{best['top10_lift']}x")
    st.caption("If you flag the 10% riskiest orders, the model finds "
               f"{best['top10_recall'] * 100:.0f}% of all late orders. Random "
               "flagging would find 10%.")

    imp = pd.read_csv(IMPORTANCE_PATH).head(10)
    st.plotly_chart(
        px.bar(imp, x="importance", y="feature", orientation="h",
               title="What drives late delivery (drop in AUC when shuffled)")
        .update_yaxes(autorange="reversed"), use_container_width=True)

    st.markdown("#### Try it: score an order")
    bundle = joblib.load(MODEL_PATH)
    ch, dflt = bundle["choices"], bundle["defaults"]

    def pick(options, value):
        return options.index(value) if value in options else 0

    with st.form("predict"):
        a, b, c = st.columns(3)
        cust = a.selectbox("Customer state", ch["customer_state"],
                           index=pick(ch["customer_state"], "SP"))
        sell = b.selectbox("Seller state", ch["seller_state"],
                           index=pick(ch["seller_state"], "SP"))
        cat = c.selectbox("Category", ch["category"],
                          index=pick(ch["category"], "health_beauty"))
        d, e, f = st.columns(3)
        price = d.number_input("Price", 1.0, 10000.0, 100.0)
        freight = e.number_input("Freight", 0.0, 1000.0, 20.0)
        est = f.number_input("Promised delivery window (days)", 1.0, 90.0,
                             float(round(dflt["estimated_days"])))
        weight = st.number_input("Weight (g)", 1.0, 50000.0,
                                 float(round(dflt["weight_g"])))
        go = st.form_submit_button("Score this order")

    if go:
        row = dict(dflt)
        row.update(price=price, freight=freight, estimated_days=est,
                   weight_g=weight, customer_state=cust, seller_state=sell,
                   category=cat, freight_ratio=freight / price,
                   same_state="yes" if cust == sell else "no")
        X = pd.DataFrame([row])[bundle["num"] + bundle["cat"]]
        p = float(bundle["model"].predict_proba(X)[0, 1])
        level = ("High" if p >= bundle["threshold"]
                 else "Medium" if p >= bundle["base_rate"] else "Low")
        st.metric("Probability of arriving late", f"{p * 100:.1f}%",
                  f"{level} risk (average order: {bundle['base_rate'] * 100:.1f}%)",
                  delta_color="off")


if __name__ == "__main__":
    train()