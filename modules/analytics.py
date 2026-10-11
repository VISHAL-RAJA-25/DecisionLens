"""analytics.py - KPI functions that read from the SQLite database."""
import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "database" / "decisionlens.db"
PROCESSED = ROOT / "data" / "processed"

# full months only (the dataset has partial months at both ends)
START, END = "2017-01", "2018-08"


def query(sql, params=()):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def kpi_summary():
    """Headline KPIs for the dashboard cards."""
    sales = query("""
        SELECT COUNT(DISTINCT o.order_id) AS orders,
               SUM(i.price)               AS revenue,
               SUM(i.freight_value)       AS freight
        FROM orders o JOIN order_items i ON o.order_id = i.order_id
        WHERE o.order_status = 'delivered'
    """).iloc[0]

    ops = query("""
        SELECT AVG(o.is_late) * 100 AS late_pct,
               AVG(o.delivery_days) AS avg_delivery_days,
               AVG(r.review_score)  AS avg_review
        FROM orders o LEFT JOIN reviews r ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered'
    """).iloc[0]

    repeat = query("""
        WITH c AS (
            SELECT cu.customer_unique_id, COUNT(DISTINCT o.order_id) AS n
            FROM orders o JOIN customers cu ON o.customer_id = cu.customer_id
            WHERE o.order_status = 'delivered'
            GROUP BY cu.customer_unique_id)
        SELECT 100.0 * SUM(n > 1) / COUNT(*) AS repeat_pct, COUNT(*) AS customers
        FROM c
    """).iloc[0]

    return {
        "total_orders": int(sales["orders"]),
        "total_revenue": round(sales["revenue"], 2),
        "avg_order_value": round(sales["revenue"] / sales["orders"], 2),
        "freight_share_pct": round(100 * sales["freight"] / sales["revenue"], 1),
        "unique_customers": int(repeat["customers"]),
        "repeat_customer_pct": round(repeat["repeat_pct"], 2),
        "late_delivery_pct": round(ops["late_pct"], 2),
        "avg_delivery_days": round(ops["avg_delivery_days"], 1),
        "avg_review_score": round(ops["avg_review"], 2),
    }


def monthly_trend():
    df = query("""
        SELECT o.purchase_month AS month,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(SUM(i.price), 2)     AS revenue,
               ROUND(AVG(o.is_late) * 100, 1) AS late_pct,
               ROUND(AVG(o.delivery_days), 1) AS avg_delivery_days
        FROM orders o JOIN order_items i ON o.order_id = i.order_id
        WHERE o.order_status = 'delivered' AND o.purchase_month BETWEEN ? AND ?
        GROUP BY o.purchase_month ORDER BY o.purchase_month
    """, (START, END))
    df["aov"] = (df["revenue"] / df["orders"]).round(2)
    df["revenue_growth_pct"] = (df["revenue"].pct_change() * 100).round(1)
    return df


def review_by_month():
    return query("""
        SELECT o.purchase_month AS month,
               ROUND(AVG(r.review_score), 2) AS avg_review,
               ROUND(100.0 * AVG(r.review_score <= 2), 1) AS bad_review_pct
        FROM orders o JOIN reviews r ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered' AND o.purchase_month BETWEEN ? AND ?
        GROUP BY o.purchase_month ORDER BY o.purchase_month
    """, (START, END))


def state_performance():
    return query("""
        SELECT c.customer_state AS state,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(SUM(i.price), 2) AS revenue,
               ROUND(100.0 * SUM(i.freight_value) / SUM(i.price), 1) AS freight_pct,
               ROUND(AVG(o.is_late) * 100, 1) AS late_pct,
               ROUND(AVG(o.delivery_days), 1) AS avg_delivery_days,
               ROUND(AVG(r.review_score), 2) AS avg_review
        FROM orders o
        JOIN customers c   ON o.customer_id = c.customer_id
        JOIN order_items i ON o.order_id = i.order_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered'
        GROUP BY c.customer_state ORDER BY revenue DESC
    """)


def category_performance(min_orders=100):
    return query("""
        SELECT p.product_category_name_english AS category,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(SUM(i.price), 2) AS revenue,
               ROUND(AVG(r.review_score), 2) AS avg_review,
               ROUND(AVG(o.is_late) * 100, 1) AS late_pct
        FROM orders o
        JOIN order_items i ON o.order_id = i.order_id
        JOIN products p    ON i.product_id = p.product_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered'
        GROUP BY category HAVING orders >= ?
        ORDER BY revenue DESC
    """, (min_orders,))


def seller_performance(min_orders=30):
    return query("""
        SELECT i.seller_id,
               s.seller_state AS state,
               COUNT(DISTINCT o.order_id) AS orders,
               ROUND(SUM(i.price), 2) AS revenue,
               ROUND(AVG(r.review_score), 2) AS avg_review,
               ROUND(AVG(o.is_late) * 100, 1) AS late_pct
        FROM orders o
        JOIN order_items i ON o.order_id = i.order_id
        JOIN sellers s     ON i.seller_id = s.seller_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered'
        GROUP BY i.seller_id HAVING orders >= ?
        ORDER BY revenue DESC
    """, (min_orders,))


def late_vs_review():
    return query("""
        SELECT CASE WHEN o.is_late = 1 THEN 'Late' ELSE 'On time' END AS delivery,
               COUNT(*) AS orders,
               ROUND(AVG(r.review_score), 2) AS avg_review
        FROM orders o JOIN reviews r ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered'
        GROUP BY delivery
    """)


def order_status_mix():
    return query("""
        SELECT order_status, COUNT(*) AS orders,
               ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM orders), 2) AS pct
        FROM orders GROUP BY order_status ORDER BY orders DESC
    """)


def save_all():
    """Save every KPI table to data/processed for the dashboard and AI engine."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tables = {
        "monthly_trend": monthly_trend(),
        "review_by_month": review_by_month(),
        "state_performance": state_performance(),
        "category_performance": category_performance(),
        "seller_performance": seller_performance(),
        "late_vs_review": late_vs_review(),
        "order_status_mix": order_status_mix(),
    }
    for name, df in tables.items():
        df.to_csv(PROCESSED / f"{name}.csv", index=False)
        print(f"{name:<22} -> {len(df):>4} rows saved")

    pd.Series(kpi_summary()).to_json(PROCESSED / "kpi_summary.json", indent=2)
    print("kpi_summary            -> saved")


if __name__ == "__main__":
    print(kpi_summary())
    print()
    save_all()