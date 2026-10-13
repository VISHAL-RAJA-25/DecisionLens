"""Flatten the Olist database into one table to test the upload page."""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
conn = sqlite3.connect(ROOT / "database" / "decisionlens.db")
df = pd.read_sql_query("""
    SELECT o.order_id,
           o.order_purchase_timestamp      AS order_date,
           i.price                         AS amount,
           c.customer_unique_id            AS customer_id,
           p.product_category_name_english AS category,
           c.customer_state                AS region,
           o.order_delivered_customer_date AS delivered_date,
           o.order_estimated_delivery_date AS promised_date,
           r.review_score                  AS rating
    FROM orders o
    JOIN order_items i ON o.order_id = i.order_id
    JOIN customers c   ON o.customer_id = c.customer_id
    JOIN products p    ON i.product_id = p.product_id
    LEFT JOIN reviews r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
""", conn)
out = ROOT / "data" / "test_olist_flat.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df):,} rows to {out}")