"""build_db.py - load cleaned CSVs into a SQLite database."""
import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEANED = ROOT / "data" / "cleaned"
DB_PATH = ROOT / "database" / "decisionlens.db"

TABLES = [
    "orders", "order_items", "payments", "reviews",
    "customers", "sellers", "geolocation", "products",
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_orders_id ON orders(order_id)",
    "CREATE INDEX IF NOT EXISTS idx_orders_cust ON orders(customer_id)",
    "CREATE INDEX IF NOT EXISTS idx_items_order ON order_items(order_id)",
    "CREATE INDEX IF NOT EXISTS idx_items_product ON order_items(product_id)",
    "CREATE INDEX IF NOT EXISTS idx_items_seller ON order_items(seller_id)",
    "CREATE INDEX IF NOT EXISTS idx_pay_order ON payments(order_id)",
    "CREATE INDEX IF NOT EXISTS idx_rev_order ON reviews(order_id)",
    "CREATE INDEX IF NOT EXISTS idx_cust_id ON customers(customer_id)",
    "CREATE INDEX IF NOT EXISTS idx_prod_id ON products(product_id)",
    "CREATE INDEX IF NOT EXISTS idx_sell_id ON sellers(seller_id)",
]


def build():
    if DB_PATH.exists():
        DB_PATH.unlink()  # rebuild from scratch

    conn = sqlite3.connect(DB_PATH)
    for t in TABLES:
        df = pd.read_csv(CLEANED / f"{t}_clean.csv")
        df.to_sql(t, conn, index=False)
        print(f"{t:<12} -> {len(df):>9,} rows loaded")

    for stmt in INDEXES:
        conn.execute(stmt)
    conn.commit()
    conn.close()
    print(f"\nDatabase created: {DB_PATH}")


if __name__ == "__main__":
    build()