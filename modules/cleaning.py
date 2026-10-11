"""cleaning.py - load the 9 raw Olist CSVs, clean them, save to data/cleaned."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEANED = ROOT / "data" / "cleaned"


def load(filename):
    return pd.read_csv(RAW / filename)


def clean_orders():
    df = load("olist_orders_dataset.csv")

    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    day = 86400  # seconds in a day
    df["delivery_days"] = (
        (df["order_delivered_customer_date"] - df["order_purchase_timestamp"])
        .dt.total_seconds() / day
    ).round(2)
    # positive = delivered after the promised date
    df["delay_days"] = (
        (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"])
        .dt.total_seconds() / day
    ).round(2)
    # 1 = late, 0 = on time, empty = not delivered
    df["is_late"] = (df["delay_days"] > 0).astype("Int64").where(df["delay_days"].notna())
    df["purchase_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    return df


def clean_order_items():
    df = load("olist_order_items_dataset.csv").drop_duplicates()
    df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"], errors="coerce")
    df["item_total"] = df["price"] + df["freight_value"]
    return df


def clean_payments():
    return load("olist_order_payments_dataset.csv").drop_duplicates()


def clean_reviews():
    df = load("olist_order_reviews_dataset.csv")
    df["review_creation_date"] = pd.to_datetime(df["review_creation_date"], errors="coerce")
    df["review_answer_timestamp"] = pd.to_datetime(df["review_answer_timestamp"], errors="coerce")

    df["has_comment"] = df["review_comment_message"].notna().astype(int)
    df["review_comment_title"] = df["review_comment_title"].fillna("")
    df["review_comment_message"] = df["review_comment_message"].fillna("")

    # some orders have more than one review: keep the latest one
    df = df.sort_values("review_answer_timestamp").drop_duplicates("order_id", keep="last")
    return df


def clean_customers():
    df = load("olist_customers_dataset.csv").drop_duplicates()
    df["customer_city"] = df["customer_city"].str.strip().str.lower()
    df["customer_state"] = df["customer_state"].str.strip().str.upper()
    return df


def clean_sellers():
    df = load("olist_sellers_dataset.csv").drop_duplicates()
    df["seller_city"] = df["seller_city"].str.strip().str.lower()
    df["seller_state"] = df["seller_state"].str.strip().str.upper()
    return df


def clean_geolocation():
    df = load("olist_geolocation_dataset.csv").drop_duplicates()
    # many points per zip prefix -> one row per zip prefix
    df = (
        df.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(
            geolocation_lat=("geolocation_lat", "mean"),
            geolocation_lng=("geolocation_lng", "mean"),
            geolocation_city=("geolocation_city", "first"),
            geolocation_state=("geolocation_state", "first"),
        )
    )
    return df


def clean_products():
    df = load("olist_products_dataset.csv")
    translation = load("product_category_name_translation.csv")

    # fix the typo in the original column names
    df = df.rename(columns={
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    })

    df["product_category_name"] = df["product_category_name"].fillna("unknown")

    df = df.merge(translation, on="product_category_name", how="left")
    # categories missing from the translation table keep their original name
    df["product_category_name_english"] = df["product_category_name_english"].fillna(
        df["product_category_name"]
    )

    # fill missing numeric values with the column median
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


def run_cleaning():
    CLEANED.mkdir(parents=True, exist_ok=True)

    tables = {
        "orders": clean_orders,
        "order_items": clean_order_items,
        "payments": clean_payments,
        "reviews": clean_reviews,
        "customers": clean_customers,
        "sellers": clean_sellers,
        "geolocation": clean_geolocation,
        "products": clean_products,
    }

    for name, func in tables.items():
        df = func()
        df.to_csv(CLEANED / f"{name}_clean.csv", index=False)
        print(f"{name:<12} -> {len(df):>9,} rows, {df.shape[1]} columns saved")

    print("\nDone. Cleaned files are in data/cleaned/")


if __name__ == "__main__":
    run_cleaning()