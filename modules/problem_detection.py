"""problem_detection.py - rule-based detection of business problems from KPIs."""
import json
import pandas as pd
from pathlib import Path

try:
    from modules import analytics as an
except ImportError:  # when run directly as a script
    import analytics as an

PROCESSED = an.PROCESSED
SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def _problem(ptype, entity, metric, value, benchmark, severity, detail):
    return {
        "type": ptype,
        "entity": str(entity),
        "metric": metric,
        "value": round(float(value), 2),
        "benchmark": round(float(benchmark), 2),
        "severity": severity,
        "detail": detail,
    }


def detect_state_delivery(kpis, states, min_orders=200):
    """States where late deliveries are far above the national rate."""
    overall = float(kpis["late_delivery_pct"])
    out = []
    for _, r in states[states["orders"] >= min_orders].iterrows():
        if r["late_pct"] >= overall * 1.5:
            sev = "High" if r["late_pct"] >= overall * 2 else "Medium"
            out.append(_problem(
                "Late delivery by state", r["state"], "late_pct",
                r["late_pct"], overall, sev,
                f"{r['late_pct']}% of orders late vs {overall}% nationally; "
                f"avg review {r['avg_review']} across {int(r['orders'])} orders."))
    return out


def detect_state_freight(kpis, states, min_orders=200):
    """States where freight cost is high relative to product value."""
    overall = float(kpis["freight_share_pct"])
    out = []
    for _, r in states[states["orders"] >= min_orders].iterrows():
        if r["freight_pct"] >= overall * 1.4:
            sev = "High" if r["freight_pct"] >= overall * 1.8 else "Medium"
            out.append(_problem(
                "High freight cost", r["state"], "freight_pct",
                r["freight_pct"], overall, sev,
                f"Freight is {r['freight_pct']}% of product revenue vs "
                f"{overall}% overall ({int(r['orders'])} orders)."))
    return out


def detect_month_anomalies(trend, reviews):
    """Months where late rate or bad-review share spikes, or revenue drops."""
    out = []

    mean, std = trend["late_pct"].mean(), trend["late_pct"].std()
    for _, r in trend[trend["late_pct"] > mean + 1.5 * std].iterrows():
        out.append(_problem(
            "Late delivery spike", r["month"], "late_pct",
            r["late_pct"], mean, "High",
            f"Late rate hit {r['late_pct']}% vs a monthly average of {mean:.1f}%."))

    mean, std = reviews["bad_review_pct"].mean(), reviews["bad_review_pct"].std()
    for _, r in reviews[reviews["bad_review_pct"] > mean + 1.5 * std].iterrows():
        out.append(_problem(
            "Review score drop", r["month"], "bad_review_pct",
            r["bad_review_pct"], mean, "Medium",
            f"{r['bad_review_pct']}% of reviews were 1-2 stars vs {mean:.1f}% average."))

    for _, r in trend[trend["revenue_growth_pct"] <= -10].iterrows():
        out.append(_problem(
            "Revenue drop", r["month"], "revenue_growth_pct",
            r["revenue_growth_pct"], 0, "Low",
            f"Revenue fell {abs(r['revenue_growth_pct'])}% vs the previous month "
            f"(check for seasonality, e.g. after Black Friday)."))
    return out


def detect_sellers(sellers):
    """Sellers with poor reviews or frequent late deliveries."""
    out = []
    for _, r in sellers.iterrows():
        if r["avg_review"] < 3.5 or r["late_pct"] > 15:
            sev = "High" if (r["avg_review"] < 3.0 or r["late_pct"] > 25) else "Medium"
            out.append(_problem(
                "Underperforming seller", r["seller_id"][:8], "avg_review",
                r["avg_review"], 4.0, sev,
                f"Seller {r['seller_id'][:8]} ({r['state']}): review {r['avg_review']}, "
                f"{r['late_pct']}% late, {int(r['orders'])} orders, "
                f"revenue {r['revenue']:,.0f}."))
    return out


def detect_categories(kpis, categories):
    """Categories whose reviews are well below the overall average."""
    overall = float(kpis["avg_review_score"])
    out = []
    for _, r in categories.iterrows():
        if r["avg_review"] <= overall - 0.35:
            sev = "High" if r["avg_review"] <= overall - 0.5 else "Medium"
            out.append(_problem(
                "Low-rated category", r["category"], "avg_review",
                r["avg_review"], overall, sev,
                f"Avg review {r['avg_review']} vs {overall} overall; "
                f"{int(r['orders'])} orders, {r['late_pct']}% late."))
    return out


def detect_retention(kpis):
    rate = float(kpis["repeat_customer_pct"])
    if rate < 5:
        return [_problem(
            "Low customer retention", "All customers", "repeat_customer_pct",
            rate, 5, "High",
            f"Only {rate}% of {kpis['unique_customers']:,} customers ordered more than once.")]
    return []


def run_detection():
    kpis = an.kpi_summary()
    problems = (
        detect_state_delivery(kpis, an.state_performance())
        + detect_state_freight(kpis, an.state_performance())
        + detect_month_anomalies(an.monthly_trend(), an.review_by_month())
        + detect_sellers(an.seller_performance())
        + detect_categories(kpis, an.category_performance())
        + detect_retention(kpis)
    )
    df = pd.DataFrame(problems)
    df["rank"] = df["severity"].map(SEVERITY_ORDER)
    df = df.sort_values(["rank", "type"]).drop(columns="rank").reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "problems.csv", index=False)
    with open(PROCESSED / "problems.json", "w") as f:
        json.dump(df.to_dict("records"), f, indent=2)
    return df


if __name__ == "__main__":
    df = run_detection()
    print(f"{len(df)} problems detected\n")
    print(df.groupby(["type", "severity"]).size().to_string())
    print("\nTop 10 problems:")
    for _, r in df.head(10).iterrows():
        print(f"[{r['severity']}] {r['type']} - {r['entity']}: {r['detail']}")