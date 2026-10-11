"""ai_engine.py - root causes and recommendations from detected problems.

Tries Gemini (free tier) across several models. If every attempt fails,
falls back to rule-based insights so the project always works.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
load_dotenv(ROOT / ".env")

PREFERRED_MODEL = "gemini-3.8-flash"
MAX_PER_TYPE = 3
SKIP_WORDS = ("image", "tts", "live", "audio", "embedding", "robotics")


def select_problems():
    df = pd.read_csv(PROCESSED / "problems.csv")
    sev = {"High": 0, "Medium": 1, "Low": 2}
    df["rank"] = df["severity"].map(sev)
    df = df.sort_values(["type", "rank"]).groupby("type").head(MAX_PER_TYPE)
    return df.sort_values("rank").drop(columns="rank").to_dict("records")


def build_prompt(kpis, problems, late_vs_review):
    return f"""You are a senior business analyst for an e-commerce marketplace.

HEADLINE KPIs:
{json.dumps(kpis, indent=2)}

LATE vs ON-TIME REVIEW SCORES:
{late_vs_review}

DETECTED PROBLEMS (computed from the data):
{json.dumps(problems, indent=2)}

Use ONLY the numbers above. Do not invent figures.
Return ONLY valid JSON, with no markdown, in this format:
{{
  "executive_summary": "3 sentences for a non-technical manager",
  "insights": [
    {{
      "problem": "short title",
      "root_cause": "most likely cause, supported by a number from the data",
      "recommendation": "one specific action",
      "priority": "High | Medium | Low",
      "expected_impact": "what should improve"
    }}
  ]
}}
Give 5 to 7 insights, ordered by priority."""


# ---------- Gemini ----------
def candidate_models(client):
    """Preferred model first, then every other text 'flash' model the key can use."""
    names = []
    for m in client.models.list():
        n = m.name.replace("models/", "")
        if "flash" in n and "generateContent" in (m.supported_actions or []):
            if not any(w in n for w in SKIP_WORDS):
                names.append(n)
    names = sorted(set(names), key=lambda n: (n != PREFERRED_MODEL, n))
    return names or [PREFERRED_MODEL]


def call_llm(prompt):
    from google import genai
    from google.genai import types

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(response_mime_type="application/json")
    last_error = None

    for model in candidate_models(client):
        for attempt in range(2):
            try:
                print(f"Trying {model} (attempt {attempt + 1})...")
                resp = client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                print(f"Success with {model}")
                return resp.text
            except Exception as e:
                last_error = e
                msg = str(e)
                if "404" in msg or "NOT_FOUND" in msg:
                    break  # model unavailable: go to the next model
                print(f"  failed: {msg[:90]}")
                time.sleep(15)
    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


# ---------- Rule-based fallback ----------
TEMPLATES = {
    "Late delivery by state": (
        "Deliveries to this state regularly miss the promised date.",
        "Review carrier performance for this state and set more realistic delivery estimates.",
        "Fewer late orders and higher review scores."),
    "Late delivery spike": (
        "A one-off logistics problem pushed late deliveries far above normal.",
        "Investigate carrier capacity and seller handling time in this month.",
        "A lower late rate in peak periods."),
    "High freight cost": (
        "Shipping costs are high compared with product value in this state.",
        "Offer regional fulfilment or free-shipping thresholds.",
        "Higher conversion and lower cart abandonment."),
    "Review score drop": (
        "A larger share of customers gave 1 to 2 star reviews in this month.",
        "Read the low-score comments from this month and fix the common issue.",
        "Better average review score."),
    "Revenue drop": (
        "Revenue fell against the previous month, possibly due to seasonality.",
        "Compare with the same month last year before reacting.",
        "Avoids a wrong reaction to normal seasonal change."),
    "Underperforming seller": (
        "This seller has low reviews or frequent late deliveries.",
        "Warn the seller, set targets, and limit listings if there is no improvement.",
        "Fewer bad customer experiences."),
    "Low-rated category": (
        "Products in this category receive lower reviews than average.",
        "Check product quality and descriptions, and review the top sellers in it.",
        "A higher category review score."),
    "Low customer retention": (
        "Almost all customers buy only once.",
        "Launch a repeat-purchase campaign, such as discounts or email follow-ups.",
        "A higher repeat-customer rate."),
}


def fallback_insights(kpis, problems):
    insights = []
    seen = set()
    for p in problems:
        if p["type"] in seen or p["type"] not in TEMPLATES:
            continue
        seen.add(p["type"])
        cause, action, impact = TEMPLATES[p["type"]]
        insights.append({
            "problem": f"{p['type']}: {p['entity']}",
            "root_cause": f"{cause} Data: {p['detail']}",
            "recommendation": action,
            "priority": p["severity"],
            "expected_impact": impact,
        })
    summary = (
        f"The business has {kpis['total_orders']:,} delivered orders and "
        f"R$ {kpis['total_revenue']:,.0f} in revenue. "
        f"{kpis['late_delivery_pct']}% of deliveries are late, and the average review is "
        f"{kpis['avg_review_score']}. Only {kpis['repeat_customer_pct']}% of customers "
        f"buy again."
    )
    return {"executive_summary": summary, "insights": insights[:7]}


def generate_insights():
    kpis = json.loads((PROCESSED / "kpi_summary.json").read_text())
    lvr = pd.read_csv(PROCESSED / "late_vs_review.csv").to_string(index=False)
    problems = select_problems()

    try:
        text = call_llm(build_prompt(kpis, problems, lvr)).strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        result["source"] = "gemini"
    except Exception as e:
        print(f"\nAI unavailable ({str(e)[:120]}).\nUsing rule-based fallback.\n")
        result = fallback_insights(kpis, problems)
        result["source"] = "rule-based"

    (PROCESSED / "ai_insights.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    result = generate_insights()
    print(f"SOURCE: {result['source']}\n")
    print("EXECUTIVE SUMMARY\n" + result["executive_summary"] + "\n")
    for i, x in enumerate(result["insights"], 1):
        print(f"{i}. [{x['priority']}] {x['problem']}")
        print(f"   Cause:  {x['root_cause']}")
        print(f"   Action: {x['recommendation']}")
        print(f"   Impact: {x['expected_impact']}\n")