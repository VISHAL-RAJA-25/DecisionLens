-- 1. Monthly revenue and orders (delivered only)
SELECT o.purchase_month,
       COUNT(DISTINCT o.order_id) AS orders,
       ROUND(SUM(i.price), 2)     AS revenue,
       ROUND(SUM(i.price) / COUNT(DISTINCT o.order_id), 2) AS avg_order_value
FROM orders o
JOIN order_items i ON o.order_id = i.order_id
WHERE o.order_status = 'delivered'
GROUP BY o.purchase_month
ORDER BY o.purchase_month;

-- 2. Month-over-month revenue growth (CTE + window function)
WITH monthly AS (
    SELECT o.purchase_month, SUM(i.price) AS revenue
    FROM orders o
    JOIN order_items i ON o.order_id = i.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY o.purchase_month
)
SELECT purchase_month,
       ROUND(revenue, 2) AS revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY purchase_month))
             / LAG(revenue) OVER (ORDER BY purchase_month), 1) AS growth_pct
FROM monthly;

-- 3. Late delivery rate and average review score by customer state
SELECT c.customer_state,
       COUNT(*) AS orders,
       ROUND(100.0 * AVG(o.is_late), 1) AS late_pct,
       ROUND(AVG(r.review_score), 2)    AS avg_review
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN reviews r   ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
GROUP BY c.customer_state
ORDER BY late_pct DESC;

-- 4. Review score for late vs on-time orders (the root-cause query)
SELECT CASE WHEN o.is_late = 1 THEN 'Late' ELSE 'On time' END AS delivery,
       COUNT(*) AS orders,
       ROUND(AVG(r.review_score), 2) AS avg_review
FROM orders o
JOIN reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'
GROUP BY delivery;

-- 5. Top 10 sellers by revenue, with rank
SELECT seller_id,
       ROUND(SUM(price), 2) AS revenue,
       RANK() OVER (ORDER BY SUM(price) DESC) AS revenue_rank
FROM order_items
GROUP BY seller_id
ORDER BY revenue_rank
LIMIT 10;