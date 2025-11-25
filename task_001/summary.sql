DROP TABLE IF EXISTS "books"; 

SELECT * FROM "books";

DROP TABLE IF EXISTS "summary_by_year";

CREATE TABLE "summary_by_year" AS
SELECT
  COALESCE(year, 0) AS publication_year,
  COUNT(*) AS book_count,
  ROUND(
    AVG(
      CASE
        WHEN upper(currency) = 'EUR' THEN (price * 1.2)
        WHEN upper(currency) = 'USD' THEN price
        ELSE NULL
      END
    )::numeric
  , 2) AS average_price_usd
FROM books
GROUP BY COALESCE(year, 0)
ORDER BY publication_year;

SELECT * FROM "summary_by_year";