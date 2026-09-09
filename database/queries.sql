-- ============================================================================
-- Tourism Experience Analytics — SQL analysis queries (Checkpoint 04)
-- ============================================================================
-- Runs against database/tourism.db, built by scripts/build_database.py from
-- database/schema.sql. Every query below is standalone, ANSI-SQL-compatible
-- SQLite, and uses only columns that exist in the schema documented in
-- database/schema.sql / docs/sql_report.md — no invented columns or values.
--
-- Format: each query is preceded by a small header parsed by
-- scripts/run_sql_analysis.py:
--   -- @id: <short_id>            (used as the result filename)
--   -- @title: <human title>
--   -- @purpose: <business question this answers>
-- ============================================================================


-- @id: 01_total_visits
-- @title: Total visits
-- @purpose: How many recorded visit/rating transactions exist in the dataset?
SELECT COUNT(*) AS total_visits
FROM fact_transaction;


-- @id: 02_unique_travelers
-- @title: Unique travelers
-- @purpose: How many distinct travelers generated these transactions?
SELECT COUNT(*) AS unique_travelers
FROM dim_user;


-- @id: 03_unique_attractions
-- @title: Unique attractions visited
-- @purpose: How many distinct attractions appear in the transaction log?
SELECT COUNT(DISTINCT AttractionId) AS unique_attractions_visited
FROM fact_transaction;


-- @id: 04_visits_by_country
-- @title: Visits by traveler home country
-- @purpose: Which home countries generate the most visits/transactions?
--           Direct input to origin-market marketing prioritization.
SELECT
    c.CountryName                          AS country,
    COUNT(*)                               AS total_visits,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fact_transaction), 2) AS pct_of_total
FROM fact_transaction f
JOIN dim_user u    ON u.UserId = f.UserId
JOIN dim_country c ON c.CountryId = u.CountryId
GROUP BY c.CountryName
ORDER BY total_visits DESC
LIMIT 15;


-- @id: 05_visits_by_region
-- @title: Visits by traveler home region
-- @purpose: Coarser-grained view of #4 for regional demand planning.
SELECT
    r.RegionName                           AS region,
    COUNT(*)                               AS total_visits,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fact_transaction), 2) AS pct_of_total
FROM fact_transaction f
JOIN dim_user u   ON u.UserId = f.UserId
JOIN dim_region r ON r.RegionId = u.RegionId
GROUP BY r.RegionName
ORDER BY total_visits DESC;


-- @id: 06_visits_by_continent
-- @title: Visits by traveler home continent
-- @purpose: Highest-level view of traveler origin demand.
SELECT
    co.ContinentName                       AS continent,
    COUNT(*)                               AS total_visits,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fact_transaction), 2) AS pct_of_total
FROM fact_transaction f
JOIN dim_user u       ON u.UserId = f.UserId
JOIN dim_continent co ON co.ContinentId = u.ContinentId
GROUP BY co.ContinentName
ORDER BY total_visits DESC;


-- @id: 07_visits_by_attraction
-- @title: Visits by attraction (full ranking)
-- @purpose: Complete demand ranking across all 30 attractions.
SELECT
    a.Attraction                           AS attraction,
    at.AttractionTypeName                  AS attraction_type,
    a.AttractionCityName                   AS city,
    COUNT(*)                               AS total_visits
FROM fact_transaction f
JOIN dim_attraction a       ON a.AttractionId = f.AttractionId
JOIN dim_attraction_type at ON at.AttractionTypeId = a.AttractionTypeId
GROUP BY a.Attraction, at.AttractionTypeName, a.AttractionCityName
ORDER BY total_visits DESC;


-- @id: 08_top_10_attractions
-- @title: Top 10 most popular attractions
-- @purpose: Which attractions should get the most capacity/staffing/on-site
--           investment priority?
SELECT
    a.Attraction AS attraction,
    COUNT(*)     AS total_visits
FROM fact_transaction f
JOIN dim_attraction a ON a.AttractionId = f.AttractionId
GROUP BY a.Attraction
ORDER BY total_visits DESC
LIMIT 10;


-- @id: 09_least_visited_attractions
-- @title: 10 least visited attractions
-- @purpose: Identify under-utilized attractions — candidates for either
--           promotion or de-prioritization.
SELECT
    a.Attraction AS attraction,
    COUNT(*)     AS total_visits
FROM fact_transaction f
JOIN dim_attraction a ON a.AttractionId = f.AttractionId
GROUP BY a.Attraction
ORDER BY total_visits ASC
LIMIT 10;


-- @id: 10_overall_average_rating
-- @title: Overall average rating
-- @purpose: Single headline satisfaction metric across all transactions.
SELECT
    ROUND(AVG(Rating), 3) AS avg_rating,
    COUNT(*)              AS n_ratings
FROM fact_transaction;


-- @id: 11_rating_distribution
-- @title: Rating distribution (1-5 stars)
-- @purpose: How skewed is satisfaction? Needed context before trusting any
--           single average-rating figure.
SELECT
    Rating,
    COUNT(*)                                                           AS n,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fact_transaction), 2) AS pct
FROM fact_transaction
GROUP BY Rating
ORDER BY Rating;


-- @id: 12_avg_rating_by_attraction
-- @title: Average rating by attraction (min. 50 ratings)
-- @purpose: Which specific attractions most/least satisfy visitors, filtered
--           to a minimum sample size so single-digit-count attractions don't
--           distort the ranking.
SELECT
    a.Attraction   AS attraction,
    COUNT(*)       AS n_ratings,
    ROUND(AVG(f.Rating), 3) AS avg_rating
FROM fact_transaction f
JOIN dim_attraction a ON a.AttractionId = f.AttractionId
GROUP BY a.Attraction
HAVING COUNT(*) >= 50
ORDER BY avg_rating DESC;


-- @id: 13_avg_rating_by_attraction_type
-- @title: Average rating by attraction type
-- @purpose: Which categories of experience (beaches, temples, water parks,
--           ...) satisfy travelers most, independent of any single site?
SELECT
    at.AttractionTypeName          AS attraction_type,
    COUNT(*)                       AS n_ratings,
    ROUND(AVG(f.Rating), 3)        AS avg_rating,
    ROUND(100.0 * SUM(CASE WHEN f.Rating <= 2 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_low_ratings
FROM fact_transaction f
JOIN dim_attraction a       ON a.AttractionId = f.AttractionId
JOIN dim_attraction_type at ON at.AttractionTypeId = a.AttractionTypeId
GROUP BY at.AttractionTypeName
ORDER BY avg_rating DESC;


-- @id: 14_visit_mode_distribution
-- @title: Visit Mode distribution
-- @purpose: What share of trips are Couples/Family/Friends/Solo/Business?
--           Core segmentation for product and marketing.
SELECT
    m.VisitModeName AS visit_mode,
    COUNT(*)        AS total_visits,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fact_transaction), 2) AS pct_of_total
FROM fact_transaction f
JOIN dim_mode m ON m.VisitModeId = f.VisitMode
GROUP BY m.VisitModeName
ORDER BY total_visits DESC;


-- @id: 15_visit_mode_by_continent
-- @title: Visit Mode by traveler home continent
-- @purpose: Does trip type vary by where travelers come from? Useful for
--           continent-targeted product/marketing messaging.
SELECT
    co.ContinentName AS continent,
    m.VisitModeName  AS visit_mode,
    COUNT(*)         AS total_visits,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY co.ContinentName), 2) AS pct_within_continent
FROM fact_transaction f
JOIN dim_user u       ON u.UserId = f.UserId
JOIN dim_continent co ON co.ContinentId = u.ContinentId
JOIN dim_mode m        ON m.VisitModeId = f.VisitMode
GROUP BY co.ContinentName, m.VisitModeName
ORDER BY co.ContinentName, total_visits DESC;


-- @id: 16_visit_mode_by_attraction_type
-- @title: Visit Mode by attraction type
-- @purpose: Do certain attraction types skew toward specific trip types
--           (e.g. Family at Water Parks vs. Couples at Temples)? Informs
--           attraction-level marketing and on-site experience design.
SELECT
    at.AttractionTypeName AS attraction_type,
    m.VisitModeName       AS visit_mode,
    COUNT(*)              AS total_visits,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY at.AttractionTypeName), 2) AS pct_within_type
FROM fact_transaction f
JOIN dim_attraction a       ON a.AttractionId = f.AttractionId
JOIN dim_attraction_type at ON at.AttractionTypeId = a.AttractionTypeId
JOIN dim_mode m              ON m.VisitModeId = f.VisitMode
GROUP BY at.AttractionTypeName, m.VisitModeName
ORDER BY at.AttractionTypeName, total_visits DESC;


-- @id: 17_yearly_trend
-- @title: Yearly visit volume and average rating
-- @purpose: How has demand and satisfaction evolved 2013-2022? Surfaces the
--           COVID-19 disruption directly from the data.
SELECT
    VisitYear,
    COUNT(*)                AS total_visits,
    ROUND(AVG(Rating), 3)   AS avg_rating
FROM fact_transaction
GROUP BY VisitYear
ORDER BY VisitYear;


-- @id: 18_monthly_seasonality
-- @title: Monthly seasonality (all years combined)
-- @purpose: Which calendar months see the most/least demand, independent of
--           year-over-year growth? Informs staffing/pricing calendars.
SELECT
    VisitMonth,
    COUNT(*)              AS total_visits,
    ROUND(AVG(Rating), 3) AS avg_rating
FROM fact_transaction
GROUP BY VisitMonth
ORDER BY VisitMonth;


-- @id: 19_yearly_visit_mode_mix
-- @title: Visit Mode share by year
-- @purpose: Has the traveler segment mix shifted over time (e.g. Business
--           travel share collapsing during 2020-2021)?
SELECT
    f.VisitYear,
    m.VisitModeName AS visit_mode,
    COUNT(*)        AS total_visits,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY f.VisitYear), 2) AS pct_within_year
FROM fact_transaction f
JOIN dim_mode m ON m.VisitModeId = f.VisitMode
GROUP BY f.VisitYear, m.VisitModeName
ORDER BY f.VisitYear, total_visits DESC;


-- @id: 20_repeat_vs_onetime_travelers
-- @title: Repeat vs. one-time travelers
-- @purpose: What share of the traveler base is repeat vs. one-time? Directly
--           relevant to how much a recommender can rely on user history.
SELECT
    CASE WHEN visit_count = 1 THEN 'One-time' ELSE 'Repeat' END AS traveler_type,
    COUNT(*)                                                     AS n_travelers,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM dim_user), 2) AS pct_of_travelers
FROM (
    SELECT UserId, COUNT(*) AS visit_count
    FROM fact_transaction
    GROUP BY UserId
) t
GROUP BY traveler_type;


-- @id: 21_popularity_vs_quality
-- @title: Popularity vs. quality per attraction
-- @purpose: Are the most-visited attractions also the best-rated ones, or
--           are popularity and satisfaction independent? Direct input to
--           whether a recommender should rank by raw popularity or by
--           rating.
SELECT
    a.Attraction            AS attraction,
    COUNT(*)                AS total_visits,
    ROUND(AVG(f.Rating), 3) AS avg_rating
FROM fact_transaction f
JOIN dim_attraction a ON a.AttractionId = f.AttractionId
GROUP BY a.Attraction
ORDER BY total_visits DESC;


-- @id: 22_top_attraction_per_type
-- @title: Top attraction within each attraction type
-- @purpose: For each category (Beaches, Temples, ...), which single
--           attraction leads on visit volume? Useful for category-level
--           "flagship" marketing.
SELECT attraction_type, attraction, total_visits
FROM (
    SELECT
        at.AttractionTypeName AS attraction_type,
        a.Attraction          AS attraction,
        COUNT(*)              AS total_visits,
        RANK() OVER (PARTITION BY at.AttractionTypeName ORDER BY COUNT(*) DESC) AS rnk
    FROM fact_transaction f
    JOIN dim_attraction a       ON a.AttractionId = f.AttractionId
    JOIN dim_attraction_type at ON at.AttractionTypeId = a.AttractionTypeId
    GROUP BY at.AttractionTypeName, a.Attraction
)
WHERE rnk = 1
ORDER BY total_visits DESC;


-- @id: 23_business_travelers_top_attractions
-- @title: Top attractions among Business travelers
-- @purpose: Business is the smallest but highest-rating Visit Mode segment
--           (see query 14) — where do these travelers actually go, so a
--           dedicated Business offering can be targeted correctly?
SELECT
    a.Attraction AS attraction,
    COUNT(*)     AS business_visits,
    ROUND(AVG(f.Rating), 3) AS avg_rating
FROM fact_transaction f
JOIN dim_attraction a ON a.AttractionId = f.AttractionId
JOIN dim_mode m        ON m.VisitModeId = f.VisitMode
WHERE m.VisitModeName = 'Business'
GROUP BY a.Attraction
ORDER BY business_visits DESC
LIMIT 10;


-- @id: 24_ambiguous_repeat_summary
-- @title: Ambiguous repeat-visit rows
-- @purpose: How many transactions share the same (user, attraction, year,
--           month) but differ in rating/mode — a data-quality caveat noted
--           in the data dictionary, quantified here via SQL for the SQL
--           deliverable's own documentation completeness.
SELECT
    is_ambiguous_repeat,
    COUNT(*) AS n_transactions,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fact_transaction), 2) AS pct
FROM fact_transaction
GROUP BY is_ambiguous_repeat;
