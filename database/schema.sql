-- ============================================================================
-- Tourism Experience Analytics — SQL schema (Checkpoint 04)
-- ============================================================================
-- This mirrors the ACTUAL project entities documented in
-- docs/data_dictionary.md (raw tables User/Item/Country/Region/Continent/
-- City/Type/Mode/Transaction), rebuilt as a normalized SQLite schema from
-- the already-cleaned tables in data/cleaned/*.csv.
--
-- No raw or cleaned CSV is modified by this schema — it is loaded read-only
-- by scripts/build_database.py into a fresh database/tourism.db file.
--
-- One documented deviation from a naive 1:1 copy of item.csv, explained in
-- full in docs/sql_report.md ("Schema decisions"):
--   dim_attraction stores AttractionCityName / AttractionCountry directly
--   (sourced from data/processed/consolidated.csv, the already-locked,
--   already-validated output of the project's cleaning pipeline) instead of
--   joining item.AttractionCityId -> dim_city -> dim_country. That join path
--   is documented as broken for exactly these 30 attractions in
--   docs/data_dictionary.md, Data-Quality Finding 7 (their AttractionCityId
--   values collide with unrelated City.xlsx rows in Cameroon/Chad). Using
--   dim_city for attractions here would silently reintroduce a data-quality
--   bug this project already fixed upstream. AttractionCityId is still
--   stored for reference/traceability, just not used as a live FK to
--   dim_city.
-- ============================================================================

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS fact_transaction;
DROP TABLE IF EXISTS dim_attraction;
DROP TABLE IF EXISTS dim_attraction_type;
DROP TABLE IF EXISTS dim_mode;
DROP TABLE IF EXISTS dim_user;
DROP TABLE IF EXISTS dim_city;
DROP TABLE IF EXISTS dim_country;
DROP TABLE IF EXISTS dim_region;
DROP TABLE IF EXISTS dim_continent;

-- ---------------------------------------------------------------------------
-- Traveler geography dimensions (Continent -> Region -> Country -> City)
-- Source: data/cleaned/continent.csv, region.csv, country.csv, city.csv
-- These joins are NOT affected by Data-Quality Finding 7 (that issue is
-- specific to attraction city codes only) and are used as-is.
-- ---------------------------------------------------------------------------

CREATE TABLE dim_continent (
    ContinentId   INTEGER PRIMARY KEY,
    ContinentName TEXT NOT NULL
);

CREATE TABLE dim_region (
    RegionId    INTEGER PRIMARY KEY,
    RegionName  TEXT NOT NULL,
    ContinentId INTEGER,
    FOREIGN KEY (ContinentId) REFERENCES dim_continent (ContinentId)
);

CREATE TABLE dim_country (
    CountryId          INTEGER PRIMARY KEY,
    CountryName        TEXT NOT NULL,
    RegionId           INTEGER,
    flag_duplicate_name INTEGER,  -- carried over from data/cleaned/country.csv (0/1)
    FOREIGN KEY (RegionId) REFERENCES dim_region (RegionId)
);

CREATE TABLE dim_city (
    CityId      INTEGER PRIMARY KEY,
    CityName    TEXT,
    CountryId   INTEGER,
    FOREIGN KEY (CountryId) REFERENCES dim_country (CountryId)
);

-- ---------------------------------------------------------------------------
-- Traveler dimension. Source: data/cleaned/user.csv
-- ---------------------------------------------------------------------------

CREATE TABLE dim_user (
    UserId      INTEGER PRIMARY KEY,
    ContinentId INTEGER,
    RegionId    INTEGER,
    CountryId   INTEGER,
    CityId      INTEGER,
    FOREIGN KEY (ContinentId) REFERENCES dim_continent (ContinentId),
    FOREIGN KEY (RegionId)    REFERENCES dim_region (RegionId),
    FOREIGN KEY (CountryId)   REFERENCES dim_country (CountryId),
    FOREIGN KEY (CityId)      REFERENCES dim_city (CityId)
);

-- ---------------------------------------------------------------------------
-- Visit mode lookup. Source: distinct (VisitMode, VisitModeName) pairs in
-- data/processed/consolidated.csv (guaranteed consistent with the fact
-- table's VisitMode codes; equivalent content to data/cleaned/mode.csv).
-- ---------------------------------------------------------------------------

CREATE TABLE dim_mode (
    VisitModeId   INTEGER PRIMARY KEY,
    VisitModeName TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Attraction type lookup. Source: data/cleaned/type.csv
-- ---------------------------------------------------------------------------

CREATE TABLE dim_attraction_type (
    AttractionTypeId   INTEGER PRIMARY KEY,
    AttractionTypeName TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Attraction dimension. Source: data/cleaned/item.csv joined with the
-- corrected AttractionCityName/AttractionCountry from
-- data/processed/consolidated.csv (see header note above).
-- ---------------------------------------------------------------------------

CREATE TABLE dim_attraction (
    AttractionId       INTEGER PRIMARY KEY,
    Attraction         TEXT NOT NULL,
    AttractionAddress  TEXT,
    AttractionTypeId   INTEGER,
    AttractionCityId   INTEGER,  -- kept for traceability only; NOT a reliable
                                  -- FK to dim_city for this table (see header)
    AttractionCityName TEXT,     -- corrected value, from consolidated.csv
    AttractionCountry  TEXT,     -- corrected value, from consolidated.csv
    FOREIGN KEY (AttractionTypeId) REFERENCES dim_attraction_type (AttractionTypeId)
);

-- ---------------------------------------------------------------------------
-- Fact table: one row per transaction (rating event).
-- Source: data/cleaned/transaction.csv, loaded unchanged (49,208 rows).
-- ---------------------------------------------------------------------------

CREATE TABLE fact_transaction (
    TransactionId       INTEGER PRIMARY KEY,
    UserId              INTEGER NOT NULL,
    AttractionId        INTEGER NOT NULL,
    VisitYear           INTEGER NOT NULL,
    VisitMonth          INTEGER NOT NULL,
    VisitMode           INTEGER NOT NULL,
    Rating              INTEGER NOT NULL,
    is_ambiguous_repeat INTEGER NOT NULL,  -- SQLite has no native bool; 0/1
    FOREIGN KEY (UserId)       REFERENCES dim_user (UserId),
    FOREIGN KEY (AttractionId) REFERENCES dim_attraction (AttractionId),
    FOREIGN KEY (VisitMode)    REFERENCES dim_mode (VisitModeId)
);

CREATE INDEX idx_fact_user ON fact_transaction (UserId);
CREATE INDEX idx_fact_attraction ON fact_transaction (AttractionId);
CREATE INDEX idx_fact_mode ON fact_transaction (VisitMode);
CREATE INDEX idx_fact_year_month ON fact_transaction (VisitYear, VisitMonth);
