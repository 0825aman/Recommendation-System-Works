"""KuaiRec 2.0 Exploratory Data Analysis.

Analyses all 5 Phase 2 Parquet files using Polars + DuckDB and produces
structured JSON and Markdown reports.

Usage (from project root):
    backend\\.venv\\Scripts\\python.exe scripts/ml/run_eda.py

Output:
    data/processed/kuairec/reports/eda_report.json
    data/processed/kuairec/reports/eda_report.md
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb

from scripts.data.common import PARQUET_FILES, REPORTS_DIR, get_logger

logger = get_logger(__name__)

EDA_REPORT_JSON = REPORTS_DIR / "eda_report.json"
EDA_REPORT_MD = REPORTS_DIR / "eda_report.md"


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------


def _duck(query: str, con: duckdb.DuckDBPyConnection) -> list[dict]:  # type: ignore[type-arg]
    """Execute a DuckDB query and return rows as list of dicts (no numpy needed)."""
    cursor = con.execute(query)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    return [dict(zip(columns, row)) for row in rows]


# ---------------------------------------------------------------------------
# Individual analysis sections
# ---------------------------------------------------------------------------


def analyze_small_matrix(con: duckdb.DuckDBPyConnection) -> dict:  # type: ignore[type-arg]
    """Comprehensive analysis of the interaction matrix."""
    logger.info("  Analyzing small_matrix ...")
    path = str(PARQUET_FILES["small_matrix"])

    # Basic counts
    basic = _duck(
        f"""
        SELECT
            COUNT(*)                                       AS total_rows,
            COUNT(DISTINCT user_id)                        AS unique_users,
            COUNT(DISTINCT video_id)                       AS unique_videos,
            ROUND(COUNT(*) * 100.0 /
                (COUNT(DISTINCT user_id) * COUNT(DISTINCT video_id)), 4) AS density_pct,
            SUM(CASE WHEN time IS NULL THEN 1 ELSE 0 END)  AS null_time_count,
            ROUND(SUM(CASE WHEN time IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4)
                                                           AS null_time_pct
        FROM read_parquet('{path}')
        """,
        con,
    )[0]

    # watch_ratio statistics
    wr_stats = _duck(
        f"""
        SELECT
            ROUND(MIN(watch_ratio), 4)    AS wr_min,
            ROUND(MAX(watch_ratio), 4)    AS wr_max,
            ROUND(AVG(watch_ratio), 4)    AS wr_mean,
            ROUND(MEDIAN(watch_ratio), 4) AS wr_median,
            ROUND(STDDEV(watch_ratio), 4) AS wr_std,
            SUM(CASE WHEN watch_ratio = 0.0        THEN 1 ELSE 0 END) AS wr_zero,
            SUM(CASE WHEN watch_ratio < 0.5        THEN 1 ELSE 0 END) AS wr_below_half,
            SUM(CASE WHEN watch_ratio >= 0.5
                      AND watch_ratio < 1.0        THEN 1 ELSE 0 END) AS wr_half_to_one,
            SUM(CASE WHEN watch_ratio = 1.0        THEN 1 ELSE 0 END) AS wr_exact_one,
            SUM(CASE WHEN watch_ratio > 1.0        THEN 1 ELSE 0 END) AS wr_above_one,
            SUM(CASE WHEN watch_ratio > 5.0        THEN 1 ELSE 0 END) AS wr_above_five,
            SUM(CASE WHEN watch_ratio > 10.0       THEN 1 ELSE 0 END) AS wr_above_ten,
            SUM(CASE WHEN watch_ratio >= 0.5       THEN 1 ELSE 0 END) AS wr_relevant
        FROM read_parquet('{path}')
        """,
        con,
    )[0]

    # watch_ratio percentiles
    pct = _duck(
        f"""
        SELECT
            ROUND(QUANTILE_CONT(watch_ratio, 0.10), 4) AS p10,
            ROUND(QUANTILE_CONT(watch_ratio, 0.25), 4) AS p25,
            ROUND(QUANTILE_CONT(watch_ratio, 0.50), 4) AS p50,
            ROUND(QUANTILE_CONT(watch_ratio, 0.75), 4) AS p75,
            ROUND(QUANTILE_CONT(watch_ratio, 0.90), 4) AS p90,
            ROUND(QUANTILE_CONT(watch_ratio, 0.95), 4) AS p95,
            ROUND(QUANTILE_CONT(watch_ratio, 0.99), 4) AS p99
        FROM read_parquet('{path}')
        """,
        con,
    )[0]

    # Interactions per user distribution
    per_user = _duck(
        f"""
        SELECT
            MIN(cnt) AS min_interactions_per_user,
            MAX(cnt) AS max_interactions_per_user,
            ROUND(AVG(cnt), 2) AS mean_interactions_per_user,
            ROUND(MEDIAN(cnt), 2) AS median_interactions_per_user
        FROM (
            SELECT user_id, COUNT(*) AS cnt
            FROM read_parquet('{path}')
            GROUP BY user_id
        )
        """,
        con,
    )[0]

    # Interactions per item distribution
    per_item = _duck(
        f"""
        SELECT
            MIN(cnt) AS min_interactions_per_item,
            MAX(cnt) AS max_interactions_per_item,
            ROUND(AVG(cnt), 2) AS mean_interactions_per_item,
            ROUND(MEDIAN(cnt), 2) AS median_interactions_per_item
        FROM (
            SELECT video_id, COUNT(*) AS cnt
            FROM read_parquet('{path}')
            GROUP BY video_id
        )
        """,
        con,
    )[0]

    # Top 10 most interacted items
    top_items = _duck(
        f"""
        SELECT video_id, COUNT(*) AS interaction_count,
               ROUND(AVG(watch_ratio), 4) AS mean_watch_ratio
        FROM read_parquet('{path}')
        GROUP BY video_id
        ORDER BY interaction_count DESC
        LIMIT 10
        """,
        con,
    )

    # Top 10 most active users
    top_users = _duck(
        f"""
        SELECT user_id, COUNT(*) AS interaction_count,
               ROUND(AVG(watch_ratio), 4) AS mean_watch_ratio
        FROM read_parquet('{path}')
        GROUP BY user_id
        ORDER BY interaction_count DESC
        LIMIT 10
        """,
        con,
    )

    # Temporal distribution by date (using date column, not null in 96%+ rows)
    temporal = _duck(
        f"""
        SELECT
            MIN(date) AS earliest_date,
            MAX(date) AS latest_date,
            COUNT(DISTINCT date) AS unique_dates,
            SUM(CASE WHEN date IS NULL THEN 1 ELSE 0 END) AS null_date_count
        FROM read_parquet('{path}')
        """,
        con,
    )[0]

    return {
        "basic": basic,
        "watch_ratio_stats": wr_stats,
        "watch_ratio_percentiles": pct,
        "per_user_stats": per_user,
        "per_item_stats": per_item,
        "top_10_items_by_interactions": top_items,
        "top_10_users_by_interactions": top_users,
        "temporal": temporal,
    }


def analyze_user_features(con: duckdb.DuckDBPyConnection) -> dict:  # type: ignore[type-arg]
    """Analysis of user demographic and engagement features."""
    logger.info("  Analyzing user_features ...")
    path = str(PARQUET_FILES["user_features"])

    basic = _duck(
        f"SELECT COUNT(*) AS total_users, COUNT(DISTINCT user_id) AS unique_user_ids "
        f"FROM read_parquet('{path}')",
        con,
    )[0]

    activity = _duck(
        f"""
        SELECT user_active_degree, COUNT(*) AS count,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
        FROM read_parquet('{path}')
        GROUP BY user_active_degree
        ORDER BY count DESC
        """,
        con,
    )

    register_days = _duck(
        f"""
        SELECT
            MIN(register_days) AS min_days,
            MAX(register_days) AS max_days,
            ROUND(AVG(register_days), 1) AS mean_days,
            ROUND(MEDIAN(register_days), 1) AS median_days
        FROM read_parquet('{path}')
        """,
        con,
    )[0]

    flags = _duck(
        f"""
        SELECT
            SUM(CASE WHEN is_lowactive_period THEN 1 ELSE 0 END) AS low_active_count,
            SUM(CASE WHEN is_live_streamer    THEN 1 ELSE 0 END) AS live_streamer_count,
            SUM(CASE WHEN is_video_author     THEN 1 ELSE 0 END) AS video_author_count
        FROM read_parquet('{path}')
        """,
        con,
    )[0]

    return {
        "basic": basic,
        "activity_distribution": activity,
        "register_days_stats": register_days,
        "boolean_flags": flags,
    }


def analyze_item_categories(con: duckdb.DuckDBPyConnection) -> dict:  # type: ignore[type-arg]
    """Analysis of item category tags."""
    logger.info("  Analyzing item_categories ...")
    path = str(PARQUET_FILES["item_categories"])

    basic = _duck(
        f"""
        SELECT COUNT(*) AS total_rows, COUNT(DISTINCT video_id) AS unique_videos
        FROM read_parquet('{path}')
        """,
        con,
    )[0]

    return {"basic": basic}


def analyze_caption_category(con: duckdb.DuckDBPyConnection) -> dict:  # type: ignore[type-arg]
    """Analysis of video captions and category hierarchy."""
    logger.info("  Analyzing kuairec_caption_category ...")
    path = str(PARQUET_FILES["kuairec_caption_category"])

    basic = _duck(
        f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT video_id) AS unique_videos,
            SUM(CASE WHEN caption IS NULL THEN 1 ELSE 0 END) AS null_captions,
            ROUND(SUM(CASE WHEN caption IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2)
                AS null_caption_pct
        FROM read_parquet('{path}')
        """,
        con,
    )[0]

    top_level_cats = _duck(
        f"""
        SELECT first_level_category_name AS category,
               COUNT(*) AS video_count,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct
        FROM read_parquet('{path}')
        GROUP BY first_level_category_name
        ORDER BY video_count DESC
        LIMIT 20
        """,
        con,
    )

    second_level_count = _duck(
        f"""
        SELECT COUNT(DISTINCT second_level_category_name) AS unique_second_level_cats
        FROM read_parquet('{path}')
        """,
        con,
    )[0]

    return {
        "basic": basic,
        "top_first_level_categories": top_level_cats,
        "second_level_category_count": second_level_count,
    }


def analyze_cross_file_coverage(con: duckdb.DuckDBPyConnection) -> dict:  # type: ignore[type-arg]
    """Cross-file user and item coverage analysis."""
    logger.info("  Analyzing cross-file coverage ...")
    sm_path = str(PARQUET_FILES["small_matrix"])
    uf_path = str(PARQUET_FILES["user_features"])
    ic_path = str(PARQUET_FILES["item_categories"])
    cc_path = str(PARQUET_FILES["kuairec_caption_category"])

    user_coverage = _duck(
        f"""
        SELECT
            COUNT(DISTINCT sm.user_id) AS users_in_matrix,
            COUNT(DISTINCT uf.user_id) AS users_in_features,
            COUNT(DISTINCT CASE WHEN uf.user_id IS NOT NULL THEN sm.user_id END)
                AS users_in_both
        FROM read_parquet('{sm_path}') sm
        LEFT JOIN read_parquet('{uf_path}') uf ON sm.user_id = uf.user_id
        """,
        con,
    )[0]

    item_coverage = _duck(
        f"""
        SELECT
            COUNT(DISTINCT sm.video_id) AS items_in_matrix,
            COUNT(DISTINCT ic.video_id) AS items_in_categories,
            COUNT(DISTINCT cc.video_id) AS items_in_captions,
            COUNT(DISTINCT CASE WHEN ic.video_id IS NOT NULL THEN sm.video_id END)
                AS items_in_matrix_and_categories
        FROM read_parquet('{sm_path}') sm
        LEFT JOIN read_parquet('{ic_path}') ic ON sm.video_id = ic.video_id
        LEFT JOIN read_parquet('{cc_path}') cc ON sm.video_id = cc.video_id
        """,
        con,
    )[0]

    # Items in interaction matrix but NOT in category files
    missing_items = _duck(
        f"""
        SELECT COUNT(DISTINCT sm.video_id) AS items_missing_category_info
        FROM read_parquet('{sm_path}') sm
        LEFT JOIN read_parquet('{ic_path}') ic ON sm.video_id = ic.video_id
        WHERE ic.video_id IS NULL
        """,
        con,
    )[0]

    return {
        "user_coverage": user_coverage,
        "item_coverage": item_coverage,
        "items_missing_category_info": missing_items,
    }


def analyze_popularity_distribution(con: duckdb.DuckDBPyConnection) -> dict:  # type: ignore[type-arg]
    """Item popularity distribution — Pareto analysis."""
    logger.info("  Analyzing popularity distribution ...")
    sm_path = str(PARQUET_FILES["small_matrix"])

    # What fraction of items account for 50%, 80%, 90% of interactions?
    pareto = _duck(
        f"""
        WITH item_counts AS (
            SELECT video_id, COUNT(*) AS cnt,
                   COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() AS frac
            FROM read_parquet('{sm_path}')
            GROUP BY video_id
        ),
        ranked AS (
            SELECT video_id, cnt, frac,
                   SUM(frac) OVER (ORDER BY cnt DESC) AS cumulative_frac,
                   ROW_NUMBER() OVER (ORDER BY cnt DESC) AS rank
            FROM item_counts
        )
        SELECT
            COUNT(DISTINCT video_id) AS total_items,
            SUM(CASE WHEN cumulative_frac <= 0.50 THEN 1 ELSE 0 END) AS items_top_50pct,
            SUM(CASE WHEN cumulative_frac <= 0.80 THEN 1 ELSE 0 END) AS items_top_80pct,
            SUM(CASE WHEN cumulative_frac <= 0.90 THEN 1 ELSE 0 END) AS items_top_90pct
        FROM ranked
        """,
        con,
    )[0]

    # Popularity metric preview: mean(watch_ratio) * log(1 + count) — top 15
    weighted_pop = _duck(
        f"""
        SELECT video_id,
               COUNT(*) AS interaction_count,
               ROUND(AVG(watch_ratio), 4) AS mean_watch_ratio,
               ROUND(AVG(watch_ratio) * LN(1 + COUNT(*)), 4) AS weighted_popularity
        FROM read_parquet('{sm_path}')
        GROUP BY video_id
        ORDER BY weighted_popularity DESC
        LIMIT 15
        """,
        con,
    )

    return {
        "pareto": pareto,
        "top_15_by_weighted_popularity": weighted_pop,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def build_markdown_report(report: dict) -> str:  # type: ignore[type-arg]
    """Convert the JSON report dict to a readable Markdown document."""
    sm = report["small_matrix"]
    basic = sm["basic"]
    wr = sm["watch_ratio_stats"]
    pct = sm["watch_ratio_percentiles"]
    pu = sm["per_user_stats"]
    pi = sm["per_item_stats"]
    t = sm["temporal"]
    uf = report["user_features"]
    cc = report["caption_category"]
    cov = report["cross_file_coverage"]
    pop = report["popularity_distribution"]
    pareto = pop["pareto"]

    total = basic["total_rows"]
    relevant = wr["wr_relevant"]

    lines = [
        "# KuaiRec 2.0 EDA Report",
        "",
        f"**Generated:** {report['generated_at']}",
        "",
        "---",
        "",
        "## 1. Interaction Matrix (small_matrix)",
        "",
        "### Basic Statistics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total interactions | {total:,} |",
        f"| Unique users | {basic['unique_users']:,} |",
        f"| Unique videos | {basic['unique_videos']:,} |",
        f"| Matrix density | {basic['density_pct']}% |",
        f"| Null timestamps | {basic['null_time_count']:,} ({basic['null_time_pct']}%) |",
        "",
        "### watch_ratio Distribution",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Min | {wr['wr_min']} |",
        f"| Max | {wr['wr_max']} |",
        f"| Mean | {wr['wr_mean']} |",
        f"| Median | {wr['wr_median']} |",
        f"| Std dev | {wr['wr_std']} |",
        f"| Zero ratio (wr = 0.0) | {wr['wr_zero']:,} ({round(wr['wr_zero']*100/total,2)}%) |",
        f"| Below threshold (wr < 0.5) | {wr['wr_below_half']:,} ({round(wr['wr_below_half']*100/total,2)}%) |",
        f"| Partial watch (0.5 ≤ wr < 1.0) | {wr['wr_half_to_one']:,} ({round(wr['wr_half_to_one']*100/total,2)}%) |",
        f"| Full watch (wr = 1.0) | {wr['wr_exact_one']:,} ({round(wr['wr_exact_one']*100/total,2)}%) |",
        f"| Replays (wr > 1.0) | {wr['wr_above_one']:,} ({round(wr['wr_above_one']*100/total,2)}%) |",
        f"| Extreme replays (wr > 5.0) | {wr['wr_above_five']:,} ({round(wr['wr_above_five']*100/total,2)}%) |",
        f"| **Relevant (wr ≥ 0.5)** | **{relevant:,} ({round(relevant*100/total,2)}%)** |",
        "",
        "### watch_ratio Percentiles",
        "",
        "| Percentile | Value |",
        "|---|---|",
        f"| P10 | {pct['p10']} |",
        f"| P25 | {pct['p25']} |",
        f"| P50 (median) | {pct['p50']} |",
        f"| P75 | {pct['p75']} |",
        f"| P90 | {pct['p90']} |",
        f"| P95 | {pct['p95']} |",
        f"| P99 | {pct['p99']} |",
        "",
        "### Interactions Per User",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Min | {pu['min_interactions_per_user']:,} |",
        f"| Max | {pu['max_interactions_per_user']:,} |",
        f"| Mean | {pu['mean_interactions_per_user']:,} |",
        f"| Median | {pu['median_interactions_per_user']:,} |",
        "",
        "### Interactions Per Item",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Min | {pi['min_interactions_per_item']:,} |",
        f"| Max | {pi['max_interactions_per_item']:,} |",
        f"| Mean | {pi['mean_interactions_per_item']:,} |",
        f"| Median | {pi['median_interactions_per_item']:,} |",
        "",
        "### Temporal Coverage",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Earliest date | {t['earliest_date']} |",
        f"| Latest date | {t['latest_date']} |",
        f"| Unique dates | {t['unique_dates']:,} |",
        f"| Null date count | {t['null_date_count']:,} |",
        "",
        "---",
        "",
        "## 2. User Features",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total users | {uf['basic']['total_users']:,} |",
        f"| Unique user IDs | {uf['basic']['unique_user_ids']:,} |",
        f"| Low-active period | {uf['boolean_flags']['low_active_count']:,} |",
        f"| Live streamers | {uf['boolean_flags']['live_streamer_count']:,} |",
        f"| Video authors | {uf['boolean_flags']['video_author_count']:,} |",
        f"| Mean register days | {uf['register_days_stats']['mean_days']} |",
        "",
        "### User Activity Distribution",
        "",
        "| Activity Level | Count | % |",
        "|---|---|---|",
    ]

    for row in uf["activity_distribution"]:
        lines.append(f"| {row['user_active_degree']} | {row['count']:,} | {row['pct']}% |")

    lines += [
        "",
        "---",
        "",
        "## 3. Item Categories and Captions",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Videos in category file | {cc['basic']['unique_videos']:,} |",
        f"| Null captions | {cc['basic']['null_captions']:,} ({cc['basic']['null_caption_pct']}%) |",
        f"| Unique second-level categories | {cc['second_level_category_count']['unique_second_level_cats']:,} |",
        "",
        "### Top First-Level Categories",
        "",
        "| Category | Videos | % |",
        "|---|---|---|",
    ]

    for row in cc["top_first_level_categories"]:
        lines.append(f"| {row['category']} | {row['video_count']:,} | {row['pct']}% |")

    cov_data = cov
    lines += [
        "",
        "---",
        "",
        "## 4. Cross-File Coverage",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Users in interaction matrix | {cov_data['user_coverage']['users_in_matrix']:,} |",
        f"| Users in features file | {cov_data['user_coverage']['users_in_features']:,} |",
        f"| Users in both | {cov_data['user_coverage']['users_in_both']:,} |",
        f"| Items in interaction matrix | {cov_data['item_coverage']['items_in_matrix']:,} |",
        f"| Items in category file | {cov_data['item_coverage']['items_in_categories']:,} |",
        f"| Items in both | {cov_data['item_coverage']['items_in_matrix_and_categories']:,} |",
        f"| Items missing category info | {cov_data['items_missing_category_info']['items_missing_category_info']:,} |",
        "",
        "---",
        "",
        "## 5. Popularity Distribution",
        "",
        "| Top N items covering X% of interactions | Value |",
        "|---|---|",
        f"| Total items | {pareto['total_items']:,} |",
        f"| Items covering 50% of interactions | {pareto['items_top_50pct']} |",
        f"| Items covering 80% of interactions | {pareto['items_top_80pct']} |",
        f"| Items covering 90% of interactions | {pareto['items_top_90pct']} |",
        "",
        "### Top 15 Items by Weighted Popularity (mean_wr × log(1+count))",
        "",
        "| Rank | video_id | Interactions | Mean watch_ratio | Weighted Score |",
        "|---|---|---|---|---|",
    ]

    for i, row in enumerate(pop["top_15_by_weighted_popularity"], 1):
        lines.append(
            f"| {i} | {row['video_id']} | {row['interaction_count']:,} | "
            f"{row['mean_watch_ratio']} | {row['weighted_popularity']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Phase 3 Target Label Summary",
        "",
        "| Decision | Value |",
        "|---|---|",
        "| Relevance signal | `watch_ratio` (continuous) |",
        "| Binary threshold | `watch_ratio ≥ 0.5` |",
        f"| Relevant interactions | {relevant:,} ({round(relevant*100/total,2)}%) |",
        f"| Non-relevant interactions | {total-relevant:,} ({round((total-relevant)*100/total,2)}%) |",
        "| Zero-ratio rows | Classified as non-relevant; kept in training data |",
        "| Null timestamps | Assigned to training split (3.89% of rows) |",
        "",
        "---",
        "",
        "*Generated by `scripts/ml/run_eda.py`*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_eda() -> dict:  # type: ignore[type-arg]
    """Run full EDA and produce JSON + Markdown reports.

    Returns:
        EDA report dictionary.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("KuaiRec 2.0 EDA")
    logger.info("=" * 60)

    # Verify Parquet files exist
    for path in PARQUET_FILES.values():
        if not path.exists():
            logger.error("Parquet file not found: %s", path)
            raise FileNotFoundError(
                f"Parquet file missing: {path}. "
                "Run scripts/data/convert_to_parquet.py first."
            )

    con = duckdb.connect()

    logger.info("Running EDA sections ...")

    report: dict = {  # type: ignore[type-arg]
        "title": "KuaiRec 2.0 EDA Report",
        "generated_at": datetime.now(UTC).isoformat(),
        "small_matrix": analyze_small_matrix(con),
        "user_features": analyze_user_features(con),
        "item_categories": analyze_item_categories(con),
        "caption_category": analyze_caption_category(con),
        "cross_file_coverage": analyze_cross_file_coverage(con),
        "popularity_distribution": analyze_popularity_distribution(con),
        "runtime_seconds": 0.0,
    }

    report["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    con.close()

    # Write JSON
    with EDA_REPORT_JSON.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("JSON report: %s", EDA_REPORT_JSON)

    # Write Markdown
    md = build_markdown_report(report)
    with EDA_REPORT_MD.open("w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Markdown report: %s", EDA_REPORT_MD)

    logger.info("EDA complete in %.2fs", report["runtime_seconds"])
    return report


if __name__ == "__main__":
    run_eda()
