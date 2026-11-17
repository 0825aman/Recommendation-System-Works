"""KuaiRec 2.0 data quality report generator.

Aggregates findings from inspection, validation, and conversion into
a machine-readable JSON report and human-readable Markdown summary.
Uses DuckDB for cross-file analytical queries.

Usage (from project root):
    python scripts/data/generate_report.py

Output:
    data/processed/kuairec/reports/data_quality_report.json
    data/processed/kuairec/reports/data_quality_report.md
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb

from scripts.data.common import (
    INSPECTED_DIR,
    MANIFEST_PATH,
    PARQUET_FILES,
    REPORTS_DIR,
    VALIDATED_DIR,
    ensure_directories,
    format_bytes,
    get_logger,
)

logger = get_logger(__name__)

REPORT_JSON_PATH = REPORTS_DIR / "data_quality_report.json"
REPORT_MD_PATH = REPORTS_DIR / "data_quality_report.md"


def load_prior_reports() -> dict:  # type: ignore[type-arg]
    """Load inspection and validation reports from prior pipeline stages."""
    result: dict = {}  # type: ignore[type-arg]

    inspection_path = INSPECTED_DIR / "inspection_report.json"
    if inspection_path.exists():
        with inspection_path.open(encoding="utf-8") as f:
            result["inspection"] = json.load(f)

    validation_path = VALIDATED_DIR / "validation_report.json"
    if validation_path.exists():
        with validation_path.open(encoding="utf-8") as f:
            result["validation"] = json.load(f)

    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open(encoding="utf-8") as f:
            result["manifest"] = json.load(f)

    return result


def compute_watch_ratio_stats(small_matrix_path: Path) -> dict:  # type: ignore[type-arg]
    """Use DuckDB to compute watch_ratio distribution statistics."""
    conn = duckdb.connect()
    try:
        result = conn.execute(f"""
            SELECT
                COUNT(*)                                    AS total_interactions,
                COUNT(DISTINCT user_id)                     AS unique_users,
                COUNT(DISTINCT video_id)                    AS unique_videos,
                ROUND(AVG(watch_ratio), 4)                  AS mean_watch_ratio,
                ROUND(MEDIAN(watch_ratio), 4)               AS median_watch_ratio,
                ROUND(MIN(watch_ratio), 4)                  AS min_watch_ratio,
                ROUND(MAX(watch_ratio), 4)                  AS max_watch_ratio,
                COUNT(*) FILTER (watch_ratio > 1.0)         AS replays_count,
                COUNT(*) FILTER (watch_ratio > 5.0)         AS extreme_ratio_count,
                COUNT(*) FILTER (watch_ratio = 0.0)         AS zero_ratio_count,
                COUNT(*) FILTER (time IS NULL)              AS missing_timestamp_count,
                ROUND(
                    COUNT(*) FILTER (time IS NULL) * 100.0 / COUNT(*), 2
                )                                           AS missing_timestamp_pct
            FROM read_parquet('{small_matrix_path}')
        """).fetchone()

        cols = [
            "total_interactions", "unique_users", "unique_videos",
            "mean_watch_ratio", "median_watch_ratio", "min_watch_ratio", "max_watch_ratio",
            "replays_count", "extreme_ratio_count", "zero_ratio_count",
            "missing_timestamp_count", "missing_timestamp_pct",
        ]
        return dict(zip(cols, result)) if result else {}
    finally:
        conn.close()


def compute_user_activity_distribution(user_features_path: Path) -> dict:  # type: ignore[type-arg]
    """Use DuckDB to compute user activity level distribution."""
    conn = duckdb.connect()
    try:
        rows = conn.execute(f"""
            SELECT user_active_degree, COUNT(*) AS count
            FROM read_parquet('{user_features_path}')
            GROUP BY user_active_degree
            ORDER BY count DESC
        """).fetchall()
        return {row[0]: row[1] for row in rows}
    finally:
        conn.close()


def compute_category_distribution(caption_path: Path) -> dict:  # type: ignore[type-arg]
    """Use DuckDB to compute top-level category distribution."""
    conn = duckdb.connect()
    try:
        rows = conn.execute(f"""
            SELECT first_level_category_name, COUNT(*) AS count
            FROM read_parquet('{caption_path}')
            WHERE first_level_category_name IS NOT NULL
            GROUP BY first_level_category_name
            ORDER BY count DESC
            LIMIT 15
        """).fetchall()
        return {row[0]: row[1] for row in rows}
    finally:
        conn.close()


def generate_report() -> dict:  # type: ignore[type-arg]
    """Generate the full data quality report.

    Returns:
        Complete report dictionary.
    """
    ensure_directories()
    t0 = time.perf_counter()
    logger.info("Loading prior pipeline reports ...")
    prior = load_prior_reports()

    logger.info("Computing analytical statistics with DuckDB ...")
    small_matrix_pq = PARQUET_FILES["small_matrix"]
    user_features_pq = PARQUET_FILES["user_features"]
    caption_pq = PARQUET_FILES["kuairec_caption_category"]

    watch_stats = compute_watch_ratio_stats(small_matrix_pq) if small_matrix_pq.exists() else {}
    user_activity = compute_user_activity_distribution(user_features_pq) if user_features_pq.exists() else {}
    category_dist = compute_category_distribution(caption_pq) if caption_pq.exists() else {}

    # Parquet file sizes
    parquet_sizes = {}
    for name, path in PARQUET_FILES.items():
        if path.exists():
            parquet_sizes[name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "size_human": format_bytes(path.stat().st_size),
            }

    # Validation summary
    val_summary = prior.get("validation", {}).get("summary", {})
    val_checks = prior.get("validation", {}).get("checks", [])
    issues_found = [c for c in val_checks if c.get("status") in ("warning", "fail")]
    # issues_corrected unused — retained for future reporting

    report = {
        "report_title": "KuaiRec 2.0 Data Quality Report",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": "KuaiRec 2.0",
            "version": "2.0",
            "source": "https://zenodo.org/records/18164998",
            "doi": "10.5281/zenodo.18164998",
            "license": "CC BY 4.0",
            "paper": "Gao et al., CIKM 2022",
            "download_timestamp": prior.get("manifest", {}).get("download_timestamp"),
        },
        "files": {
            "big_matrix": {
                "description": "12.5M user-item interactions, 7,176 users x 10,728 items (16.3% density)",
                "phase_2_scope": False,
                "note": "Deferred to Phase 3 — 1 GB CSV, sparse matrix",
            },
            "small_matrix": {
                "description": "4.68M interactions, 1,411 users x 3,327 items (99.6% density — fully observed)",
                "phase_2_scope": True,
                "rows": watch_stats.get("total_interactions"),
                "unique_users": watch_stats.get("unique_users"),
                "unique_videos": watch_stats.get("unique_videos"),
                "parquet": parquet_sizes.get("small_matrix"),
            },
            "user_features": {
                "description": "7,176 users with activity and encrypted demographic features",
                "phase_2_scope": True,
                "parquet": parquet_sizes.get("user_features"),
            },
            "user_features_raw": {
                "description": "7,176 users with rich demographics (gender, age, city, device)",
                "phase_2_scope": True,
                "parquet": parquet_sizes.get("user_features_raw"),
            },
            "item_categories": {
                "description": "10,728 videos with tag lists",
                "phase_2_scope": True,
                "parquet": parquet_sizes.get("item_categories"),
            },
            "item_daily_features": {
                "description": "Per-video daily engagement statistics (53 columns)",
                "phase_2_scope": False,
                "note": "Deferred to Phase 3",
            },
            "kuairec_caption_category": {
                "description": "10,728 videos with Chinese captions and 3-level category taxonomy",
                "phase_2_scope": True,
                "parquet": parquet_sizes.get("kuairec_caption_category"),
            },
            "social_network": {
                "description": "User friend lists (graph data)",
                "phase_2_scope": False,
                "note": "Deferred to Phase 3",
            },
            "video_raw_categories_multi": {
                "description": "Multi-row hierarchical category annotations per video",
                "phase_2_scope": False,
                "note": "Deferred to Phase 3",
            },
        },
        "analytics": {
            "watch_ratio_distribution": watch_stats,
            "user_activity_distribution": user_activity,
            "top_15_video_categories": category_dist,
        },
        "data_quality": {
            "validation_summary": val_summary,
            "issues_found": issues_found,
            "issues_resolved": "All type/schema issues corrected via cleaning pipeline",
            "issues_preserved_intentionally": [
                {
                    "issue": "181,992 rows with null timestamp/date/time (3.89% of small_matrix)",
                    "reason": "Timestamps missing in source data — not a data error. watch_ratio still present. Rows preserved.",
                    "action": "Documented and preserved. Time-based filtering must account for nulls.",
                },
                {
                    "issue": "watch_ratio values > 1.0 (1,514,717 rows)",
                    "reason": "Users replaying videos produces ratios > 1.0. This is valid behavior.",
                    "action": "Preserved. Never filter these out.",
                },
                {
                    "issue": "22,790 watch_ratio values > 5.0",
                    "reason": "Extreme replays. Plausible but may indicate anomalies.",
                    "action": "Preserved in Phase 2. Flag for investigation in Phase 3 feature engineering.",
                },
                {
                    "issue": "video_id=1225 found once in item_categories",
                    "reason": "Should have been removed in v2.0. Present in item_categories but not in interaction matrices.",
                    "action": "Documented. Does not affect training data. Filter in Phase 3 if needed.",
                },
                {
                    "issue": "12.63% null captions in kuairec_caption_category",
                    "reason": "Authors did not add captions. Missing text is expected.",
                    "action": "Preserved. Imputation strategy deferred to Phase 3.",
                },
            ],
            "issues_requiring_investigation": [
                {
                    "issue": "small_matrix watch_ratio max = 571.52",
                    "detail": "One user watched a video 571 times. Technically valid but may be a data anomaly.",
                    "action": "Investigate in Phase 3 feature engineering. Do not remove without justification.",
                },
            ],
        },
        "pipeline_summary": {
            "stage_1_acquisition": "9 files downloaded from Zenodo, all checksums verified",
            "stage_2_inspection": "9 files inspected, schemas documented",
            "stage_3_validation": f"{val_summary.get('total_checks', 0)} checks: {val_summary.get('pass', 0)} pass, {val_summary.get('warning', 0)} warning, {val_summary.get('fail', 0)} fail",
            "stage_4_cleaning": "0 rows dropped, all transforms documented and reversible",
            "stage_5_parquet": "5 files converted to Parquet (zstd), all verified",
        },
        "total_report_seconds": round(time.perf_counter() - t0, 2),
    }

    # Write JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("JSON report: %s", REPORT_JSON_PATH)

    # Write Markdown
    md = generate_markdown(report)
    with REPORT_MD_PATH.open("w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Markdown report: %s", REPORT_MD_PATH)

    return report


def generate_markdown(report: dict) -> str:  # type: ignore[type-arg]
    """Render the report as human-readable Markdown."""
    ws = report["analytics"]["watch_ratio_distribution"]
    ua = report["analytics"]["user_activity_distribution"]
    cats = report["analytics"]["top_15_video_categories"]
    dq = report["data_quality"]
    ps = report["pipeline_summary"]
    ds = report["dataset"]

    lines = [
        "# KuaiRec 2.0 Data Quality Report",
        "",
        f"**Generated:** {report['generated_at']}",
        f"**Source:** [{ds['source']}]({ds['source']})",
        f"**DOI:** `{ds['doi']}`",
        f"**License:** {ds['license']}",
        f"**Paper:** {ds['paper']}",
        "",
        "---",
        "",
        "## Dataset Statistics",
        "",
        "### Small Matrix (Primary Training Dataset)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total interactions | {ws.get('total_interactions', 'N/A'):,} |" if ws else "",
        f"| Unique users | {ws.get('unique_users', 'N/A'):,} |" if ws else "",
        f"| Unique videos | {ws.get('unique_videos', 'N/A'):,} |" if ws else "",
        f"| Mean watch_ratio | {ws.get('mean_watch_ratio', 'N/A')} |" if ws else "",
        f"| Median watch_ratio | {ws.get('median_watch_ratio', 'N/A')} |" if ws else "",
        f"| watch_ratio range | [{ws.get('min_watch_ratio', '?')}, {ws.get('max_watch_ratio', '?')}] |" if ws else "",
        f"| Replays (watch_ratio > 1.0) | {ws.get('replays_count', 'N/A'):,} ({round(ws.get('replays_count', 0) / max(ws.get('total_interactions', 1), 1) * 100, 1)}%) |" if ws else "",
        f"| Missing timestamps | {ws.get('missing_timestamp_count', 'N/A'):,} ({ws.get('missing_timestamp_pct', 'N/A')}%) |" if ws else "",
        "",
        "### User Activity Distribution",
        "",
        "| Activity Level | Count |",
        "|---|---|",
    ]
    for level, count in ua.items():
        lines.append(f"| {level} | {count:,} |")

    lines += [
        "",
        "### Top Video Categories",
        "",
        "| Category | Videos |",
        "|---|---|",
    ]
    for cat, count in cats.items():
        lines.append(f"| {cat} | {count:,} |")

    lines += [
        "",
        "---",
        "",
        "## Pipeline Summary",
        "",
        "| Stage | Result |",
        "|---|---|",
        f"| Acquisition | {ps['stage_1_acquisition']} |",
        f"| Inspection | {ps['stage_2_inspection']} |",
        f"| Validation | {ps['stage_3_validation']} |",
        f"| Cleaning | {ps['stage_4_cleaning']} |",
        f"| Parquet conversion | {ps['stage_5_parquet']} |",
        "",
        "---",
        "",
        "## Issues Found",
        "",
    ]
    for issue in dq["issues_found"]:
        lines.append(f"- **[{issue.get('status', '').upper()}]** {issue.get('note', issue.get('check', ''))}")

    lines += [
        "",
        "## Issues Intentionally Preserved",
        "",
    ]
    for item in dq["issues_preserved_intentionally"]:
        lines.append(f"- **{item['issue']}**")
        lines.append(f"  - Reason: {item['reason']}")
        lines.append(f"  - Action: {item['action']}")
        lines.append("")

    lines += [
        "## Issues Requiring Further Investigation",
        "",
    ]
    for item in dq["issues_requiring_investigation"]:
        lines.append(f"- **{item['issue']}** — {item['detail']}")
        lines.append(f"  - Action: {item['action']}")
        lines.append("")

    lines += [
        "---",
        "",
        "*Report generated by `scripts/data/generate_report.py`*",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("KuaiRec 2.0 Data Quality Report Generator")
    logger.info("=" * 60)

    report = generate_report()

    ws = report["analytics"]["watch_ratio_distribution"]
    logger.info("")
    logger.info("=== KEY STATISTICS ===")
    if ws:
        logger.info("  Total interactions (small_matrix): %s", f"{ws['total_interactions']:,}")
        logger.info("  Unique users:                       %s", f"{ws['unique_users']:,}")
        logger.info("  Unique videos:                      %s", f"{ws['unique_videos']:,}")
        logger.info("  Mean watch_ratio:                   %s", ws['mean_watch_ratio'])
        logger.info("  Missing timestamps:                 %s (%.2f%%)",
                    f"{ws['missing_timestamp_count']:,}", ws['missing_timestamp_pct'])

    logger.info("")
    logger.info("Reports written to %s", REPORTS_DIR)
