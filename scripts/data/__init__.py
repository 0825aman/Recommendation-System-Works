"""Data engineering scripts package for the Hybrid Recommendation Intelligence Platform.

This package contains standalone offline batch scripts for:
- Downloading and verifying the KuaiRec 2.0 dataset
- Inspecting raw files
- Validating data integrity
- Cleaning and standardizing
- Converting to Parquet
- Generating data quality reports

These scripts are NOT part of the FastAPI web application.
Run them directly from the project root:

    python scripts/data/download_dataset.py
    python scripts/data/inspect_dataset.py
    python scripts/data/validate_dataset.py
    python scripts/data/clean_dataset.py
    python scripts/data/convert_to_parquet.py
    python scripts/data/generate_report.py
"""
