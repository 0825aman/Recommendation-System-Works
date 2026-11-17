"""KuaiRec 2.0 dataset downloader with checksum verification.

Downloads all required files from Zenodo, verifies MD5 checksums,
extracts the zip archive, and writes a dataset manifest with provenance.

Usage (from project root):
    python scripts/data/download_dataset.py

The script is idempotent: files that already exist and pass checksum
verification are skipped. Use --force to re-download everything.

Requirements:
    pip install -e ".[data]"   (tqdm, rich already installed)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.common import (
    DOWNLOAD_MANIFEST,
    DATASET_LICENSE,
    KUAIREC_ZIP,
    MANIFEST_PATH,
    RAW_DIR,
    ZENODO_DOI,
    ZENODO_RECORD,
    ensure_directories,
    format_bytes,
    get_logger,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logger = get_logger(__name__)

CHUNK_SIZE = 65_536  # 64 KB chunks


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_md5(path: Path, show_progress: bool = False) -> str:
    """Compute the MD5 hash of a file.

    Args:
        path: Path to the file.
        show_progress: Show a tqdm progress bar.

    Returns:
        Lowercase hex MD5 digest string.
    """
    hasher = hashlib.md5()
    total = path.stat().st_size

    if show_progress and HAS_TQDM:
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=f"MD5 {path.name}", leave=False)
    else:
        bar = None

    with path.open("rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            hasher.update(chunk)
            if bar:
                bar.update(len(chunk))

    if bar:
        bar.close()

    return hasher.hexdigest()


def download_file(url: str, destination: Path, expected_size: int) -> None:
    """Download a file from a URL with a progress bar.

    Args:
        url: Source URL.
        destination: Local path to write to.
        expected_size: Expected file size in bytes (for progress display).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s -> %s", url, destination.name)
    logger.info("Expected size: %s", format_bytes(expected_size))

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "HRIP-DataPipeline/2.0 (academic research)"},
    )

    with urllib.request.urlopen(request) as response:  # noqa: S310
        if HAS_TQDM:
            bar = tqdm(
                total=expected_size,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            )
        else:
            bar = None

        with destination.open("wb") as out_file:
            downloaded = 0
            while chunk := response.read(CHUNK_SIZE):
                out_file.write(chunk)
                downloaded += len(chunk)
                if bar:
                    bar.update(len(chunk))

        if bar:
            bar.close()

    actual_size = destination.stat().st_size
    logger.info("Downloaded: %s (%s)", destination.name, format_bytes(actual_size))


def verify_checksum(path: Path, expected_md5: str) -> bool:
    """Verify a file's MD5 checksum.

    Args:
        path: Path to the file.
        expected_md5: Expected MD5 hex string.

    Returns:
        True if the checksum matches.
    """
    logger.info("Verifying checksum for %s ...", path.name)
    actual = compute_md5(path, show_progress=True)
    if actual == expected_md5:
        logger.info("✓ Checksum OK: %s", path.name)
        return True
    logger.error(
        "✗ Checksum MISMATCH for %s: expected %s, got %s",
        path.name,
        expected_md5,
        actual,
    )
    return False


def extract_zip(zip_path: Path, extract_to: Path) -> list[str]:
    """Extract a zip archive, flattening KuaiRec's internal directory structure.

    The zip contains files under KuaiRec/data/*.csv.
    This function extracts CSVs directly into extract_to/.

    Args:
        zip_path: Path to the zip file.
        extract_to: Directory to extract CSV files into.

    Returns:
        List of extracted file names.
    """
    logger.info("Extracting %s ...", zip_path.name)
    extracted = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        csv_members = [m for m in members if m.endswith(".csv")]
        logger.info("Found %d CSV files in zip", len(csv_members))

        for member in csv_members:
            filename = Path(member).name
            dest = extract_to / filename

            if dest.exists():
                logger.info("  Already exists, skipping: %s", filename)
                extracted.append(filename)
                continue

            logger.info("  Extracting: %s", filename)
            with zf.open(member) as src, dest.open("wb") as dst:
                while chunk := src.read(CHUNK_SIZE):
                    dst.write(chunk)

            extracted.append(filename)
            logger.info("  → %s (%s)", filename, format_bytes(dest.stat().st_size))

    logger.info("Extraction complete: %d files", len(extracted))
    return extracted


def write_manifest(results: list[dict]) -> None:  # type: ignore[type-arg]
    """Write the dataset manifest JSON file.

    Args:
        results: List of download result dictionaries.
    """
    manifest = {
        "dataset": "KuaiRec 2.0",
        "version": "2.0",
        "source": ZENODO_RECORD,
        "doi": ZENODO_DOI,
        "license": DATASET_LICENSE,
        "download_timestamp": datetime.now(UTC).isoformat(),
        "files": results,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written to %s", MANIFEST_PATH)


# ---------------------------------------------------------------------------
# Main download routine
# ---------------------------------------------------------------------------

def download_all(force: bool = False) -> bool:
    """Download and verify all KuaiRec 2.0 files.

    Args:
        force: If True, re-download even if files already exist.

    Returns:
        True if all files downloaded and verified successfully.
    """
    ensure_directories()
    results = []
    all_ok = True

    for entry in DOWNLOAD_MANIFEST:
        dest = Path(entry["destination"])
        expected_md5 = entry["md5"]
        url = entry["url"]
        expected_size = entry["size_bytes"]

        result: dict = {  # type: ignore[type-arg]
            "key": entry["key"],
            "url": url,
            "destination": str(dest),
            "expected_size_bytes": expected_size,
            "expected_md5": expected_md5,
            "status": "unknown",
        }

        # Skip if already downloaded and checksum matches
        if not force and dest.exists():
            if verify_checksum(dest, expected_md5):
                result["status"] = "already_verified"
                result["actual_size_bytes"] = dest.stat().st_size
                result["actual_md5"] = expected_md5
                results.append(result)
                continue
            else:
                logger.warning("Checksum mismatch on existing file, re-downloading: %s", dest.name)

        try:
            download_file(url, dest, expected_size)

            if verify_checksum(dest, expected_md5):
                result["status"] = "downloaded_verified"
                result["actual_size_bytes"] = dest.stat().st_size
                result["actual_md5"] = expected_md5
            else:
                result["status"] = "checksum_failed"
                all_ok = False

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to download %s: %s", entry["key"], exc)
            result["status"] = "failed"
            result["error"] = str(exc)
            all_ok = False

        results.append(result)

        # Extract zip after successful download
        if entry.get("extract") and result["status"] in ("downloaded_verified", "already_verified"):
            try:
                extract_zip(KUAIREC_ZIP, RAW_DIR)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to extract zip: %s", exc)
                all_ok = False

    write_manifest(results)
    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and verify the KuaiRec 2.0 dataset from Zenodo."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("KuaiRec 2.0 Dataset Downloader")
    logger.info("Source: %s", ZENODO_RECORD)
    logger.info("License: %s", DATASET_LICENSE)
    logger.info("=" * 60)

    success = download_all(force=args.force)

    if success:
        logger.info("All files downloaded and verified successfully.")
        logger.info("Manifest: %s", MANIFEST_PATH)
        sys.exit(0)
    else:
        logger.error("One or more files failed. Check the log above.")
        sys.exit(1)
