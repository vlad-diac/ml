#!/usr/bin/env python3
"""Download the Zenodo record files-archive (ZIP) and extract it locally."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

DEFAULT_URL = "https://zenodo.org/api/records/5884485/files-archive"
DEFAULT_ARCHIVE_NAME = "5884485.zip"
CHUNK_SIZE = 1 << 20  # 1 MiB


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for name in zf.namelist():
        member = Path(name)
        if member.is_absolute() or ".." in member.parts:
            raise RuntimeError(f"Unsafe path in archive: {name!r}")
        full = (dest / member).resolve()
        full.relative_to(dest)
    zf.extractall(dest)


def download_archive(url: str, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as resp:  # noqa: S310 — URL is CLI-controlled / default Zenodo API
            total = resp.headers.get("Content-Length")
            total_n = int(total) if total and total.isdigit() else None
            written = 0
            with zip_path.open("wb") as out:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                    if total_n and written % (CHUNK_SIZE * 5) < CHUNK_SIZE:
                        pct = 100.0 * written / total_n
                        print(f"\r  {written / 1e6:.1f} / {total_n / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
            if total_n:
                print()
    except HTTPError as e:
        print(f"HTTP error {e.code}: {e.reason}", file=sys.stderr)
        raise SystemExit(1) from e
    except URLError as e:
        print(f"Download failed: {e.reason}", file=sys.stderr)
        raise SystemExit(1) from e


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"files-archive URL (default: Zenodo record 5884485)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to extract files into (created if missing)",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help=f"Where to save the ZIP before extracting (default: <output-dir>/{DEFAULT_ARCHIVE_NAME})",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Do not delete the ZIP after successful extraction",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    zip_path = args.zip_path if args.zip_path is not None else out_dir / DEFAULT_ARCHIVE_NAME

    print(f"Downloading:\n  {args.url}")
    print(f"Saving archive to:\n  {zip_path}")
    download_archive(args.url, zip_path)

    print(f"Extracting to:\n  {out_dir.resolve()}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        _safe_extract(zf, out_dir)

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        print("Removed archive (use --keep-zip to retain it).")

    print("Done.")


if __name__ == "__main__":
    main()
