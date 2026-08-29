"""Join predicted tags to the product metadata sheet and rank tags by engagement.

The brief opens with the real problem -- marketing wants to know "what content
is popular and engaging" -- and supplies meta_data.xslx with file names, product
names, categories, prices and image views. Tagging alone answers "what is in
this image"; joining tags to views answers the question actually asked: which
kinds of image earn attention.

The sheet's exact column headers are not specified, so columns are matched
fuzzily by keyword ("views", "price", "categor", ...) and the join is on image
file name. Both .xlsx and .csv are accepted.
"""
from __future__ import annotations

import os
from statistics import median

from content_analysis_agent.logconf import get_logger

log = get_logger(__name__)

# (keyword, canonical field), most specific first. Order matters: "Product
# price" must resolve to price rather than product, and a bare "Name" column --
# which is what the supplied sheet actually uses for the file name -- must only
# fall through to `file` after "product name" has had its chance.
_COLUMN_HINTS = [
    ("price", "price"),
    ("view", "views"), ("impression", "views"),
    ("categor", "category"),
    ("filename", "file"), ("file name", "file"), ("file", "file"),
    ("image name", "file"),
    ("product name", "product"), ("model", "product"), ("product", "product"),
    ("name", "file"),
]


def _canonical(headers: list[str]) -> dict[str, str]:
    """Map each source header to a canonical field name where recognised."""
    out: dict[str, str] = {}
    for header in headers:
        low = str(header).strip().lower()
        for hint, field in _COLUMN_HINTS:
            if hint in low and field not in out.values():
                out[header] = field
                break
    return out


def load_metadata(path: str) -> list[dict]:
    """Read the metadata sheet into records with canonical field names.

    Raises FileNotFoundError with an actionable message when the sheet is
    missing, since it ships with the task rather than with this repo.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Metadata sheet not found at {path!r}. It is supplied with the "
            f"task data (meta_data.xslx); pass its location with --metadata.")

    import pandas as pd
    frame = (pd.read_csv(path) if path.lower().endswith(".csv")
             else pd.read_excel(path))
    mapping = _canonical(list(frame.columns))
    if "file" not in mapping.values():
        raise ValueError(
            f"No file-name column found in {path!r}. Columns present: "
            f"{list(frame.columns)}")

    records = []
    for row in frame.to_dict("records"):
        rec = {field: row[src] for src, field in mapping.items()}
        rec["file"] = os.path.basename(str(rec["file"])).strip()
        records.append(rec)
    log.info("metadata_loaded", extra={"path": path, "rows": len(records),
                                       "fields": sorted(set(mapping.values()))})
    return records


def _key(category, product, file) -> tuple:
    return (str(category or "").strip().lower(),
            str(product or "").strip().lower(), str(file or "").strip())


def join_tags(results: list[dict], metadata: list[dict]) -> list[dict]:
    """Attach metadata to tagged results.

    Matches on (category, model, file name) where the result carries them,
    falling back to file name alone. The supplied sheet repeats the same file
    name across products -- the labelled training names are tag lists, not
    unique ids -- so file name on its own is not a key.

    `results` are the dicts produced by pipeline.results_to_dicts.
    Images with no metadata row are dropped and reported by the caller.
    """
    by_key = {_key(r.get("category"), r.get("product"), r["file"]): r
              for r in metadata}
    by_name: dict[str, dict] = {}
    for r in metadata:
        by_name.setdefault(r["file"], r)
    joined = []
    for res in results:
        name = os.path.basename(res["path"])
        meta = (by_key.get(_key(res.get("category"), res.get("model"), name))
                or by_name.get(name))
        if meta is None:
            continue
        row = dict(res)
        row.update({k: v for k, v in meta.items() if k != "file"})
        joined.append(row)
    log.info("metadata_joined", extra={"tagged": len(results),
                                       "matched": len(joined)})
    return joined


def rows_from_sheet(records: list[dict]) -> list[dict]:
    """Use the sheet's own labelled file names as the tags.

    The supplied sheet documents the *training* images, whose names encode
    their tags, and carries no rows for the unlabelled test images. So the
    engagement question can be answered straight from the sheet -- real tags
    against real view counts, with no model in the loop and nothing to join.
    """
    from content_analysis_agent.labels import parse_tags_from_filename

    out = []
    for rec in records:
        tags = parse_tags_from_filename(rec["file"])
        if not tags:
            continue
        row = dict(rec)
        row["tags"] = tags
        row["path"] = rec["file"]
        out.append(row)
    log.info("rows_from_sheet", extra={"rows": len(records),
                                       "labelled": len(out)})
    return out


def _to_number(value) -> float | None:
    try:
        num = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return None if num != num else num          # drop NaN


def tag_engagement(joined: list[dict], metric: str = "views",
                   min_support: int = 2) -> list[dict]:
    """Mean engagement per tag, with lift against the overall mean.

    Returns one record per tag: how many images carry it, the mean and median
    metric for those images, and the ratio to the overall mean (lift > 1 means
    images with that tag out-perform the average).
    """
    values = [(row, _to_number(row.get(metric))) for row in joined]
    values = [(row, val) for row, val in values if val is not None]
    if not values:
        return []
    overall = sum(v for _, v in values) / len(values)

    per_tag: dict[str, list[float]] = {}
    for row, val in values:
        for tag in row.get("tags", []):
            per_tag.setdefault(tag, []).append(val)

    report = []
    for tag, vals in per_tag.items():
        if len(vals) < min_support:
            continue
        mean = sum(vals) / len(vals)
        report.append({
            "tag": tag,
            "images": len(vals),
            f"mean_{metric}": round(mean, 1),
            f"median_{metric}": round(median(vals), 1),
            "lift": round(mean / overall, 2) if overall else 0.0,
        })
    report.sort(key=lambda r: r["lift"], reverse=True)
    return report


def format_engagement(report: list[dict], metric: str = "views",
                      overall: float | None = None) -> str:
    """Render the engagement table for the CLI."""
    if not report:
        return "No tag met the minimum support, or no numeric metric found."
    width = max(len(r["tag"]) for r in report)
    lines = [f"{'tag':<{width}}  images  mean {metric:<10} lift",
             "-" * (width + 32)]
    for r in report:
        lines.append(f"{r['tag']:<{width}}  {r['images']:>6}  "
                     f"{r[f'mean_{metric}']:>14,.1f}  {r['lift']:>5.2f}x")
    if overall is not None:
        lines.append(f"\nOverall mean {metric}: {overall:,.1f}")
    return "\n".join(lines)


# --------------------------- synthetic sheet --------------------------------

def write_synthetic_metadata(image_paths: list[str], out_path: str,
                             seed: int = 7) -> str:
    """Write a SYNTHETIC metadata sheet so the join can be exercised offline.

    The real meta_data.xslx ships with the task data. When it is not to hand,
    this generates a stand-in with the same shape (file, product, category,
    price, views). The brief permits public or synthetic/mock data for anything
    extra; numbers produced from this sheet demonstrate the pipeline and are
    not findings about real products.

    Views are drawn deterministically from the seed, with a deliberate bump for
    a couple of tags so the ranking has something to surface.
    """
    import csv
    import random

    rng = random.Random(seed)
    rows = []
    for path in image_paths:
        parts = os.path.normpath(path).split(os.sep)
        category = parts[-3] if len(parts) >= 3 else ""
        product = parts[-2] if len(parts) >= 2 else ""
        # A carousel's first images are the ones shoppers actually see.
        name = os.path.basename(path)
        position = 1
        for chunk in name.replace(".", "_").split("_"):
            if chunk.isdigit() and len(chunk) <= 2:
                position = int(chunk)
                break
        views = int(rng.gauss(5000, 1200) / max(position, 1) ** 0.5)
        rows.append({"file name": name, "product name": product,
                     "category": category,
                     "price": round(rng.uniform(80, 2200), 2),
                     "image views": max(views, 50)})

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.warning("synthetic_metadata_written", extra={"path": out_path,
                                                     "rows": len(rows)})
    return out_path
