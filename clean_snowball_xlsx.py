import argparse
import csv
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from html import unescape
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET


XML_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
TAG_ROW = f"{{{XML_NS}}}row"
TAG_CELL = f"{{{XML_NS}}}c"
TAG_VALUE = f"{{{XML_NS}}}v"
TAG_TEXT = f"{{{XML_NS}}}t"
STOCK_CODE_RE = re.compile(r"(?:SH|SZ)?(\d{6})", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
HTML_BREAK_RE = re.compile(r"<br\\s*/?>", re.IGNORECASE)
MULTI_WS_RE = re.compile(r"\s+")
TRAILING_DECIMAL_RE = re.compile(r"\.0$")

EXPECTED_HEADER = [
    "股票代码",
    "ID",
    "发帖者ID",
    "帖子发表者呢称",
    "帖子内容",
    "来源",
    "发帖时间",
    "点赞量",
    "转发量",
    "浏览量",
    "评论量",
    "帖子转评原发表者ID",
    "股票数量",
]

OUTPUT_FIELDS = [
    "post_id",
    "author_id",
    "author_name",
    "source",
    "publish_time",
    "publish_date",
    "publish_year",
    "stock_codes_raw",
    "stock_codes",
    "stock_count_raw",
    "stock_count_clean",
    "like_count",
    "repost_count",
    "view_count",
    "comment_count",
    "original_author_id",
    "contains_html",
    "content_length",
    "content",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="directory containing Snowball xlsx files")
    parser.add_argument("--output_csv", type=str, required=True, help="cleaned csv.gz output path")
    parser.add_argument("--summary_json", type=str, required=True, help="summary json output path")
    parser.add_argument("--stats_csv", type=str, required=True, help="per-file stats csv output path")
    parser.add_argument("--preview_csv", type=str, default="", help="optional preview csv output path")
    parser.add_argument("--preview_rows", type=int, default=1000, help="number of preview rows to save")
    parser.add_argument("--progress_every", type=int, default=100000, help="progress log interval")
    return parser.parse_args()


def workbook_sort_key(path):
    match = re.search(r"_(\d+)\.xlsx$", path.name)
    return int(match.group(1)) if match else path.name


def column_index_from_ref(cell_ref):
    letters = []
    for ch in cell_ref:
        if ch.isalpha():
            letters.append(ch.upper())
        else:
            break

    col = 0
    for ch in letters:
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return max(col - 1, 0)


def get_shared_strings(zf):
    shared = []
    if "xl/sharedStrings.xml" not in zf.namelist():
        return shared

    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    for si in root:
        text_parts = []
        for node in si.iter(TAG_TEXT):
            text_parts.append(node.text or "")
        shared.append("".join(text_parts))
    return shared


def get_first_sheet_target(zf):
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    workbook_rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in workbook_rels}
    sheets = workbook.find(f"{{{XML_NS}}}sheets")
    if sheets is None or not list(sheets):
        raise ValueError("Workbook does not contain any sheets")

    first_sheet = list(sheets)[0]
    rel_id = first_sheet.attrib[f"{{{REL_NS}}}id"]
    target = rel_map[rel_id]
    if not target.startswith("xl/"):
        target = f"xl/{target}"
    return target


def cell_value(cell, shared_strings):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join((node.text or "") for node in cell.iter(TAG_TEXT))

    value_node = cell.find(TAG_VALUE)
    if value_node is None:
        return ""

    value = value_node.text or ""
    if cell_type == "s":
        return shared_strings[int(value)]
    return value


def iter_xlsx_rows(xlsx_path):
    with ZipFile(xlsx_path) as zf:
        shared_strings = get_shared_strings(zf)
        sheet_target = get_first_sheet_target(zf)
        with zf.open(sheet_target) as handle:
            context = ET.iterparse(handle, events=("end",))
            for _, elem in context:
                if elem.tag != TAG_ROW:
                    continue

                row_values = []
                current_index = 0
                for cell in elem.findall(TAG_CELL):
                    cell_ref = cell.attrib.get("r", "")
                    target_index = column_index_from_ref(cell_ref) if cell_ref else current_index
                    while current_index < target_index:
                        row_values.append("")
                        current_index += 1
                    row_values.append(cell_value(cell, shared_strings))
                    current_index += 1

                yield row_values
                elem.clear()


def strip_text(value):
    return str(value or "").strip()


def parse_datetime(value):
    text = strip_text(value)
    if not text:
        return None

    text = TRAILING_DECIMAL_RE.sub("", text)
    for fmt in (
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue

    try:
        numeric = float(text)
        base = datetime(1899, 12, 30)
        return base + timedelta(days=numeric)
    except ValueError:
        return None


def parse_int(value):
    text = strip_text(value)
    if not text:
        return 0
    text = TRAILING_DECIMAL_RE.sub("", text).replace(",", "")
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return 0


def normalize_stock_codes(raw_value):
    raw_text = strip_text(raw_value)
    if not raw_text:
        return []

    codes = []
    seen = set()
    for match in STOCK_CODE_RE.finditer(raw_text):
        code = match.group(1)
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def clean_content(raw_value):
    text = strip_text(raw_value)
    contains_html = bool(re.search(r"<[^>]+>", text))
    if not text:
        return "", contains_html

    text = unescape(text)
    text = HTML_BREAK_RE.sub("\n", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = text.replace("\u3000", " ").replace("\xa0", " ").replace("\ufeff", "")
    text = MULTI_WS_RE.sub(" ", text).strip()
    return text, contains_html


def build_record(header, row):
    padded = row + [""] * max(0, len(header) - len(row))
    record = dict(zip(header, padded[: len(header)]))

    publish_dt = parse_datetime(record.get("发帖时间"))
    stock_codes = normalize_stock_codes(record.get("股票代码"))
    content, contains_html = clean_content(record.get("帖子内容"))

    publish_time = publish_dt.strftime("%Y-%m-%d %H:%M:%S") if publish_dt else ""
    publish_date = publish_dt.strftime("%Y-%m-%d") if publish_dt else ""
    publish_year = publish_dt.year if publish_dt else ""

    return {
        "post_id": strip_text(record.get("ID")),
        "author_id": strip_text(record.get("发帖者ID")),
        "author_name": strip_text(record.get("帖子发表者呢称")),
        "source": strip_text(record.get("来源")),
        "publish_time": publish_time,
        "publish_date": publish_date,
        "publish_year": publish_year,
        "stock_codes_raw": strip_text(record.get("股票代码")),
        "stock_codes": ",".join(stock_codes),
        "stock_count_raw": parse_int(record.get("股票数量")),
        "stock_count_clean": len(stock_codes),
        "like_count": parse_int(record.get("点赞量")),
        "repost_count": parse_int(record.get("转发量")),
        "view_count": parse_int(record.get("浏览量")),
        "comment_count": parse_int(record.get("评论量")),
        "original_author_id": strip_text(record.get("帖子转评原发表者ID")),
        "contains_html": int(contains_html),
        "content_length": len(content),
        "content": content,
    }


def ensure_parent(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def clean_directory(args):
    input_dir = Path(args.input_dir)
    workbook_paths = sorted(input_dir.glob("*.xlsx"), key=workbook_sort_key)
    if not workbook_paths:
        raise FileNotFoundError(f"No xlsx files found in {input_dir}")

    ensure_parent(args.output_csv)
    ensure_parent(args.summary_json)
    ensure_parent(args.stats_csv)
    if args.preview_csv:
        ensure_parent(args.preview_csv)

    summary = {
        "input_dir": str(input_dir),
        "files_processed": [],
        "rows_total": 0,
        "rows_written": 0,
        "rows_deduplicated": 0,
        "rows_missing_post_id": 0,
        "rows_missing_publish_time": 0,
        "rows_empty_content": 0,
        "rows_with_html": 0,
        "rows_multistock": 0,
        "rows_stock_count_mismatch": 0,
        "unique_sources": set(),
        "year_counter": Counter(),
        "source_counter": Counter(),
        "global_min_time": None,
        "global_max_time": None,
    }

    seen_post_ids = set()
    preview_rows = []

    with gzip.open(args.output_csv, "wt", encoding="utf-8", newline="") as out_handle, open(
        args.stats_csv, "w", encoding="utf-8", newline=""
    ) as stats_handle:
        writer = csv.DictWriter(out_handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        stats_writer = csv.DictWriter(
            stats_handle,
            fieldnames=[
                "file_name",
                "rows_total",
                "rows_written",
                "rows_deduplicated",
                "rows_missing_post_id",
                "rows_missing_publish_time",
                "rows_empty_content",
                "rows_with_html",
                "rows_multistock",
                "rows_stock_count_mismatch",
                "min_publish_time",
                "max_publish_time",
            ],
        )
        stats_writer.writeheader()

        for workbook_path in workbook_paths:
            file_stats = defaultdict(int)
            file_stats["file_name"] = workbook_path.name
            file_min_time = None
            file_max_time = None
            header = None

            for row_index, row in enumerate(iter_xlsx_rows(workbook_path)):
                if row_index == 0:
                    header = row[: len(EXPECTED_HEADER)]
                    if header != EXPECTED_HEADER:
                        raise ValueError(
                            f"Unexpected header in {workbook_path.name}: {header}. Expected {EXPECTED_HEADER}."
                        )
                    continue

                file_stats["rows_total"] += 1
                summary["rows_total"] += 1

                record = build_record(header, row)
                post_id = record["post_id"]
                if not post_id:
                    summary["rows_missing_post_id"] += 1
                    file_stats["rows_missing_post_id"] += 1
                    continue

                if post_id in seen_post_ids:
                    summary["rows_deduplicated"] += 1
                    file_stats["rows_deduplicated"] += 1
                    continue
                seen_post_ids.add(post_id)

                if not record["publish_time"]:
                    summary["rows_missing_publish_time"] += 1
                    file_stats["rows_missing_publish_time"] += 1

                if not record["content"]:
                    summary["rows_empty_content"] += 1
                    file_stats["rows_empty_content"] += 1

                if record["contains_html"]:
                    summary["rows_with_html"] += 1
                    file_stats["rows_with_html"] += 1

                if record["stock_count_clean"] > 1:
                    summary["rows_multistock"] += 1
                    file_stats["rows_multistock"] += 1

                if record["stock_count_raw"] != record["stock_count_clean"]:
                    summary["rows_stock_count_mismatch"] += 1
                    file_stats["rows_stock_count_mismatch"] += 1

                if record["publish_time"]:
                    publish_dt = datetime.strptime(record["publish_time"], "%Y-%m-%d %H:%M:%S")
                    file_min_time = publish_dt if file_min_time is None or publish_dt < file_min_time else file_min_time
                    file_max_time = publish_dt if file_max_time is None or publish_dt > file_max_time else file_max_time
                    summary["global_min_time"] = (
                        publish_dt
                        if summary["global_min_time"] is None or publish_dt < summary["global_min_time"]
                        else summary["global_min_time"]
                    )
                    summary["global_max_time"] = (
                        publish_dt
                        if summary["global_max_time"] is None or publish_dt > summary["global_max_time"]
                        else summary["global_max_time"]
                    )
                    summary["year_counter"][str(record["publish_year"])] += 1

                if record["source"]:
                    summary["unique_sources"].add(record["source"])
                    summary["source_counter"][record["source"]] += 1

                writer.writerow(record)
                summary["rows_written"] += 1
                file_stats["rows_written"] += 1

                if args.preview_csv and len(preview_rows) < args.preview_rows:
                    preview_rows.append(record.copy())

                if args.progress_every and summary["rows_total"] % args.progress_every == 0:
                    print(
                        f"[progress] processed={summary['rows_total']:,} "
                        f"written={summary['rows_written']:,} dedup={summary['rows_deduplicated']:,}"
                    )

            file_stats["min_publish_time"] = file_min_time.strftime("%Y-%m-%d %H:%M:%S") if file_min_time else ""
            file_stats["max_publish_time"] = file_max_time.strftime("%Y-%m-%d %H:%M:%S") if file_max_time else ""
            stats_writer.writerow(file_stats)
            summary["files_processed"].append(dict(file_stats))
            print(
                f"[file] {workbook_path.name} rows={file_stats['rows_total']:,} "
                f"written={file_stats['rows_written']:,} min={file_stats['min_publish_time']} "
                f"max={file_stats['max_publish_time']}"
            )

    if args.preview_csv:
        with open(args.preview_csv, "w", encoding="utf-8", newline="") as preview_handle:
            preview_writer = csv.DictWriter(preview_handle, fieldnames=OUTPUT_FIELDS)
            preview_writer.writeheader()
            preview_writer.writerows(preview_rows)

    summary["unique_sources"] = sorted(summary["unique_sources"])
    summary["year_counter"] = dict(sorted(summary["year_counter"].items()))
    summary["source_counter"] = dict(summary["source_counter"].most_common())
    summary["global_min_time"] = (
        summary["global_min_time"].strftime("%Y-%m-%d %H:%M:%S") if summary["global_min_time"] else ""
    )
    summary["global_max_time"] = (
        summary["global_max_time"].strftime("%Y-%m-%d %H:%M:%S") if summary["global_max_time"] else ""
    )

    with open(args.summary_json, "w", encoding="utf-8") as summary_handle:
        json.dump(summary, summary_handle, ensure_ascii=False, indent=2)

    print(f"[done] cleaned csv: {args.output_csv}")
    print(f"[done] summary json: {args.summary_json}")
    print(f"[done] stats csv: {args.stats_csv}")
    if args.preview_csv:
        print(f"[done] preview csv: {args.preview_csv}")


if __name__ == "__main__":
    clean_directory(parse_args())
