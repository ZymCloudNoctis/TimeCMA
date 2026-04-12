import argparse
import csv
import gzip
import os
import re
import sys
from itertools import combinations

csv.field_size_limit(sys.maxsize)
STOCK_CODE_RE = re.compile(r"(\d{6})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="cleaned Snowball csv or csv.gz")
    parser.add_argument("--stock_pool_file", required=True, help="canonical stock pool csv/txt")
    parser.add_argument("--output_csv", required=True, help="output matrix csv")
    parser.add_argument("--start_date", default="", help="inclusive start date in YYYY-MM-DD")
    parser.add_argument("--end_date", default="", help="inclusive end date in YYYY-MM-DD")
    return parser.parse_args()


def open_csv(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def normalize_stock_code(value):
    if value is None:
        return ""
    text = str(value).strip()
    match = STOCK_CODE_RE.search(text)
    if match:
        return match.group(1)
    if text.isdigit():
        return text.zfill(6)
    return text


def load_stock_pool(stock_pool_file):
    codes = []
    seen = set()
    with open(stock_pool_file, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
        if first_row:
            first_code = normalize_stock_code(first_row[0] if first_row else "")
            has_header = not first_code.isdigit() or len(first_code) != 6
            if not has_header and first_code not in seen:
                codes.append(first_code)
                seen.add(first_code)
        for row in reader:
            if not row:
                continue
            code = normalize_stock_code(row[0])
            if len(code) == 6 and code.isdigit() and code not in seen:
                codes.append(code)
                seen.add(code)
    if not codes:
        raise ValueError(f"No stock codes found in {stock_pool_file}")
    return codes


def parse_stock_codes(raw_value, valid_codes):
    if not raw_value:
        return []
    seen = set()
    parsed = []
    for token in str(raw_value).split(","):
        code = normalize_stock_code(token)
        if code and code in valid_codes and code not in seen:
            seen.add(code)
            parsed.append(code)
    return parsed


def build_matrix(input_csv, codes, start_date="", end_date=""):
    code_set = set(codes)
    code_to_idx = {code: idx for idx, code in enumerate(codes)}
    matrix = [[0] * len(codes) for _ in codes]

    stats = {
        "total_rows": 0,
        "rows_in_window": 0,
        "rows_with_valid_codes": 0,
        "rows_with_pairs": 0,
        "rows_without_valid_codes": 0,
        "rows_outside_window": 0,
        "ignored_unknown_codes": set(),
        "window_min_date": "",
        "window_max_date": "",
    }

    with open_csv(input_csv) as handle:
        reader = csv.DictReader(handle)
        if "stock_codes" not in (reader.fieldnames or []):
            raise ValueError("Input csv does not contain a 'stock_codes' column.")
        if "publish_date" not in (reader.fieldnames or []):
            raise ValueError("Input csv does not contain a 'publish_date' column.")

        for row in reader:
            stats["total_rows"] += 1
            publish_date = (row.get("publish_date") or "").strip()
            if start_date and publish_date and publish_date < start_date:
                stats["rows_outside_window"] += 1
                continue
            if end_date and publish_date and publish_date > end_date:
                stats["rows_outside_window"] += 1
                continue

            stats["rows_in_window"] += 1
            if publish_date:
                if not stats["window_min_date"] or publish_date < stats["window_min_date"]:
                    stats["window_min_date"] = publish_date
                if not stats["window_max_date"] or publish_date > stats["window_max_date"]:
                    stats["window_max_date"] = publish_date

            raw_codes = (row.get("stock_codes") or "").split(",")
            for raw_code in raw_codes:
                code = normalize_stock_code(raw_code)
                if code and code not in code_set:
                    stats["ignored_unknown_codes"].add(code)

            parsed_codes = parse_stock_codes(row.get("stock_codes", ""), code_set)
            if not parsed_codes:
                stats["rows_without_valid_codes"] += 1
                continue

            stats["rows_with_valid_codes"] += 1
            if len(parsed_codes) < 2:
                continue

            stats["rows_with_pairs"] += 1
            for left_code, right_code in combinations(parsed_codes, 2):
                left_idx = code_to_idx[left_code]
                right_idx = code_to_idx[right_code]
                matrix[left_idx][right_idx] += 1
                matrix[right_idx][left_idx] += 1

    stats["ignored_unknown_codes"] = sorted(stats["ignored_unknown_codes"])
    return matrix, stats


def write_matrix(output_csv, codes, matrix):
    output_path = os.path.abspath(output_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + codes)
        for code, row in zip(codes, matrix):
            writer.writerow([code] + row)


def main():
    args = parse_args()
    codes = load_stock_pool(args.stock_pool_file)
    matrix, stats = build_matrix(
        input_csv=args.input_csv,
        codes=codes,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    write_matrix(args.output_csv, codes, matrix)

    print(f"Saved matrix to: {args.output_csv}")
    print(f"Stocks in matrix: {len(codes)}")
    print(f"Rows scanned: {stats['total_rows']}")
    print(f"Rows in date window: {stats['rows_in_window']}")
    print(f"Rows with valid stock codes: {stats['rows_with_valid_codes']}")
    print(f"Rows contributing co-occurrence pairs: {stats['rows_with_pairs']}")
    print(f"Rows without valid stock codes: {stats['rows_without_valid_codes']}")
    print(f"Rows outside date window: {stats['rows_outside_window']}")
    print(f"Window min date: {stats['window_min_date']}")
    print(f"Window max date: {stats['window_max_date']}")
    print(f"Ignored unknown codes: {len(stats['ignored_unknown_codes'])}")
    if stats["ignored_unknown_codes"]:
        preview = ", ".join(stats["ignored_unknown_codes"][:20])
        print(f"Unknown code preview: {preview}")


if __name__ == "__main__":
    main()
