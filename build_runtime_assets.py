import argparse
import csv
import os

import pandas as pd

from utils.graph_utils import load_stock_pool, normalize_stock_code


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market_data", type=str, required=True, help="market data csv path")
    parser.add_argument("--stock_pool_file", type=str, required=True, help="canonical stock pool csv/txt path")
    parser.add_argument("--graph_file", type=str, default="", help="canonical graph csv path")
    parser.add_argument("--output_stock_pool", type=str, required=True, help="runtime stock pool output csv path")
    parser.add_argument("--output_graph", type=str, default="", help="runtime graph output csv path")
    return parser.parse_args()


def _detect_code_field(fieldnames):
    lowered = {field.lower(): field for field in fieldnames}
    for candidate in ["stock_code", "code", "ts_code", "ticker", "symbol", "股票代码"]:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(f"Unable to find stock code column in market data. Found columns: {fieldnames}")


def load_market_codes(market_data):
    with open(market_data, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        code_field = _detect_code_field(reader.fieldnames or [])
        codes = set()
        for row in reader:
            code = normalize_stock_code(row.get(code_field))
            if code:
                codes.add(code)
    if not codes:
        raise ValueError(f"No valid stock codes found in market data: {market_data}")
    return codes


def prepare_runtime_assets(market_data, stock_pool_file, graph_file, output_stock_pool, output_graph):
    canonical_pool = load_stock_pool(stock_pool_file)
    market_codes = load_market_codes(market_data)
    graph_df = None
    graph_codes = set()

    if graph_file:
        graph_df = pd.read_csv(graph_file, index_col=0)
        graph_df.index = graph_df.index.map(normalize_stock_code)
        graph_df.columns = graph_df.columns.map(normalize_stock_code)
        graph_codes = set(graph_df.index).intersection(graph_df.columns)
        runtime_codes = [code for code in canonical_pool if code in market_codes and code in graph_codes]
    else:
        runtime_codes = [code for code in canonical_pool if code in market_codes]

    if not runtime_codes:
        if graph_file:
            raise ValueError("No overlapping stock codes between market data, stock pool, and graph.")
        raise ValueError("No overlapping stock codes between market data and stock pool.")

    os.makedirs(os.path.dirname(os.path.abspath(output_stock_pool)), exist_ok=True)
    pd.DataFrame({"stock_code": runtime_codes}).to_csv(output_stock_pool, index=False)
    if graph_df is not None and output_graph:
        os.makedirs(os.path.dirname(os.path.abspath(output_graph)), exist_ok=True)
        graph_df.loc[runtime_codes, runtime_codes].to_csv(output_graph)

    missing_from_market = [code for code in canonical_pool if code not in market_codes]
    missing_from_graph = [code for code in canonical_pool if graph_file and code not in graph_codes]

    print(f"Canonical stock pool: {len(canonical_pool)}")
    print(f"Market data codes: {len(market_codes)}")
    if graph_file:
        print(f"Graph codes: {len(graph_codes)}")
    print(f"Runtime stock pool: {len(runtime_codes)}")
    if missing_from_market:
        print(f"Excluded {len(missing_from_market)} codes missing from market data: {missing_from_market[:20]}")
    if missing_from_graph:
        print(f"Excluded {len(missing_from_graph)} codes missing from graph: {missing_from_graph[:20]}")
    print(f"Saved runtime stock pool to: {output_stock_pool}")
    if graph_df is not None and output_graph:
        print(f"Saved runtime graph to: {output_graph}")


if __name__ == "__main__":
    args = parse_args()
    prepare_runtime_assets(
        market_data=args.market_data,
        stock_pool_file=args.stock_pool_file,
        graph_file=args.graph_file,
        output_stock_pool=args.output_stock_pool,
        output_graph=args.output_graph,
    )
