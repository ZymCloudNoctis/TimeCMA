import argparse

from utils.graph_utils import build_and_save_cooccurrence_graph


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=".", help="root path of the dataset")
    parser.add_argument("--data_path", type=str, required=True, help="csv file or dataset name")
    parser.add_argument("--output_path", type=str, default="", help="output csv path")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--seq_len", type=int, default=96, help="sequence length used by the dataset split")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction length used by the dataset split")
    parser.add_argument("--target", type=str, default="OT", help="target column name")
    parser.add_argument("--event_threshold", type=float, default=0.5, help="z-score threshold for event counting")
    parser.add_argument("--min_weight", type=int, default=1, help="minimum edge weight kept in the graph")
    return parser.parse_args()


def main():
    args = parse_args()
    graph_file, node_columns = build_and_save_cooccurrence_graph(
        root_path=args.root_path,
        data_path=args.data_path,
        output_path=args.output_path or None,
        split=args.split,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        target=args.target,
        event_threshold=args.event_threshold,
        min_weight=args.min_weight,
    )
    print(f"Saved graph to: {graph_file}")
    print(f"Node count: {len(node_columns)}")


if __name__ == "__main__":
    main()
