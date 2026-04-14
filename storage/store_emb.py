import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.data_loader_multistock import MultiStockSaveDataset, resolve_embedding_dir
from gen_prompt_emb import GenPromptEmb

try:
    import h5py
except ImportError:
    h5py = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_path", type=str, default="dataset/HS300/all_stocks_complete_data.csv")
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--freq", type=str, default="d")
    parser.add_argument("--num_nodes", type=int, default=300)
    parser.add_argument("--input_len", type=int, default=60)
    parser.add_argument("--output_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--l_layers", type=int, default=12)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--divide", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num_workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--task_name", type=str, default="multistock", choices=["multistock"])
    parser.add_argument("--stock_pool_file", type=str, default="")
    parser.add_argument("--target_horizon", type=int, default=5)
    parser.add_argument("--start_date", type=str, default="")
    parser.add_argument("--end_date", type=str, default="")
    parser.add_argument("--train_start_date", type=str, default="")
    parser.add_argument("--train_end_date", type=str, default="")
    parser.add_argument("--val_start_date", type=str, default="")
    parser.add_argument("--val_end_date", type=str, default="")
    parser.add_argument("--trainval_start_date", type=str, default="")
    parser.add_argument("--trainval_end_date", type=str, default="")
    parser.add_argument("--test_start_date", type=str, default="")
    parser.add_argument("--test_end_date", type=str, default="")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--embedding_tag", type=str, default="")
    return parser.parse_args()


def get_dataset(args):
    if not args.stock_pool_file:
        raise ValueError("Embedding generation requires --stock_pool_file.")
    return MultiStockSaveDataset(
        root_path=args.root_path,
        stock_pool_file=args.stock_pool_file,
        flag=args.divide,
        size=[args.input_len, 0, args.output_len],
        data_path=args.data_path,
        freq=args.freq,
        scale=False,
        target_horizon=args.target_horizon,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        train_start_date=args.train_start_date or None,
        train_end_date=args.train_end_date or None,
        val_start_date=args.val_start_date or None,
        val_end_date=args.val_end_date or None,
        trainval_start_date=args.trainval_start_date or None,
        trainval_end_date=args.trainval_end_date or None,
        test_start_date=args.test_start_date or None,
        test_end_date=args.test_end_date or None,
        val_ratio=args.val_ratio,
        embedding_tag=args.embedding_tag or "",
    )


def _save_embedding_file(file_stem, save_data):
    if h5py is not None:
        file_path = f"{file_stem}.h5"
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("embeddings", data=save_data)
        return file_path

    file_path = f"{file_stem}.npy"
    np.save(file_path, save_data)
    return file_path


def save_embeddings(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(args)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    gen_prompt_emb = GenPromptEmb(
        device=device,
        input_len=args.input_len,
        data_path=args.data_path,
        model_name=args.model_name,
        d_model=args.d_model,
        layer=args.l_layers,
        divide=args.divide,
        freq=args.freq,
        task_name="multistock",
        stock_codes=dataset.stock_codes,
        prompt_feature_index=dataset.prompt_feature_index,
    ).to(device)

    save_path = resolve_embedding_dir(args.data_path, args.divide, embedding_tag=args.embedding_tag or "")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("./Results/emb_logs/", exist_ok=True)

    count = 0
    for x, _, x_mark, _ in data_loader:
        embeddings = gen_prompt_emb.generate_embeddings(x.to(device), x_mark.to(device))
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)

        batch_size = embeddings.shape[0]
        for batch_index in range(batch_size):
            file_stem = os.path.join(save_path, str(count))
            save_data = embeddings[batch_index].cpu().numpy()
            _save_embedding_file(file_stem, save_data)
            count += 1


if __name__ == "__main__":
    parsed_args = parse_args()
    start_time = time.time()
    save_embeddings(parsed_args)
    end_time = time.time()
    print(f"Total time spent: {(end_time - start_time) / 60:.4f} minutes")
