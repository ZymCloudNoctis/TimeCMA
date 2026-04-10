import torch
import sys
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader
from data_provider.data_loader_save import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from data_provider.data_loader_multistock import MultiStockSaveDataset
from gen_prompt_emb import GenPromptEmb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/", help="root path of the data file")
    parser.add_argument("--freq", type=str, default="h", help="frequency for time features encoding")
    parser.add_argument("--num_nodes", type=int, default=7)
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--output_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--l_layers", type=int, default=12)
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--divide", type=str, default="train")
    parser.add_argument("--num_workers", type=int, default=min(10, os.cpu_count()))
    parser.add_argument("--task_name", type=str, default="legacy", choices=["legacy", "multistock"])
    parser.add_argument("--stock_pool_file", type=str, default="", help="ordered stock pool file for multi-stock tasks")
    parser.add_argument("--target_horizon", type=int, default=5, help="future return horizon for multi-stock tasks")
    parser.add_argument("--start_date", type=str, default="", help="inclusive start date for multi-stock filtering")
    parser.add_argument("--end_date", type=str, default="", help="inclusive end date for multi-stock filtering")
    parser.add_argument("--trainval_start_date", type=str, default="", help="inclusive start date for train/val samples")
    parser.add_argument("--trainval_end_date", type=str, default="", help="inclusive end date for train/val samples")
    parser.add_argument("--test_start_date", type=str, default="", help="inclusive start date for test samples")
    parser.add_argument("--test_end_date", type=str, default="", help="inclusive end date for test samples")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="chronological validation ratio within train/val range")
    return parser.parse_args()

def get_dataset(root_path, data_path, flag, input_len, output_len, freq, args):
    datasets = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute
    }
    if args.task_name == "multistock":
        return MultiStockSaveDataset(
            root_path=root_path,
            stock_pool_file=args.stock_pool_file,
            flag=flag,
            size=[input_len, 0, output_len],
            data_path=data_path,
            freq=freq,
            scale=False,
            target_horizon=args.target_horizon,
            start_date=args.start_date or None,
            end_date=args.end_date or None,
            trainval_start_date=args.trainval_start_date or None,
            trainval_end_date=args.trainval_end_date or None,
            test_start_date=args.test_start_date or None,
            test_end_date=args.test_end_date or None,
            val_ratio=args.val_ratio,
        )
    dataset_class = datasets.get(data_path, Dataset_Custom)
    return dataset_class(root_path=root_path, flag=flag, size=[input_len, 0, output_len], data_path=data_path, freq=freq)

def save_embeddings(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_set = get_dataset(args.root_path, args.data_path, 'train', args.input_len, args.output_len, args.freq, args)
    test_set = get_dataset(args.root_path, args.data_path, 'test', args.input_len, args.output_len, args.freq, args)
    val_set = get_dataset(args.root_path, args.data_path, 'val', args.input_len, args.output_len, args.freq, args)

    data_loader = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'test': DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers),
        'val': DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    }[args.divide]

    gen_prompt_emb = GenPromptEmb(
        device=device, # type: ignore
        input_len=args.input_len,
        data_path=args.data_path,
        model_name=args.model_name,
        d_model=args.d_model,
        layer=args.l_layers,
        divide=args.divide,
        freq=args.freq,
        task_name=args.task_name,
        stock_codes=getattr(train_set, "stock_codes", None),
        prompt_feature_index=getattr(train_set, "prompt_feature_index", 0),
    ).to(device)

    # 核心修复：确保路径逻辑与 data_loader_emb.py 一致，只取文件名
    data_name = os.path.basename(args.data_path).replace('.csv', '')
    save_path = f"./Embeddings/{data_name}/{args.divide}/"
    os.makedirs(save_path, exist_ok=True)

    emb_time_path = f"./Results/emb_logs/"
    os.makedirs(emb_time_path, exist_ok=True)

    count = 0
    for i, (x, y, x_mark, y_mark) in enumerate(data_loader):
        embeddings = gen_prompt_emb.generate_embeddings(x.to(device), x_mark.to(device))
        # embeddings shape: [Batch, Nodes, 768] or [Nodes, 768]
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        
        batch_size = embeddings.shape[0]
        for b in range(batch_size):
            file_path = f"{save_path}{count}.h5"
            save_data = embeddings[b].cpu().numpy()

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('embeddings', data=save_data)
            count += 1

        # # Save and visualize the first sample
        # if i >= 0:
        #     break
    
if __name__ == "__main__":
    args = parse_args()
    t1 = time.time()
    save_embeddings(args)
    t2 = time.time()
    print(f"Total time spent: {(t2 - t1)/60:.4f} minutes")
