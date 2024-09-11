import os
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import litdata as ld
import argparse


def read_data(fn: str) -> Dict:
    batch_data: Dict = dict(np.load(fn, allow_pickle=True))
    return batch_data


if __name__ == "__main__":
    """
    python mdt/datasets/litdata_scripts/litdata_optimize.py -i /home/geyuan/datasets/CALVIN/dataset/task_D_D
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--in_data", type=str,
                            help="path to original data")
    arg_parser.add_argument("-o", "--out_data", type=str,
                            default='/home/geyuan/datasets/CALVIN/dataset/litdata',
                            help="path to optimized output data")
    args = arg_parser.parse_args()

    ''' 1. Check data format '''
    args.in_data = os.path.abspath(args.in_data)
    args.out_data = os.path.abspath(args.out_data)
    root = os.path.dirname(args.in_data)
    task = os.path.basename(args.in_data)
    sub1_dir = os.listdir(args.in_data)
    for sub1 in sub1_dir:  # 'training', 'validation'
        print(f'[litdata_optimize] now_dir={sub1} Reading...')
        sub1_path = os.path.join(args.in_data, sub1)
        all_files = os.listdir(sub1_path)
        npz_files = [os.path.join(sub1_path, x) for x in all_files if x.endswith('.npz') and 'episode' in x]
        other_files = [x for x in all_files if 'episode' not in x]
        print(f'[litdata_optimize] now_dir={sub1}, following files or dirs are excluded: {other_files}')

        ''' create out_dir '''
        sub1_out = str(os.path.join(args.out_data, task, sub1))
        os.makedirs(sub1_out, exist_ok=True)

        ''' parse episode names (e.g episode_0360804.npz) '''
        print(f'[litdata_optimize] now_dir={sub1} Create dict ep_idx_to_dataset_order.npz in {sub1_out}')
        ep_indices = [int(os.path.basename(x).split('.')[0].split('_')[-1]) for x in npz_files]
        dict_ep_idx_to_dataset_order = {ep_indices[x]: x for x in range(len(ep_indices))}
        np.save(os.path.join(sub1_out, 'dict_ep_idx_to_dataset_order'),
                dict_ep_idx_to_dataset_order, allow_pickle=True)

        print(f'[litdata_optimize] now_dir={sub1} Optimizing... saved to {sub1_out}')
        # ld.optimize(
        #     fn=read_data,  # the function applied to each input
        #     inputs=npz_files,  # the inputs to the function (here it's a list of numbers)
        #     output_dir=sub1_out,  # optimized data is stored here
        #     num_workers=4,  # The number of workers on the same machine
        #     chunk_bytes="64MB"  # size of each chunk
        # )
