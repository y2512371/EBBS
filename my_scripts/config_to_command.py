# based on https://github.com/NLP-Playground/LaSS/blob/master/toolbox/train.py

import math
import os
import re
import yaml
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", type=str)
    parser.add_argument("--multi-machine", action='store_true', default=False)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--dual-model", action='store_true', default=False)


    return parser.parse_known_args()

def pop_from_config(config, key):
    if key in config:
        return config.pop(key)
    key = key.replace("_", "-")
    if key in config:
        return config.pop(key)
    key = key.replace("-", "_")
    if key in config:
        return config.pop(key)
    return None

def get_train_script_from_config(args):
    multi_machine_config = ''
    entry_script = []
    if args.multi_machine:
        multi_card_prefix =["NCCL_IB_DISABLE=0"]
        multi_card_prefix.append("NCCL_IB_GID_INDEX=3")
        multi_card_prefix.append("NCCL_SOCKET_IFNAME=eth0")
        multi_card_prefix.append("OMP_NUM_THREADS=1")
        multi_machine_config = '-m torch.distributed.launch --nproc_per_node=$ARNOLD_WORKER_GPU ' \
                            '--nnodes=$ARNOLD_NUM --node_rank=$ARNOLD_ID --master_addr=$ARNOLD_WORKER_0_HOST ' \
                            '--master_port=$ARNOLD_WORKER_0_PORT'
        entry_script.extend(multi_card_prefix)
        entry_script.append('python3')
        entry_script.append(multi_machine_config)
    elif args.debug:
        entry_script.append("CUDA_VISIBLE_DEVICES=0")
        entry_script.append("python3")
        entry_script.append("-m debugpy --listen 5678")
    else:
        entry_script.append('python3')

    config_paths = args.config
    config = {}
    for config_path in config_paths:
        with open(config_path, "r") as f:
            config = {**yaml.safe_load(f), **config}

    fs_args = []
    train_model = 'default'
    for k, v in config.items():
        
        if k == 'train_mode':
            train_model = v
            continue
        
        if k == "data_bin":
            fs_args.append(v)
            continue
        
        k = k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                k = f"--{k}"
            else:
                k = ""
        else:
            k = f"--{k} {v}"
        fs_args.append(k)
    fs_args = " ".join(fs_args)

    
    if train_model == 'default':
        return f"{' '.join(entry_script)} train.py {fs_args} --user-dir fs_plugins "
    elif train_model == 'dual_model':
        return f"{' '.join(entry_script)} fairseq_cli/train_dual.py {fs_args} --user-dir fs_plugins "       


if __name__ == "__main__":
    args, unknown = get_args()
    # print("Extra arguments:", ' '.join(unknown))

    train_script = get_train_script_from_config(args)

    train_script += ' '.join(unknown)

    print(train_script)

    # os.system(train_script)


