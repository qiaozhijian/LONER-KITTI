import os
import sys
import psutil
import numpy as np
import torch

if __name__ == "__main__":
    output_dir = "~/LonerSLAM/outputs"
    all_log_dirs = os.listdir(os.path.expanduser(output_dir))
    print(all_log_dirs)
    for log_dir in all_log_dirs:
        if 'kitti' not in log_dir:
            continue
        seq = log_dir.split('_')[1]
        print(f"Processing sequence {seq}")

        log_dir = os.path.join(os.path.expanduser(output_dir), log_dir)
        print(log_dir)

        # python3 plot_poses.py ../outputs/<output_folder>
        if not os.path.exists(f"{log_dir}/trajectory"):
            print(f"Trajectory folder not found in {log_dir}")
            # raise ValueError(f"Trajectory folder not found in {log_dir}")
            continue

        cmd = f"pwd; python3 analysis/plot_poses.py {log_dir}"
        os.system(cmd)

        checkpoint_dir = os.path.join(log_dir, 'checkpoints', 'final.tar')
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint not found in {checkpoint_dir}")
            continue

        checkpoint = torch.load(checkpoint_dir)
        network_state_dict = checkpoint['network_state_dict']
        occ_model_state_dict = checkpoint.get('occ_model_state_dict', None)
        # compute model size in MB
        model_size = sum(p.numel() for p in network_state_dict.values()) / 1e6
        print(f"Model size: {model_size:.2f} MB")
        occ_model_size = sum(p.numel() for p in occ_model_state_dict.values()) / 1e6 if occ_model_state_dict else 0
        print(f"Occ Model size: {occ_model_size:.2f} MB")
        total_size = model_size + occ_model_size
        # write to a file
        with open(f"{log_dir}/model_size.txt", "w") as f:
            f.write(f"Model size: {model_size:.2f} MB\n")
            f.write(f"Occ Model size: {occ_model_size:.2f} MB\n")
            f.write(f"Total size: {total_size:.2f} MB\n")