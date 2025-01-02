import yaml
import os
import numpy as np
from copy import deepcopy

if __name__ == "__main__":
    template_path = "~/LonerSLAM/cfg/kitti/kitti_00.yaml"
    template_path = os.path.expanduser(template_path)
    with open(template_path, 'r') as f:
        template = yaml.load(f, Loader=yaml.FullLoader)

    # Set the path to the KITTI dataset
    for seq in range(1, 11):
        seq = str(seq).zfill(2)
        template_copy = deepcopy(template)
        template_copy['dataset'] = template_copy['dataset'].replace('00', seq)
        template_copy['experiment_name'] = template_copy['experiment_name'].replace('00', seq)
        template_copy['groundtruth_traj'] = template_copy['groundtruth_traj'].replace('00', seq)
        template_path_new = template_path.replace('00', seq)
        with open(template_path_new, 'w') as f:
            yaml.dump(template_copy, f)