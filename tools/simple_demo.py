import argparse
from logging import Logger
from pathlib import Path
from typing import Any

import _init_path
import numpy as np
import torch
from numpy._typing._array_like import NDArray

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# ./.venv/bin/python ./tools/simple_demo.py /vol/enableato/our-datasets/250909_vorm_CITEC/Dataset_2025_09_09-14_01_48_Genie_M1P/data/M1P/000050_00027.26152/point_cloud_xyzi.npy --cfg_file ./tools/cfgs/nuscenes_models/transfusion_lidar.yaml --ckpt ./cbgs_transfusion_lidar.pth


def rotate_points(points, axis, angle_deg):
    """
    Rotates points around X, Y, or Z axis by a specific angle.
    
    Args:
        points (np.ndarray): Shape (N, 3+) array.
        axis (str): 'x', 'y', or 'z'.
        angle_deg (float): Angle in degrees. Positive = Counter-Clockwise.
    
    Returns:
        np.ndarray: Rotated points (same shape as input).
    """
    # Convert to radians
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    # Define Rotation Matrices
    if axis.lower() == 'x':
        R = np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis.lower() == 'y':
        R = np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis.lower() == 'z':
        R = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unknown axis: {axis}")

    # Apply rotation only to x, y, z (first 3 columns)
    # Using matrix multiplication: (N, 3) @ (3, 3)
    # We transpose R because points are row vectors
    points[:, :3] = points[:, :3] @ R.T
    
    return points

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except ImportError:
    print("Open3D not found, falling back to mayavi if available.")
    try:
        import mayavi.mlab as mlab
        from visual_utils import visualize_utils as V
        OPEN3D_FLAG = False
    except ImportError:
        print("Neither Open3D nor Mayavi found. Cannot visualize.")
        exit(1)

class SimpleDemoDataset(DatasetTemplate):
    def __init__(self,
                 dataset_cfg,
                 class_names,
                 training: bool = True,
                 root_path: Path = None,
                 logger: Logger = None,
                 ext: str = '.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            root_path=root_path,
            logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        if self.root_path.is_file():
            self.sample_file_list = [self.root_path]
        else:
            self.sample_file_list = []

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        file_path = self.sample_file_list[index]
        if str(file_path).endswith('.npy'):
            points = np.load(file_path)
        elif str(file_path).endswith('.bin'):
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        else:
            raise NotImplementedError(f"Unsupported file extension: {file_path}")

        if points.shape[-1] == 4:
            points = np.concatenate([points, np.zeros((points.shape[0], 1), dtype=points.dtype)], axis=1)

        mask = ~np.isnan(points).any(axis=1)
        points = points[mask]

        print("Swapping axes to [X, Y, Z]...")
        points[:, [0, 1, 2]] = points[:, [2, 0, 1]]

        # Normalize points to fit into the model's range
        #z_median = np.median(points[:, 2])
        #points[:, 2] -= (z_median + 1.75)
        #mask_z = (points[:, 2] > -5.0) & (points[:, 2] < 3.0)
        #points = points[mask_z]

        #points = rotate_points(points, 'x', 90)

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def main():
    parser = argparse.ArgumentParser(description='Simple OpenPCDet Demo for single .npy file')
    parser.add_argument('npy_file', type=str, help='Path to the .npy point cloud file')
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to the config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file')

    args = parser.parse_args()

    logger: Logger = common_utils.create_logger()

    cfg_from_yaml_file(args.cfg_file, cfg)

    npy_path = Path(args.npy_file).resolve()
    if not npy_path.exists():
        logger.error(f'File not found: {npy_path}')
        return

    dataset = SimpleDemoDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=npy_path,
        ext='.npy',
        logger=logger
    )

    logger.info(f'Loading model from {args.cfg_file}...')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for _, data_dict in enumerate(dataset):
            logger.info(f'Visualizing: {args.npy_file}')
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            if OPEN3D_FLAG:
                V.draw_scenes(
                    points=data_dict['points'][:, 1:],
                    ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'],
                    ref_labels=pred_dicts[0]['pred_labels'],
                    class_names=cfg.CLASS_NAMES
                )
            else:
                # Mayavi visualization might differ slightly in API
                V.draw_scenes(
                    points=data_dict['points'][:, 1:],
                    ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'],
                    ref_labels=pred_dicts[0]['pred_labels'],
                    class_names=cfg.CLASS_NAMES
                )
                mlab.show(stop=True)

    logger.info('Demo finished.')

if __name__ == '__main__':
    main()
