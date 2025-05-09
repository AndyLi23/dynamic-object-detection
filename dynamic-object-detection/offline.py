import argparse
from params import Params
from raft_wrapper import RaftWrapper
from viz import OpticalFlowVisualizer
from flow import GeometricOpticalFlow
from tracker import DynamicObjectTracker
import numpy as np
from tqdm import tqdm
import os
import torch
import gc
import cv2 as cv
from dod_util import copy_params_file, preprocess_depth, compute_relative_poses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--params', '-p', type=str, default='', help='Path to the parameters file')

    args = parser.parse_args()

    params = Params.from_yaml(args.params)

    print('loading raft...')
    raft = RaftWrapper(params.raft_params, params.device)
    print(f'raft loaded with model {params.raft_params.model}')

    print('loading data...')
    cam_pose_data = params.load_camera_pose_data()
    img_data = params.load_img_data()
    depth_data = params.load_depth_data()
    print('pose, img, depth data loaded')

    times = img_data.times[::params.skip_frames]
    N_frames = len(times)
    imgs = np.stack([img_data.img(t) for t in times], axis=0) # (N, H, W, 3)
    depth_imgs = np.stack([preprocess_depth(depth_data.img(t), params.depth_data_params) for t in times], axis=0) # (N, H, W)
    cam_poses = np.stack([cam_pose_data.pose(t) for t in times], axis=0)
    T_1_0 = compute_relative_poses(torch.tensor(cam_poses, dtype=torch.float32, device=params.device)) # (N-1, 4, 4)

    print(f'running algorithm on {N_frames} frames from t={times[0]} to t={times[-1]}')

    parent_dir = os.path.dirname(params.output) if os.path.dirname(params.output) else '.'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    effective_fps = params.original_fps / params.skip_frames

    viz = OpticalFlowVisualizer(params.viz_params, f'{params.output}.avi', effective_fps)
    gof_flow = GeometricOpticalFlow(depth_data.camera_params, device=params.device)
    tracker = DynamicObjectTracker(params.tracking_params, depth_data.camera_params, effective_fps)

    for index in tqdm(range(0, N_frames - 1, params.batch_size)):

        torch.cuda.empty_cache()
        gc.collect()

        batch_end = min(index + params.batch_size, N_frames - 1)

        # print(f'processing frames {index} to {batch_end}')

        batch_imgs_np = imgs[index:batch_end+1]
        batch_imgs = torch.tensor(batch_imgs_np, dtype=torch.float32, device=params.device) # (B+1 H W 3)
        batch_img0s = batch_imgs[:-1] # (B H W 3)
        batch_img1s = batch_imgs[1:] # (B H W 3)
        batch_depth_imgs_np = depth_imgs[index:batch_end + 1]
        batch_depth_imgs = torch.tensor(batch_depth_imgs_np, dtype=torch.float32, device=params.device) # (B+1 H W)
        batch_T_1_0 = T_1_0[index:batch_end]

        # print('computing raft optical flow...')
        raft_flows = raft.run_raft_batch(batch_img0s, batch_img1s) # (B H W 2)

        # print('computing optical flow residual...')
        residual, coords_3d, raft_coords_3d_1= gof_flow.compute_flow(raft_flows, batch_depth_imgs, batch_T_1_0, use_3d=params.use_3d) # (B H W), (B H W 3), (B H W 3)

        # convert to numpy for tracking and visualization
        raft_flows = raft_flows.cpu().numpy()
        residual = residual.cpu().numpy()
        coords_3d = coords_3d.cpu().numpy()
        raft_coords_3d_1 = raft_coords_3d_1.cpu().numpy()

        # print('running dynamic object tracker...')
        dynamic_masks, orig_dynamic_masks = tracker.run_tracker(
            residual, 
            batch_imgs_np[:-1],
            batch_depth_imgs_np[:-1],
            coords_3d,
            raft_coords_3d_1,
            times=times[index:batch_end],
            draw_objects=params.viz_params.viz_dynamic_object_masks,
        )

        # # print('writing video frames...')
        viz.write_batch_frames(
            batch_imgs_np[:-1],
            batch_depth_imgs_np[:-1],
            dynamic_masks,
            orig_dynamic_masks,
            raft_flows,
            residual,
        )

    viz.end()
    copy_params_file(parent_dir, params, args)
    tracker.save_all_objects_to_pickle(os.path.join(parent_dir, f'{os.path.basename(params.output)}_objects.pkl'))
