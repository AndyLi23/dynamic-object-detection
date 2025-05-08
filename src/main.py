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


def copy_params_file(parent_dir, params, args):
    params_copy_path = os.path.join(parent_dir, f'{os.path.basename(params.output)}.yaml')
    with open(args.params, 'r') as src_file, open(params_copy_path, 'w') as dest_file:
        dest_file.write(src_file.read())
    print(f'saving params file to {params_copy_path}')

def preprocess_depth(depth, depth_params):
    depth = depth.astype(np.float32)

    if depth_params.bilateral_smooth_depth is not None:
        d, sigmaColor, sigmaSpace = depth_params.bilateral_smooth_depth
        depth = cv.bilateralFilter(depth, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    depth = depth / depth_params.depth_scale

    if depth_params.max_depth is not None: depth[depth > depth_params.max_depth] = 0
    return depth

def compute_relative_poses(poses):
    pose0 = poses[:-1]                                                                                  # (N, 4, 4)
    pose1_inv = torch.stack([torch.linalg.inv(pose) for pose in poses[1:]], dim=0)                      # (N, 4, 4)

    # T_1_0 = T_w_1^-1 @ T_w_0: transform from frame 0 to frame 1
    T_1_0 = pose1_inv @ pose0                                                                           # (N, 4, 4)      

    return T_1_0

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
    imgs = np.stack([img_data.img(t) for t in times], axis=0)
    depth_imgs = np.stack([preprocess_depth(depth_data.img(t), params.depth_data_params) for t in times], axis=0)
    cam_poses = np.stack([cam_pose_data.pose(t) for t in times], axis=0)
    T_1_0 = compute_relative_poses(cam_poses) # torch tensor

    print(f'running algorithm on {N_frames} frames from t={times[0]} to t={times[-1]}')

    parent_dir = os.path.dirname(params.output) if os.path.dirname(params.output) else '.'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    effective_fps = params.original_fps / params.skip_frames

    viz = OpticalFlowVisualizer(params.viz_params, f'{params.output}.avi', effective_fps)
    gof_flow = GeometricOpticalFlow(depth_data.camera_params, device=params.device)
    tracker = DynamicObjectTracker(params.tracking_params, effective_fps)

    for index in tqdm(range(0, N_frames - 1, params.batch_size)):

        torch.cuda.empty_cache()
        gc.collect()

        batch_end = min(index + params.batch_size, N_frames - 1)

        # print(f'processing frames {index} to {batch_end}')

        batch_imgs = imgs[index:batch_end]
        batch_next_imgs = imgs[index + 1:batch_end + 1]
        batch_depth_imgs = depth_imgs[index:batch_end + 1]
        batch_T_1_0 = T_1_0[index:batch_end]

        # print('computing raft optical flow...')
        raft_flows = raft.run_raft_batch(batch_imgs, batch_next_imgs) # (B H W 2)

        # print('computing optical flow residual...')
        residual = gof_flow.compute_flow(raft_flows, batch_depth_imgs, batch_T_1_0, use_3d=params.use_3d) # (B H W)

        # print('running dynamic object tracker...')
        dynamic_masks, orig_dynamic_masks = tracker.run_tracker(
            residual, 
            batch_depth_imgs[:-1],
            batch_imgs,
            T_1_0,
            draw_objects=params.viz_params.viz_dynamic_object_masks,
        )

        # # print('writing video frames...')
        viz.write_batch_frames(
            batch_imgs,
            batch_depth_imgs,
            dynamic_masks,
            orig_dynamic_masks,
            raft_flows,
            residual,
        )


    viz.end()
    copy_params_file(parent_dir, params, args)





    # raft_flows = []

    # print('computing raft optical flow...')

    # for index in tqdm(range(0, N_frames - 1, params.raft_params.batch_size)):
    #     batch_end = min(index + params.raft_params.batch_size, N_frames - 1)
    #     batch_imgs = imgs[index:batch_end]
    #     batch_next_imgs = imgs[index + 1:batch_end + 1]

    #     raft_flows.append(raft.run_raft(batch_imgs, batch_next_imgs))

    # gof = GeometricOpticalFlow(params.depth_data_params, depth_data.camera_params, device=params.device)

    # gof_flows = []

    # print('computing geometric optical flow...')

    # for index in tqdm(range(0, N_frames - 1, params.batch_size)):
    #     batch_end = min(index + params.batch_size, N_frames - 1)
    #     batch_depth_imgs = depth_imgs[index:batch_end]
    #     batch_cam_poses = cam_poses[index:batch_end + 1]

    #     gof_flows.append(gof.compute_optical_flow_batch(batch_depth_imgs, batch_cam_poses))

    # raft_flows = np.concatenate(raft_flows, axis=0)
    # gof_flows = np.concatenate(gof_flows, axis=0)

    # print('computing optical flow difference...')

    # del depth_imgs
    # del img_data
    # del depth_data
    # torch.cuda.empty_cache()
    # gc.collect()

    # flow_differences = []
    # for index in tqdm(range(0, len(raft_flows), params.batch_size)):
    #     batch_end = min(index + params.batch_size, N_frames - 1)
    #     batch_raft_flows = raft_flows[index:batch_end]
    #     batch_gof_flows = gof_flows[index:batch_end]
    #     flow_differences.append(gof.compute_batched_flow_difference_torch(batch_raft_flows, batch_gof_flows))
        
    # magnitude_diff_batch, norm_magnitude_diff_batch, angle_diff_batch = zip(*flow_differences)

    # magnitude_diff_batch = np.concatenate(magnitude_diff_batch, axis=0)
    # norm_magnitude_diff_batch = np.concatenate(norm_magnitude_diff_batch, axis=0)
    # angle_diff_batch = np.concatenate(angle_diff_batch, axis=0)


    # parent_dir = os.path.dirname(params.output) if os.path.dirname(params.output) else '.'
    # if not os.path.exists(parent_dir):
    #     os.makedirs(parent_dir)

    # viz_optical_flow_diff_batch(N=N_frames - 1, 
    #                             geometric_flow_batch=gof_flows, 
    #                             raft_flow_batch=raft_flows, 
    #                             image_batch=imgs[:-1], 
    #                             magnitude_diff_batch=magnitude_diff_batch, 
    #                             norm_magnitude_diff_batch=norm_magnitude_diff_batch, 
    #                             angle_diff_batch=angle_diff_batch, 
    #                             fps=params.original_fps / params.skip_frames,
    #                             output=f'{params.output}.avi')
    

    # copy params
    # copy_params_file(parent_dir, params, args)

