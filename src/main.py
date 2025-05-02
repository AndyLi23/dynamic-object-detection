import argparse
from params import Params
from raft_wrapper import RaftWrapper
from viz import viz_optical_flow_diff_batch
from flow import GeometricOpticalFlow
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--params', type=str, default='', help='Path to the parameters file')

    args = parser.parse_args()

    params = Params.from_yaml(args.params)

    print('loading raft...')
    raft = RaftWrapper(params.raft_params)
    print(f'raft loaded with model {params.raft_params.model}')

    print('loading data...')
    cam_pose_data = params.load_camera_pose_data()
    img_data = params.load_img_data()
    depth_data = params.load_depth_data()
    print('pose, img, depth data loaded')

    times = img_data.times
    N_frames = len(times)
    imgs = [img_data.img(t) for t in times]
    depth_imgs = [depth_data.img(t) for t in times]
    cam_poses = [cam_pose_data.pose(t) for t in times]

    skip = params.skip_frames

    imgs = imgs[::skip]
    depth_imgs = depth_imgs[::skip]
    cam_poses = cam_poses[::skip]
    times = times[::skip]
    N_frames = len(times)

    print(f'running algorithm on {N_frames} frames from t={times[0]} to t={times[-1]}')

    raft_flows = []

    print('computing raft optical flow...')

    for index in tqdm(range(0, N_frames - 1, params.raft_params.batch_size)):
        batch_end = min(index + params.raft_params.batch_size, N_frames - 1)
        batch_imgs = imgs[index:batch_end]
        batch_next_imgs = imgs[index + 1:batch_end + 1]

        raft_flows.append(raft.run_raft(batch_imgs, batch_next_imgs))

    gof = GeometricOpticalFlow(params.depth_data_params, depth_data.camera_params, device=params.device)

    gof_flows = []

    print('computing geometric optical flow...')

    for index in tqdm(range(0, N_frames - 1, params.batch_size)):
        batch_end = min(index + params.batch_size, N_frames - 1)
        batch_depth_imgs = depth_imgs[index:batch_end]
        batch_cam_poses = cam_poses[index:batch_end + 1]

        gof_flows.append(gof.compute_optical_flow_batch(batch_depth_imgs, batch_cam_poses))

    raft_flows = np.concatenate(raft_flows, axis=0)
    gof_flows = np.concatenate(gof_flows, axis=0)

    print('computing optical flow difference...')

    magnitude_diff_batch, norm_magnitude_diff_batch, angle_diff_batch = gof.compute_batched_flow_difference_torch(raft_flows, gof_flows)

    viz_optical_flow_diff_batch(N=N_frames - 1, 
                                geometric_flow_batch=gof_flows, 
                                raft_flow_batch=raft_flows, 
                                image_batch=imgs[:-1], 
                                magnitude_diff_batch=magnitude_diff_batch, 
                                norm_magnitude_diff_batch=norm_magnitude_diff_batch, 
                                angle_diff_batch=angle_diff_batch, 
                                fps=params.original_fps / params.skip_frames,
                                output=f'{params.output}.avi')
