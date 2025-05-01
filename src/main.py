import argparse
from params import Params
from raft_wrapper import RaftWrapper
from viz import viz_optical_flow
from flow import GeometricOpticalFlow
import numpy as np

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

    print(f'running algorithm on {N_frames} frames from t={times[0]} to t={times[-1]}')

    raft_flow = raft.run_raft(imgs[0:params.raft_params.batch_size], imgs[1:params.raft_params.batch_size + 1]) # batched
    
    viz_optical_flow(raft_flow[0])


    gof = GeometricOpticalFlow(params.depth_data_params, depth_data.camera_params, device=params.device)

    flow = gof.compute_optical_flow_batch(depth_imgs[0:params.batch_size], cam_poses[0:params.batch_size + 1]) # batched

    viz_optical_flow(flow[0])

    nobatch_flow = gof.compute_optical_flow(depth_imgs[0], *cam_poses[0:2]) # no batch

    viz_optical_flow(nobatch_flow)
    