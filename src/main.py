import argparse
from params import Params
from raft_wrapper import RaftWrapper
from viz import viz_optical_flow

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

    out = raft.run_raft(imgs[0:1], imgs[1:2]) # batched
    
    viz_optical_flow(out[0])
