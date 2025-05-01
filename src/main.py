import argparse
from params import Params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--params', type=str, default='', help='Path to the parameters file')

    args = parser.parse_args()

    params = Params.from_yaml(args.params)

    print('loading data...')
    cam_pose_data = params.load_camera_pose_data()
    img_data = params.load_img_data()
    depth_data = params.load_depth_data()
    print('pose, img, depth data loaded')

    times = img_data.times
    N_frames = len(times)
    imgs = img_data.imgs
    depth_imgs = [depth_data.img(t) for t in times]
    cam_poses = [cam_pose_data.pose(t) for t in times]

    print(f'running algorithm on {N_frames} frames from t={times[0]} to t={times[-1]}')
