import numpy as np
import cv2 as cv
from params import DepthDataParams
from robotdatapy.camera import CameraParams
import torch

class GeometricOpticalFlow:
    def __init__(self, depth_params: DepthDataParams, depth_camera_params: CameraParams, device: str = 'cuda'):
        self.depth_params = depth_params

        self.K = depth_camera_params.K
        self.D = depth_camera_params.D
        self.H = depth_camera_params.height
        self.W = depth_camera_params.width

        self.device = device

    def compute_optical_flow(self, dimg, pose0, pose1):

        # scale depth data (mm -> m)
        dimg = dimg / self.depth_params.depth_scale                                             # (H, W)

        # compute mask for valid depth values
        valid_mask = ((dimg > 0) & (dimg <= self.depth_params.max_depth)) \
                      if self.depth_params.max_depth is not None else dimg == 0                 # (H, W)
        depths = dimg[valid_mask].reshape(-1, 1)                                                # (N_valid, 1)       

        # T_w_1^-1 T_w_0: transform from frame 0 to frame 1
        T_1_0 = np.linalg.inv(pose1) @ pose0 

        # compute normalized camera coordinates for each pixel
        x, y = np.meshgrid(np.arange(self.W), np.arange(self.H))
        pixel_coords_valid = np.stack((x, y), axis=-1, dtype=np.float32)[valid_mask]
        pixel_coords_flattened_valid = pixel_coords_valid.reshape(-1, 2)                        # (N_valid, 2)

        norm_cam_coords = cv.undistortPoints(                                                   # (N_valid, 2)                                      
            pixel_coords_flattened_valid, 
            cameraMatrix=self.K, distCoeffs=self.D
        ).reshape(-1, 2)

        # project normalized camera coordinates to 3D for each pixel
        norm_cam_coords = np.concatenate((norm_cam_coords, np.ones((*norm_cam_coords.shape[:-1], 1), dtype=np.float32)), axis=-1) # (N_valid, 3)
        coords_3d = norm_cam_coords * depths                                                    # (N_valid, 1) * # (N_valid, 1) ->  (N_valid, 3)   

        # compute 3d coordinates in subsequent frame
        coords_3d_h = np.concatenate((coords_3d, np.ones((*coords_3d.shape[:-1], 1))), axis=-1) # (N_valid, 4)
        coords_3d_h_1 = T_1_0 @ coords_3d_h.T                                                   # (4, 4) @ (4, N_valid) = (4, N_valid)
        coords_3d_1 = (coords_3d_h_1.T)[:, :3]                                                  # (4, N_valid) -> (N_valid, 4) ->   (N_valid, 3)

        # project 3d coordinates in subsequent frame to original image frame
        projected_pixel_coords_flattened_valid = cv.projectPoints(                              # (N_valid, 2)     
            coords_3d_1, 
            rvec=np.zeros(3), tvec=np.zeros(3), 
            cameraMatrix=self.K, distCoeffs=self.D
        )[0].reshape(-1, 2)

        # compute flow for each pixel in original frame
        flow = np.zeros((self.H, self.W, 2), dtype=np.float32)                                  # (H, W, 2) 
        flow[valid_mask] = projected_pixel_coords_flattened_valid - pixel_coords_flattened_valid                

        return flow


    @torch.no_grad()
    def compute_optical_flow_batch(self, dimg_batch, poses):

        # scale depth data (mm -> m)
        dimg_batch = torch.Tensor(np.array(dimg_batch)).to(self.device) / self.depth_params.depth_scale     # (N, H, W)
        poses = torch.Tensor(np.array(poses)).to(self.device)                                               # (N+1, 4, 4)

        N = len(dimg_batch)
        assert(N+1 == len(poses))


        # compute relative poses between frames
        pose0 = poses[:-1]                                                                                  # (N, 4, 4)
        pose1_inv = torch.stack([torch.linalg.inv(pose) for pose in poses[1:]], dim=0)                      # (N, 4, 4)

        # T_1_0 = T_w_1^-1 @ T_w_0: transform from frame 0 to frame 1
        T_1_0 = pose1_inv @ pose0                                                                           # (N, 4, 4)     


        # compute masks for valid depth values
        invalid_mask = ~((dimg_batch > 0) & (dimg_batch <= self.depth_params.max_depth)) \
                         if self.depth_params.max_depth is not None else dimg_batch == 0                    # (N, H, W)
        invalid_mask = invalid_mask.cpu().numpy()

        flattened_depths = dimg_batch.view(N, -1, 1)                                                        # (N, H*W, 1)


        # compute normalized camera coordinates for each pixel (opencv requires numpy)
        x, y = np.meshgrid(np.arange(self.W), np.arange(self.H))
        pixel_coords = np.stack((x, y), axis=-1, dtype=np.float32)                                          # (H, W, 2)
        pixel_coords_flattened = pixel_coords.reshape(-1, 2)                                                # (H*W,  2)

        norm_cam_coords = cv.undistortPoints(                                                               # (H*W,  2)
            pixel_coords_flattened, 
            cameraMatrix=self.K, distCoeffs=self.D
        ).reshape(-1, 2)

        norm_cam_coords = torch.Tensor(norm_cam_coords).to(self.device)                                     # (H*W,  2)

        # project normalized camera coordinates to 3D for each pixel, for each frame
        norm_cam_coords = torch.concatenate((norm_cam_coords, 
                                             torch.ones((*norm_cam_coords.shape[:-1], 1), dtype=torch.float32, device=self.device)
                                            ), dim=-1).view(1, -1, 3)                                       # (1, H*W, 3)
        
        coords_3d = norm_cam_coords * flattened_depths                                                      # (1, H*W, 3) * (N, H*W, 1) ->  (N, H*W, 3)

        # compute 3d coordinates in subsequent frame
        coords_3d_h = torch.concatenate((coords_3d, 
            torch.ones((*coords_3d.shape[:-1], 1), dtype=torch.float32, device=self.device)
        ), dim=-1)                                                                                          # (N, H*W, 4)         
        coords_3d_h_1 = T_1_0 @ coords_3d_h.permute(0, 2, 1)                                                # (N, 4, 4) @ (N, 4, H*W) =     (N, 4, H*W)
        coords_3d_1 = (coords_3d_h_1.permute(0, 2, 1))[..., :3]                                             # (N, 4, H*W) -> (N, H*W, 4) -> (N, H*W, 3)   
        coords_3d_1 = coords_3d_1.cpu().numpy()                                                             # (N, H*W, 3)

        # project 3d coordinates in subsequent frame to original image frame
        projected_pixel_coords_flattened = np.stack([cv.projectPoints(                                      # (N, H*W, 2)
                coords_3d_1_frame, 
                rvec=np.zeros(3), tvec=np.zeros(3), 
                cameraMatrix=self.K, 
                distCoeffs=self.D,
            )[0].reshape(-1, 2) for coords_3d_1_frame in coords_3d_1], axis=0)

        # compute flow for each pixel in original frames
        flattened_flow = projected_pixel_coords_flattened - pixel_coords_flattened[np.newaxis, :]           # (N, H*W, 2) - (1, H*W, 2) =   (N, H*W, 2)        

        flow = flattened_flow.reshape(N, self.H, self.W, 2)                                                 # (N, H, W, 2) 

        flow[invalid_mask] = np.nan

        return flow



