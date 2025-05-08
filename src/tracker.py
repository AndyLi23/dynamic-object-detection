import cv2 as cv
import numpy as np
import open3d as o3d

from gnn import global_nearest_neighbor

class DynamicObjectTracker:
    def __init__(self, params, effective_fps):
        self.params = params
        self.min_residual_threshold = params.min_vel_threshold / effective_fps
        self.residual_threshold_gain = params.vel_threshold_gain / effective_fps

        self.tracked_objects = []
        self._id = 0

    def run_batch_tracker(self, residuals, depth_batch, img_batch, poses, draw_objects=False):
        dynamic_mask, orig_dynamic_mask = self.compute_dynamic_mask(residuals, depth_batch)  # (N, H, W)

        for frame in range(len(img_batch)):
            
            # # TODO: process new dynamic objects
            contours, _ = cv.findContours(dynamic_mask[frame].astype(np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                img_batch[frame] = cv.drawContours(img_batch[frame], contours, -1, (255, 0, 0), 3)

            if draw_objects:
                self.draw_dynamic_objects(img_batch[frame])

        return dynamic_mask, orig_dynamic_mask
    
    def draw_dynamic_objects(self, img):
        for obj in self.tracked_objects:
            # TODO: draw object on image
            pass

    def compute_dynamic_mask(self, residuals, depths):

        # Pre-processing

        if self.params.gaussian_smoothing:
            smoothed_residuals = np.zeros_like(residuals)
            for i in range(residuals.shape[0]):
                smoothed_residuals[i] = cv.GaussianBlur(residuals[i], (self.params.gaussian_kernel_size, self.params.gaussian_kernel_size), 0)
            residuals = smoothed_residuals

        threshold = self.min_residual_threshold + self.residual_threshold_gain * depths  # (N, H, W)
        mask = ((residuals > threshold) & ~np.isnan(residuals)).astype(np.uint8)  # (N, H, W)

        # Post-processing

        orig_mask = mask.copy()

        for method, kernel_size in self.params.post_processing:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            for i in range(mask.shape[0]):
                if method == 'dilate':
                    mask[i] = cv.dilate(mask[i], kernel, iterations=1)
                elif method == 'erode':
                    mask[i] = cv.erode(mask[i], kernel, iterations=1)
                elif method == 'open':
                    mask[i] = cv.morphologyEx(mask[i], cv.MORPH_OPEN, kernel)
                elif method == 'close':
                    mask[i] = cv.morphologyEx(mask[i], cv.MORPH_CLOSE, kernel)

        return mask, orig_mask


    # Old code with geometric flow
    @DeprecationWarning
    def compute_batched_dynamic_pixels(self, raft_batch, geometric_batch, ret_mag_residual=False):
        flow_diff = raft_batch - geometric_batch                                                            # (N, H, W, 2)
        valid_mask = ~np.isnan(flow_diff[..., 0]) & ~np.isnan(flow_diff[..., 1])                            # (N, H, W)
        magnitude_diff = np.zeros((*flow_diff.shape[:-1], 1), dtype=np.float32)                             # (N, H, W, 1) x 2
        magnitude_diff[valid_mask], _ = cv.cartToPolar(flow_diff[..., 0][valid_mask], flow_diff[..., 1][valid_mask])
        geometric_batch_magnitude = np.linalg.norm(geometric_batch, axis=-1, keepdims=True)                 # (N, H, W, 1)

        dynamic_mask = (magnitude_diff > self.params.absolute_threshold)[..., 0] & \
                       (magnitude_diff > self.params.relative_threshold * geometric_batch_magnitude)[..., 0] & \
                       valid_mask                                                                           # (N, H, W)

        if ret_mag_residual: magnitude_diff[~valid_mask] = np.nan

        return magnitude_diff[..., 0] if ret_mag_residual else None, dynamic_mask                           # (N, H, W) x 2
    

class DynamicObjectTrack:
    def __init__(self, id, mask, points):
        self.id = id
        self.update(mask, points)

    def update(self, mask, points):
        self.mask = mask
        self.points = points
        self.o3d = o3d.geometry.PointCloud()
        self.o3d.points = o3d.utility.Vector3dVector(points)

    @classmethod
    def similarity(cls, obj1, obj2):
        # Compute Chamfer Distance between 3d points
        dist1 = np.mean(obj1.o3d.compute_point_cloud_distance(obj2.o3d))
        dist2 = np.mean(obj2.o3d.compute_point_cloud_distance(obj1.o3d))

        chamfer_distance = dist1 + dist2
        return chamfer_distance
