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

    def run_tracker(self, residuals, depth_batch, img_batch, T_1_0, draw_objects=False):
        dynamic_mask, orig_dynamic_mask = self.compute_dynamic_mask_batch(residuals, depth_batch)  # (N, H, W)

        for frame in range(len(img_batch)):
            
            # # TODO: process new dynamic objects
            contours, _ = cv.findContours(dynamic_mask[frame].astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # if len(contours) > 0:
                # img_batch[frame] = cv.drawContours(img_batch[frame], contours, -1, (255, 0, 0), 3)

            new_objects = self.contours_to_objects(contours, depth_batch[frame])

            

            if draw_objects:
                self.draw_dynamic_objects(img_batch[frame])

        return dynamic_mask, orig_dynamic_mask
    
    def contours_to_objects(self, contours, depth):
        objects = []
        for contour in contours:
            if len(contour) < 3: continue
            mask = np.zeros_like(depth, dtype=np.uint8)
            cv.fillPoly(mask, [contour], 1)
            points = depth[mask == 1]
            obj = DynamicObjectTrack(self._id, contour, points)
            self._id += 1
            objects.append(obj)

        return objects
    
    def draw_dynamic_objects(self, img):
        for obj in self.tracked_objects:
            # TODO: draw object on image
            pass

    def compute_dynamic_mask_batch(self, residuals, depths):

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
    

class DynamicObjectTrack:
    def __init__(self, id, contour, points):
        self.id = id
        self.update(contour, points)

    def update(self, contour, points):
        self.contour = contour
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
