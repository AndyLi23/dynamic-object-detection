import cv2 as cv
import numpy as np
import open3d as o3d

from dynamic_object_detection.dod_util import global_nearest_neighbor_dynamic_objects
import pickle

class DynamicObjectTracker:
    def __init__(self, params, depth_camera_params, effective_fps):
        self.params = params
        self.H = depth_camera_params.height
        self.W = depth_camera_params.width
        self.min_residual_threshold = params.min_vel_threshold / effective_fps
        self.residual_threshold_gain = params.vel_threshold_gain / effective_fps

        self.tracked_objects = {}
        self.all_objects = {}
        self._id = 0

    def run_tracker(self, residuals, imgs, depths, coords_3d, raft_coords_3d_1, times, cam_poses, draw_objects=False):
        """
        residuals: (N, H, W)
        imgs: (N, H, W, 3)
        depths: (N, H, W)
        coords_3d: (N, H*W, 3)
        raft_coords_3d_1: (N, H*W, 3)
        cam_poses: (N, 4, 4)
        times: (N,)
        """
        dynamic_mask, orig_dynamic_mask = self.compute_dynamic_mask_batch(residuals, depths)  # (N, H, W)

        for frame in range(len(imgs)):
            
            contours, _ = cv.findContours(dynamic_mask[frame].astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            new_objects, world_pointclouds = self.contours_to_objects(contours, coords_3d[frame], cam_poses[frame])

            associations = global_nearest_neighbor_dynamic_objects(
                tracked_objects=self.tracked_objects, 
                new_objects=new_objects, 
                cost_fn=DynamicObjectTrack.chamfer_distance, 
                max_cost=self.params.max_chamfer_distance,
            )

            to_remove_ids = [obj.id for obj in self.tracked_objects.values() if obj.id not in associations.values()]
            self.remove_dynamic_objects(to_remove_ids)

            for new_obj, wpcl in zip(new_objects, world_pointclouds):
                next_frame_points = raft_coords_3d_1[frame][new_obj.mask]
                if new_obj.id in associations:
                    to_update_id = associations[new_obj.id]
                else:
                    to_update_id = new_obj.id
                    self.tracked_objects[new_obj.id] = new_obj
                    self.all_objects[new_obj.id] = new_obj # track all over time

                self.tracked_objects[to_update_id].update(new_obj.contour, next_frame_points)
                self.tracked_objects[to_update_id].add_points_time(wpcl, times[frame])

            if draw_objects:
                for obj in self.tracked_objects.values():
                    imgs[frame] = cv.drawContours(imgs[frame], [obj.contour], -1, (255, 0, 0), 4)
                    imgs[frame] = cv.putText(imgs[frame], str(obj.id), tuple(obj.contour[0][0]), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        return dynamic_mask, orig_dynamic_mask
    
    def contours_to_objects(self, contours, coords_3d_frame, cam_pose):
        """
        contours: List
        coords_3d_frame: (H*W, 3)
        """
        objects = []
        world_pointclouds = []
        for contour in contours:
            if len(contour) < 3: continue
            mask = np.zeros((self.H, self.W), dtype=np.uint8)
            cv.fillPoly(mask, [contour], 1)
            mask = mask.reshape((-1)).astype(bool)
            points = coords_3d_frame[mask] 
            obj = DynamicObjectTrack(self._id, contour, mask, points)
            self._id += 1
            objects.append(obj)

            world_points = self.transform_points(cam_pose, points)
            world_pointclouds.append(world_points)

        return objects, world_pointclouds

    def remove_dynamic_objects(self, to_remove_ids):
        for id_ in to_remove_ids:
            del self.tracked_objects[id_]

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
    
    def transform_points(self, T_w_frame, points):
        """
        Transform points from camera frame to worlod frame
        T_w_frame: (4, 4)
        points: (N, 3)
        """
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        return (T_w_frame @ points_h.T)[:3].T
    
    def save_all_objects_to_pickle(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({obj.id : obj.to_dict() for obj in self.all_objects.values()}, f)
    

class DynamicObjectTrack:
    def __init__(self, id_, contour, mask, points):
        self.id = id_
        self.mask = mask
        self.o3d = o3d.geometry.PointCloud()
        self.points_list = []
        self.times = []
        self.update(contour, points)

    def update(self, contour, points):
        self.contour = contour
        self.points = points
        self.o3d.points = o3d.utility.Vector3dVector(points)

    def add_points_time(self, points, time):
        self.points_list.append(points)
        self.times.append(time)

    def to_dict(self):
        return {
            'id': self.id,
            'times': self.times,
            'points': self.points_list,
        }

    @classmethod
    def chamfer_distance(cls, obj1, obj2):
        # Compute Chamfer Distance between 3d points
        dist1 = np.mean(obj1.o3d.compute_point_cloud_distance(obj2.o3d))
        dist2 = np.mean(obj2.o3d.compute_point_cloud_distance(obj1.o3d))

        chamfer_distance = min(dist1, dist2)
        return chamfer_distance
