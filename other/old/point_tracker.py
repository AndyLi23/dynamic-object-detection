import torch

class MultiCoTrackerWrapper:
    def __init__(self, params):
        self.params = params
        self.device = params.device
        self.trackers = {} # object id : tracker
        self.points = {} # object id : points
        self.object_last_frame = {} # object id : last frame index

    def add_tracker(self, cur_frame_ind, cur_frame, mask):

        points = None #TODO

        object_id = len(self.trackers)
        self.trackers[object_id] = self.create_tracker()
        self.points[object_id] = points
        self.object_last_frame[object_id] = cur_frame_ind

        queries = points[torch.newaxis]
        # start from t = effective frame 0
        queries = torch.cat([torch.zeros((queries.shape[:-1], 1), device=self.device), queries], dim=-1) # B N 3 in format (t, x, y)

        # permute is not alias safe, but video should stay unmodified
        self.trackers[object_id](
            video_chunk=cur_frame.permute(2, 0, 1)[torch.newaxis, torch.newaxis], # B T 3 H W
            is_first_step=True,
            queries=queries
        )

    def update_tracker(self, cur_frame_ind, images):
        for object_id, tracker in self.trackers.items():
            if self.object_last_frame[object_id] == cur_frame_ind: continue

            pred_tracks, pred_visibility = tracker(
                video_chunk=images.permute(0, 3, 1, 2)[torch.newaxis, self.object_last_frame[object_id]+1:cur_frame_ind+1] # B T 3 H W
            )

            self.points[object_id] = pred_tracks[0, -1] # B T N 2 -> N 2

    def objects(self):
        return self.points.items()

    def create_tracker(self):
        return torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(self.device)