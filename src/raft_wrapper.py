import sys
import torch
from params import RaftParams
import torchvision.transforms as T
import numpy as np

class RaftWrapper:
    def __init__(self, raft_params: RaftParams, device):
        self.raft_params = raft_params       
        self.raft_params.expand_vars() 
        self.device = device
        self._load_raft()

        self.transforms = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),  # map to [-1, 1]
            ]
        )

    def _load_raft(self):
        sys.path.append(self.raft_params.path)
        from raft import RAFT

        self.model = torch.nn.DataParallel(RAFT(self.raft_params.raft_args)).to(self.device)
        self.model.load_state_dict(torch.load(self.raft_params.model))
        self.model.eval()

    def preprocess(self, batch):
        batch = np.array(batch)
        batch = torch.Tensor(batch).permute(0, 3, 1, 2).to(self.device)  # HWC to CHW
        return self.transforms(batch)

    @torch.no_grad()
    def run_raft(self, img1_batch, img2_batch):
        _, flow_up = self.model(self.preprocess(img1_batch), self.preprocess(img2_batch), iters=self.raft_params.iters, test_mode=True)
        flow_up = flow_up.permute(0, 2, 3, 1).cpu().numpy()  # CHW to HWC
        return flow_up