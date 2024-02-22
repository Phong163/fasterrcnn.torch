import torch
import torch.nn as nn
import torch.nn.functional as F

class RoiPoolingConv(nn.Module):
    def __init__(self, pool_size, num_rois,spatial_scale):
        super(RoiPoolingConv, self).__init__()
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.spatial_scale = spatial_scale
    def forward(self, x, rois):
        assert len(x) == 3

        img = x[0]
        rois = rois[0]

        outputs = []

        for roi_idx in range(self.num_rois):
            x, y, w, h = map(int, rois[roi_idx, :])

            # Resize the RoI to pooling size (pool_size x pool_size)
            rs = F.interpolate(img[:, :, y:y+h, x:x+w], size=(self.pool_size, self.pool_size), mode='nearest')
            outputs.append(rs)

        final_output = torch.cat(outputs, dim=0)
        final_output = final_output.view(1, self.num_rois, self.pool_size, self.pool_size, -1)
        final_output = final_output.permute(0, 1, 2, 3, 4)

        return final_output
