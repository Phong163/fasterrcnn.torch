import torch
import torch.nn.functional as F
from torch.autograd import Function

class RoIPoolFunction(Function):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.feature_size = None

    def forward(self, features, rois): 
        self.feature_size = features.size()           
        batch_size, num_channels, data_height, data_width = self.feature_size

        # Convert RoIs from (x1, y1, x2, y2) to (y1, x1, y2, x2) format
        rois = rois.clone()
        rois[:, [0, 2]] = rois[:, [1, 3]]
        rois[:, [1, 3]] = rois[:, [0, 2]]

        output = F.roi_pool2d(features, rois, (self.pooled_height, self.pooled_width), self.spatial_scale)

        return output

    def backward(self, grad_output, rois):
        assert(self.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = F.roi_pool2d_backward(grad_output, self.feature_size, rois, (self.pooled_height, self.pooled_width), self.spatial_scale)

        return grad_input, None
