import torch
import torch.nn.functional as F
from torch.autograd import Function

class RoIAlignFunction(Function):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = F.grid_sample(features, self._generate_grid(rois, batch_size, num_channels, data_height, data_width), align_corners=False)

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = F.grid_sample(grad_output, self._generate_grid(self.rois, batch_size, num_channels, data_height, data_width), mode='bilinear', align_corners=False)

        return grad_input, None

    def _generate_grid(self, rois, batch_size, num_channels, data_height, data_width):
        rois[:, :, :4] /= self.spatial_scale
        rois[:, :, 2:4] = torch.clamp(rois[:, :, 2:4], min=0)

        rois[:, :, 0:4:2] = torch.clamp(rois[:, :, 0:4:2], max=data_width - 1)
        rois[:, :, 1:4:2] = torch.clamp(rois[:, :, 1:4:2], max=data_height - 1)

        rois[:, :, 0:4:2] = rois[:, :, 0:4:2] / (data_width - 1) * 2 - 1
        rois[:, :, 1:4:2] = rois[:, :, 1:4:2] / (data_height - 1) * 2 - 1

        grid = rois.new_zeros(batch_size, num_channels, self.aligned_height * self.aligned_width, 2)

        grid[:, :, :, 0] = (rois[:, :, 0] + rois[:, :, 2]).view(-1, 1).expand(batch_size, num_channels, self.aligned_height * self.aligned_width).contiguous().view(batch_size, num_channels, self.aligned_height, self.aligned_width)
        grid[:, :, :, 1] = (rois[:, :, 1] + rois[:, :, 3]).view(-1, 1).expand(batch_size, num_channels, self.aligned_height * self.aligned_width).contiguous().view(batch_size, num_channels, self.aligned_height, self.aligned_width)

        grid[:, :, :, 0] = grid[:, :, :, 0] / 2
        grid[:, :, :, 1] = grid[:, :, :, 1] / 2

        grid = grid.view(batch_size, num_channels, self.aligned_height, self.aligned_width, 2)
        grid = grid.permute(0, 1, 4, 2, 3).contiguous()
        grid = grid.view(batch_size, num_channels, 2, self.aligned_height * self.aligned_width)

        return grid
