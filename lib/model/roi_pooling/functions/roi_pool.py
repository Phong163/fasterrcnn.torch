import torch
import torch.nn.functional as F
from torch.autograd import Function

class RoIPoolFunction(Function):
    def __init__(ctx, pooled_height, pooled_width, spatial_scale):
        ctx.pooled_width = int(pooled_width)
        ctx.pooled_height = int(pooled_height)
        ctx.spatial_scale = float(spatial_scale)
        ctx.feature_size = None

    def forward(ctx, features, rois): 
        ctx.feature_size = features.size()           
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        num_rois = rois.size(0)

        # Convert RoIs from (x1, y1, x2, y2) to (y1, x1, y2, x2) format
        rois = rois.clone()
        rois[:, [0, 2]] = rois[:, [1, 3]]
        rois[:, [1, 3]] = rois[:, [0, 2]]

        output = F.roi_pool2d(features, rois, (ctx.pooled_height, ctx.pooled_width), ctx.spatial_scale)

        return output

    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = ctx.feature_size

        grad_input = F.roi_pool2d_backward(grad_output, ctx.feature_size, rois, (ctx.pooled_height, ctx.pooled_width), ctx.spatial_scale)

        return grad_input, None
