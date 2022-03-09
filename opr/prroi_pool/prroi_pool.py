import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from . import prroi_pool_ext


class PrRoIPoolFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale):
        assert features.is_cuda
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        features = features.contiguous()
        rois = rois.contiguous()
        num_channels = features.size(1)
        num_rois = rois.size(0)
        out_size = (num_rois, num_channels, out_h, out_w)
        output = features.new_zeros(out_size)
        prroi_pool_ext.forward(features, rois, out_h, out_w, spatial_scale,
                                output)
        ctx.params = (out_h, out_w, spatial_scale)
        ctx.feature_size = features.size()
        ctx.rois_size = rois.size()
        ctx.save_for_backward(features, rois, output)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        out_h, out_w, spatial_scale = ctx.params
        feature_size = ctx.feature_size
        rois_size = ctx.rois_size

        features, rois, output = ctx.saved_tensors
        assert feature_size is not None
        assert rois_size is not None

        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.new_zeros(feature_size)
            grad_rois = grad_output.new_zeros(rois_size)
            prroi_pool_ext.backward(features, rois, output,
                                     grad_output.contiguous(),
                                     out_h, out_w, spatial_scale,
                                     grad_input, grad_rois)
        if not ctx.needs_input_grad[1]:
            grad_rois = None

        return grad_input, grad_rois, None, None


prroi_pool = PrRoIPoolFunction.apply


class PrRoIPool(nn.Module):

    def __init__(self, out_size, spatial_scale):
        super(PrRoIPool, self).__init__()

        self.out_size = _pair(out_size)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return prroi_pool(features, rois, self.out_size, self.spatial_scale)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}'.format(
            self.out_size, self.spatial_scale)
        return format_str
