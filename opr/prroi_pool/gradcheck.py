import os.path as osp
import sys

import torch
from torch.autograd import gradcheck

sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from prroi_pool import PrRoIPool  # noqa: E402, isort:skip

feat = torch.randn(1, 1, 15, 15, requires_grad=True).cuda()
rois = torch.Tensor([[0, 0, 0, 5, 5], [0, 1, 3, 4, 5],
                     [0, 6, 4, 11, 12]]).cuda()
inputs = (feat, rois)
print('Gradcheck for prroi pooling...')
test = gradcheck(PrRoIPool(1, 1.0), inputs, eps=1e-5, atol=1e-3)
print(test)
