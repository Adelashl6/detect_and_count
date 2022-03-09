#include <torch/extension.h>

#include <cmath>
#include <vector>

#ifdef WITH_CUDA
int PrROIPoolForwardLaucher(const at::Tensor features, const at::Tensor rois,
                            const float spatial_scale, const int channels,
                            const int height, const int width, const int num_rois,
                            const int pooled_h, const int pooled_w,
                            at::Tensor output);

int PrROIPoolBackwardLaucher(const at::Tensor features, const at::Tensor rois,
                             const at::Tensor output, const at::Tensor top_grad,
                             const float spatial_scale, const int channels,
                             const int height, const int width,
                             const int num_rois, const int pooled_h,
                             const int pooled_w, const int batch_size,
                             at::Tensor feats_grad, at::Tensor rois_grad);

#endif

#define CHECK_CUDA(x) AT_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int prroi_pooling_forward(at::Tensor features, at::Tensor rois,
                               int pooled_height, int pooled_width,
                               float spatial_scale, at::Tensor output) {
  if (features.device().is_cuda()) {
#ifdef WITH_CUDA
    CHECK_INPUT(features);
    CHECK_INPUT(rois);
    CHECK_INPUT(output);
    at::DeviceGuard guard(features.device());

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);

    if (size_rois != 5) {
      printf("wrong roi size\n");
      return 0;
    }

    int channels = features.size(1);
    int height = features.size(2);
    int width = features.size(3);

    PrROIPoolForwardLaucher(features, rois, spatial_scale, channels, height, width,
                            num_rois, pooled_height, pooled_width, output);

    return 1;
#else
    AT_ERROR("prroi_pool is not compiled with GPU support");
#endif
  }
  AT_ERROR("prroi_pool is not implemented on CPU");
}

int prroi_pooling_backward(at::Tensor features, at::Tensor rois,
                                at::Tensor output, at::Tensor top_grad,
                                int pooled_height, int pooled_width,
                                float spatial_scale, at::Tensor feats_grad,
                                at::Tensor rois_grad) {
  if (top_grad.device().is_cuda()) {
#ifdef WITH_CUDA
    CHECK_INPUT(features);
    CHECK_INPUT(rois);
    CHECK_INPUT(output);
    CHECK_INPUT(top_grad);
    CHECK_INPUT(feats_grad);
    CHECK_INPUT(rois_grad);
    at::DeviceGuard guard(top_grad.device());

    int num_rois = rois.size(0);
    int size_rois = rois.size(1);

    if (size_rois != 5) {
      printf("wrong roi size\n");
      return 0;
    }
    int batch_size = features.size(0);
    int channels = features.size(1);
    int height = features.size(2);
    int width = features.size(3);

    PrROIPoolBackwardLaucher(features, rois, output, top_grad, spatial_scale,
                             channels, height, width, num_rois, pooled_height,
                             pooled_width, batch_size, feats_grad, rois_grad);

    return 1;
#else
    AT_ERROR("prroi_pool is not compiled with GPU support");
#endif
  }
  AT_ERROR("prroi_pool is not implemented on CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &prroi_pooling_forward, "PrRoi_Pooling forward");
  m.def("backward", &prroi_pooling_backward, "PrRoi_Pooling backward");
}
