#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__device__ static scalar_t PrRoIPoolingGetData(const scalar_t *data, const int h, const int w,
                                               const int height, const int width) {
  bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
  scalar_t retVal = overflow ? scalar_t(0.0) : data[h * width + w];
  return retVal;
}

template <typename scalar_t>
__device__ static void PrRoIPoolingDistributeDiff(scalar_t *diff, const scalar_t top_diff,
                                                  const int h, const int w,
                                                  const int height, const int width,
                                                  const scalar_t coeff) {
  bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
  if (!overflow) {
    atomicAdd(diff + h * width + w, top_diff * coeff);
  }
}

template <typename scalar_t>
__device__ static scalar_t PrRoIPoolingGetCoeff(scalar_t dh, scalar_t dw) {
  dw = dw > 0 ? dw : -dw;
  dh = dh > 0 ? dh : -dh;
  return (scalar_t(1.0) - dh) * (scalar_t(1.0) - dw);
}

template <typename scalar_t>
__device__ static scalar_t PrRoIPoolingSingleCoorIntegral(scalar_t s, scalar_t t, scalar_t c1, scalar_t c2) {
  return 0.5 * (t * t - s * s) * c2 + (t - 0.5 * t * t - s + 0.5 * s * s) * c1;
}

template <typename scalar_t>
__device__ static scalar_t PrRoIPoolingInterpolation(const scalar_t *data,
                                                     const scalar_t h,
                                                     const scalar_t w,
                                                     const int height,
                                                     const int width) {
  scalar_t retVal = 0.0;
  int h1 = floorf(h);
  int w1 = floorf(w);
  retVal += PrRoIPoolingGetData<scalar_t>(data, h1, w1, height, width) *
            PrRoIPoolingGetCoeff<scalar_t>(h - static_cast<scalar_t>(h1), w - static_cast<scalar_t>(w1));
  h1 = floorf(h) + 1;
  w1 = floorf(w);
  retVal += PrRoIPoolingGetData<scalar_t>(data, h1, w1, height, width) *
            PrRoIPoolingGetCoeff<scalar_t>(h - static_cast<scalar_t>(h1), w - static_cast<scalar_t>(w1));
  h1 = floorf(h);
  w1 = floorf(w) + 1;
  retVal += PrRoIPoolingGetData<scalar_t>(data, h1, w1, height, width) *
            PrRoIPoolingGetCoeff<scalar_t>(h - static_cast<scalar_t>(h1), w - static_cast<scalar_t>(w1));
  h1 = floorf(h) + 1;
  w1 = floorf(w) + 1;
  retVal += PrRoIPoolingGetData<scalar_t>(data, h1, w1, height, width) *
            PrRoIPoolingGetCoeff<scalar_t>(h - static_cast<scalar_t>(h1), w - static_cast<scalar_t>(w1));
  return retVal;
}

template <typename scalar_t>
__device__ static scalar_t PrRoIPoolingMatCalculation(const scalar_t *this_data,
                                                      const int s_h, const int s_w,
                                                      const int e_h, const int e_w,
                                                      const scalar_t y0, const scalar_t x0,
                                                      const scalar_t y1, const scalar_t x1,
                                                      const int h0, const int w0) {
  scalar_t alpha, beta, lim_alpha, lim_beta, tmp;
  scalar_t sum_out = 0;

  alpha = x0 - static_cast<scalar_t>(s_w);
  beta = y0 - static_cast<scalar_t>(s_h);
  lim_alpha = x1 - static_cast<scalar_t>(s_w);
  lim_beta = y1 - static_cast<scalar_t>(s_h);
  tmp = (lim_alpha - scalar_t(0.5) * lim_alpha * lim_alpha - alpha + scalar_t(0.5) * alpha * alpha) *
        (lim_beta - scalar_t(0.5) * lim_beta * lim_beta - beta + scalar_t(0.5) * beta * beta);
  sum_out += PrRoIPoolingGetData<scalar_t>(this_data, s_h, s_w, h0, w0) * tmp;

  alpha = static_cast<scalar_t>(e_w) - x1;
  lim_alpha = static_cast<scalar_t>(e_w) - x0;
  tmp = (lim_alpha - scalar_t(0.5) * lim_alpha * lim_alpha - alpha + scalar_t(0.5) * alpha * alpha) *
        (lim_beta - scalar_t(0.5) * lim_beta * lim_beta - beta + scalar_t(0.5) * beta * beta);
  sum_out += PrRoIPoolingGetData<scalar_t>(this_data, s_h, e_w, h0, w0) * tmp;

  alpha = x0 - static_cast<scalar_t>(s_w);
  beta = static_cast<scalar_t>(e_h) - y1;
  lim_alpha = x1 - static_cast<scalar_t>(s_w);
  lim_beta = static_cast<scalar_t>(e_h) - y0;
  tmp = (lim_alpha - scalar_t(0.5) * lim_alpha * lim_alpha - alpha + scalar_t(0.5) * alpha * alpha) *
        (lim_beta - scalar_t(0.5) * lim_beta * lim_beta - beta + scalar_t(0.5) * beta * beta);
  sum_out += PrRoIPoolingGetData<scalar_t>(this_data, e_h, s_w, h0, w0) * tmp;

  alpha = static_cast<scalar_t>(e_w) - x1;
  lim_alpha = static_cast<scalar_t>(e_w) - x0;
  tmp = (lim_alpha - scalar_t(0.5) * lim_alpha * lim_alpha - alpha + scalar_t(0.5) * alpha * alpha) *
        (lim_beta - scalar_t(0.5) * lim_beta * lim_beta - beta + scalar_t(0.5) * beta * beta);
  sum_out += PrRoIPoolingGetData<scalar_t>(this_data, e_h, e_w, h0, w0) * tmp;

  return sum_out;
}

template <typename scalar_t>
__device__ static void PrRoIPoolingMatDistributeDiff(scalar_t *diff, const scalar_t top_diff,
                                                     const int s_h, const int s_w,
                                                     const int e_h, const int e_w,
                                                     const scalar_t y0, const scalar_t x0,
                                                     const scalar_t y1, const scalar_t x1,
                                                     const int h0, const int w0) {
  scalar_t alpha, beta, lim_alpha, lim_beta, tmp;

  alpha = x0 - static_cast<scalar_t>(s_w);
  beta = y0 - static_cast<scalar_t>(s_h);
  lim_alpha = x1 - static_cast<scalar_t>(s_w);
  lim_beta = y1 - static_cast<scalar_t>(s_h);
  tmp = (lim_alpha - scalar_t(0.5) * lim_alpha * lim_alpha - alpha + scalar_t(0.5) * alpha * alpha) *
        (lim_beta - scalar_t(0.5) * lim_beta * lim_beta - beta + scalar_t(0.5) * beta * beta);
  PrRoIPoolingDistributeDiff<scalar_t>(diff, top_diff, s_h, s_w, h0, w0, tmp);

  alpha = static_cast<scalar_t>(e_w) - x1;
  lim_alpha = static_cast<scalar_t>(e_w) - x0;
  tmp = (lim_alpha - scalar_t(0.5) * lim_alpha * lim_alpha - alpha + scalar_t(0.5) * alpha * alpha) *
        (lim_beta - scalar_t(0.5) * lim_beta * lim_beta - beta + scalar_t(0.5) * beta * beta);
  PrRoIPoolingDistributeDiff<scalar_t>(diff, top_diff, s_h, e_w, h0, w0, tmp);

  alpha = x0 - static_cast<scalar_t>(s_w);
  beta = static_cast<scalar_t>(e_h) - y1;
  lim_alpha = x1 - static_cast<scalar_t>(s_w);
  lim_beta = static_cast<scalar_t>(e_h) - y0;
  tmp = (lim_alpha - scalar_t(0.5) * lim_alpha * lim_alpha - alpha + scalar_t(0.5) * alpha * alpha) *
        (lim_beta - scalar_t(0.5) * lim_beta * lim_beta - beta + scalar_t(0.5) * beta * beta);
  PrRoIPoolingDistributeDiff<scalar_t>(diff, top_diff, e_h, s_w, h0, w0, tmp);

  alpha = static_cast<scalar_t>(e_w) - x1;
  lim_alpha = static_cast<scalar_t>(e_w) - x0;
  tmp = (lim_alpha - scalar_t(0.5) * lim_alpha * lim_alpha - alpha + scalar_t(0.5) * alpha * alpha) *
        (lim_beta - scalar_t(0.5) * lim_beta * lim_beta - beta + scalar_t(0.5) * beta * beta);
  PrRoIPoolingDistributeDiff<scalar_t>(diff, top_diff, e_h, e_w, h0, w0, tmp);
}

template <typename scalar_t>
__global__ void PrROIPoolForward(const int nthreads, const scalar_t *bottom_data,
                                 const scalar_t *rois, const scalar_t spatial_scale,
                                 const int channels, const int height,
                                 const int width, const int pooled_h,
                                 const int pooled_w, scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_w;
    int ph = (index / pooled_w) % pooled_h;
    int c = (index / pooled_w / pooled_h) % channels;
    int n = index / pooled_w / pooled_h / channels;

    const scalar_t *offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    // calculate the roi region on feature maps
    scalar_t roi_start_w = offset_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_rois[2] * spatial_scale;
    scalar_t roi_end_w = offset_rois[3] * spatial_scale;
    scalar_t roi_end_h = offset_rois[4] * spatial_scale;

    scalar_t roi_w = max(roi_end_w - roi_start_w, static_cast<scalar_t>(0.0));
    scalar_t roi_h = max(roi_end_h - roi_start_h, static_cast<scalar_t>(0.0));
    scalar_t bin_size_w = roi_w / static_cast<scalar_t>(pooled_w);
    scalar_t bin_size_h = roi_h / static_cast<scalar_t>(pooled_h);

    scalar_t win_start_w = roi_start_w + bin_size_w * pw;
    scalar_t win_start_h = roi_start_h + bin_size_h * ph;
    scalar_t win_end_w = win_start_w + bin_size_w;
    scalar_t win_end_h = win_start_h + bin_size_h;

    scalar_t win_size = max(static_cast<scalar_t>(0.0), bin_size_w * bin_size_h);
    const scalar_t *offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

    if (win_size > static_cast<scalar_t>(0.0)) {
      int s_w = floor(win_start_w);
      int e_w = ceil(win_end_w);
      int s_h = floor(win_start_h);
      int e_h = ceil(win_end_h);
      scalar_t sum_out = 0;

    	for (int w_iter = s_w; w_iter < e_w; ++w_iter) {
        for (int h_iter = s_h; h_iter < e_h; ++h_iter) {
          sum_out += PrRoIPoolingMatCalculation<scalar_t>(
              offset_bottom_data, h_iter, w_iter, h_iter + 1, w_iter + 1,
              max(win_start_h, static_cast<scalar_t>(h_iter)),
              max(win_start_w, static_cast<scalar_t>(w_iter)),
              min(win_end_h, static_cast<scalar_t>(h_iter) + static_cast<scalar_t>(1.0)),
              min(win_end_w, static_cast<scalar_t>(w_iter) + static_cast<scalar_t>(1.0)),
              height, width);
        }
      }
      top_data[index] = sum_out / win_size;
    } else {
      top_data[index] = 0.;
    }
  }
}

int PrROIPoolForwardLaucher(const at::Tensor features, const at::Tensor rois,
                            const float spatial_scale, const int channels,
                            const int height, const int width, const int num_rois,
                            const int pooled_h, const int pooled_w,
                            at::Tensor output) {
  const int output_size = num_rois * channels * pooled_h * pooled_w;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "PrROIPoolLaucherForward", ([&] {
        const scalar_t *bottom_data = features.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();

        PrROIPoolForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                channels, height, width, pooled_h, pooled_w, top_data);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}

template <typename scalar_t>
__global__ void PrROIPoolBackward(const int nthreads, const scalar_t *bottom_data,
                                  const scalar_t *rois, const scalar_t *top_data,
                                  const scalar_t *top_diff, const scalar_t spatial_scale,
                                  const int channels, const int height, const int width,
                                  const int pooled_h, const int pooled_w,
                                  scalar_t *feats_diff, scalar_t *rois_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_w;
    int ph = (index / pooled_w) % pooled_h;
    int c = (index / pooled_w / pooled_h) % channels;
    int n = index / pooled_w / pooled_h / channels;

    const scalar_t *offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    // calculate the roi region on feature maps
    scalar_t roi_start_w = offset_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_rois[2] * spatial_scale;
    scalar_t roi_end_w = offset_rois[3] * spatial_scale;
    scalar_t roi_end_h = offset_rois[4] * spatial_scale;

	scalar_t roi_w = max(roi_end_w - roi_start_w, static_cast<scalar_t>(0.0));
    scalar_t roi_h = max(roi_end_h - roi_start_h, static_cast<scalar_t>(0.0));
    scalar_t bin_size_w = roi_w / static_cast<scalar_t>(pooled_w);
    scalar_t bin_size_h = roi_h / static_cast<scalar_t>(pooled_h);

	scalar_t win_start_w = roi_start_w + bin_size_w * pw;
    scalar_t win_start_h = roi_start_h + bin_size_h * ph;
    scalar_t win_end_w = win_start_w + bin_size_w;
    scalar_t win_end_h = win_start_h + bin_size_h;

	scalar_t win_size = max(static_cast<scalar_t>(0.0), bin_size_w * bin_size_h);
	int s_w = floor(win_start_w);
    int e_w = ceil(win_end_w);
    int s_h = floor(win_start_h);
    int e_h = ceil(win_end_h);

    // backward for features
    scalar_t *offset_feats_diff = feats_diff + (roi_batch_ind * channels + c) * height * width;
    const scalar_t *offset_top_diff = top_diff + index;
    scalar_t sum_out = win_size == static_cast<scalar_t>(0.) ? static_cast<scalar_t>(0.) : *offset_top_diff / win_size;

    for (int w_iter = s_w; w_iter < e_w; ++w_iter) {
      for (int h_iter = s_h; h_iter < e_h; ++h_iter) {
        PrRoIPoolingMatDistributeDiff<scalar_t>(
            offset_feats_diff, sum_out, h_iter, w_iter, h_iter + 1,
            w_iter + 1, max(win_start_h, static_cast<scalar_t>(h_iter)),
            max(win_start_w, static_cast<scalar_t>(w_iter)),
            min(win_end_h, static_cast<scalar_t>(h_iter) + static_cast<scalar_t>(1.0)),
            min(win_end_w, static_cast<scalar_t>(w_iter) + static_cast<scalar_t>(1.0)),
            height, width);
      }
    }

    // backward for rois
    scalar_t *offset_rois_diff = rois_diff + n * 5;
    const scalar_t *offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
    const scalar_t *offset_top_data = top_data + index;

    scalar_t g_x1_y = 0, g_x2_y = 0, g_x_y1 = 0, g_x_y2 = 0;
    for (int h_iter = s_h; h_iter < e_h; ++h_iter) {
      g_x1_y += PrRoIPoolingSingleCoorIntegral<scalar_t>(
        max(win_start_h, static_cast<scalar_t>(h_iter)) - h_iter,
        min(win_end_h, static_cast<scalar_t>(h_iter + 1)) - h_iter,
        PrRoIPoolingInterpolation<scalar_t>(offset_bottom_data, h_iter, win_start_w, height, width),
        PrRoIPoolingInterpolation<scalar_t>(offset_bottom_data, h_iter + 1, win_start_w, height, width));

      g_x2_y += PrRoIPoolingSingleCoorIntegral<scalar_t>(
        max(win_start_h, static_cast<scalar_t>(h_iter)) - h_iter,
        min(win_end_h, static_cast<scalar_t>(h_iter + 1)) - h_iter,
        PrRoIPoolingInterpolation<scalar_t>(offset_bottom_data, h_iter, win_end_w, height, width),
        PrRoIPoolingInterpolation<scalar_t>(offset_bottom_data, h_iter + 1, win_end_w, height, width));
	}
    for (int w_iter = s_w; w_iter < e_w; ++w_iter) {
      g_x_y1 += PrRoIPoolingSingleCoorIntegral<scalar_t>(
        max(win_start_w, static_cast<scalar_t>(w_iter)) - w_iter,
        min(win_end_w, static_cast<scalar_t>(w_iter + 1)) - w_iter,
        PrRoIPoolingInterpolation<scalar_t>(offset_bottom_data, win_start_h, w_iter, height, width),
        PrRoIPoolingInterpolation<scalar_t>(offset_bottom_data, win_start_h, w_iter + 1, height, width));

      g_x_y2 += PrRoIPoolingSingleCoorIntegral<scalar_t>(
        max(win_start_w, static_cast<scalar_t>(w_iter)) - w_iter,
        min(win_end_w, static_cast<scalar_t>(w_iter + 1)) - w_iter,
        PrRoIPoolingInterpolation<scalar_t>(offset_bottom_data, win_end_h, w_iter, height, width),
        PrRoIPoolingInterpolation<scalar_t>(offset_bottom_data, win_end_h, w_iter + 1, height, width));
    }

	scalar_t partial_x1 = -g_x1_y + (win_end_h - win_start_h) * (*offset_top_data);
    scalar_t partial_y1 = -g_x_y1 + (win_end_w - win_start_w) * (*offset_top_data);
	scalar_t partial_x2 = g_x2_y - (win_end_h - win_start_h) * (*offset_top_data);
	scalar_t partial_y2 = g_x_y2 - (win_end_w - win_start_w) * (*offset_top_data);

	partial_x1 = partial_x1 / win_size * spatial_scale;
	partial_x2 = partial_x2 / win_size * spatial_scale;
	partial_y1 = partial_y1 / win_size * spatial_scale;
	partial_y2 = partial_y2 / win_size * spatial_scale;

	offset_rois_diff[0] = 0;
	atomicAdd(offset_rois_diff + 1,
              (partial_x1 * (scalar_t(1.0) - static_cast<scalar_t>(pw) / pooled_w) +
              partial_x2 * (scalar_t(1.0) - static_cast<scalar_t>(pw + 1) / pooled_w)) *
              (*offset_top_diff));
	atomicAdd(offset_rois_diff + 2,
	          (partial_y1 * (scalar_t(1.0) - static_cast<scalar_t>(ph) / pooled_h) +
	          partial_y2 * (scalar_t(1.0) - static_cast<scalar_t>(ph + 1) / pooled_h)) *
	          (*offset_top_diff));
	atomicAdd(offset_rois_diff + 3,
	          (partial_x2 * static_cast<scalar_t>(pw + 1) / pooled_w +
	          partial_x1 * static_cast<scalar_t>(pw) / pooled_w) *
	          (*offset_top_diff));
	atomicAdd(offset_rois_diff + 4,
	          (partial_y2 * static_cast<scalar_t>(ph + 1) / pooled_h +
	          partial_y1 * static_cast<scalar_t>(ph) / pooled_h) *
	          (*offset_top_diff));
  }
}

int PrROIPoolBackwardLaucher(const at::Tensor features, const at::Tensor rois,
                             const at::Tensor output, const at::Tensor top_grad,
                             const float spatial_scale, const int channels,
                             const int height, const int width,
                             const int num_rois, const int pooled_h,
                             const int pooled_w, const int batch_size,
                             at::Tensor feats_grad, at::Tensor rois_grad) {
  const int output_size = num_rois * channels * pooled_h * pooled_w;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "PrROIPoolLaucherBackward", ([&] {
      	const scalar_t *bottom_data = features.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        const scalar_t *top_data = output.data<scalar_t>();
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        scalar_t *feats_diff = feats_grad.data<scalar_t>();
        scalar_t *rois_diff = rois_grad.data<scalar_t>();

        if (sizeof(scalar_t) == sizeof(double)) {
          fprintf(stderr, "double is not supported\n");
          exit(-1);
        }

        PrROIPoolBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, rois_data, top_data, top_diff,
                scalar_t(spatial_scale), channels, height, width, pooled_h,
                pooled_w, feats_diff, rois_diff);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}
