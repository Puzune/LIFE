// Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
// Modified by Haisong Liu

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> render_forward_cuda(torch::Tensor sigma,
                                               torch::Tensor origin,
                                               torch::Tensor points,
                                               torch::Tensor tindex);

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor>
render_forward(torch::Tensor sigma, torch::Tensor origin, torch::Tensor points,
               torch::Tensor tindex) {
  CHECK_INPUT(sigma);
  CHECK_INPUT(origin);
  CHECK_INPUT(points);
  CHECK_INPUT(tindex);
  return render_forward_cuda(sigma, origin, points, tindex);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_forward", &render_forward, "Render (forward pass only)");
}
