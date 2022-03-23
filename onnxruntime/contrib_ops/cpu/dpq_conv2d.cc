#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/util/math_cpuonly.h"

#include <functional>

namespace onnxruntime {
namespace contrib {
class DPQConv2d final : public OpKernel {
 public:
  explicit DPQConv2d(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttrs<int64_t>("kernel_size", attrs_.kernel_size).IsOK(), "");
    ORT_ENFORCE(info.GetAttrs<int64_t>("padding", attrs_.padding).IsOK(), "");
    ORT_ENFORCE(info.GetAttrs<int64_t>("stride", attrs_.stride).IsOK(), "");
  }
  Status Compute(OpKernelContext* ctx) const override;

 private:
  void ComputeImpl(const Tensor* input, const Tensor* bias,
                   const Tensor* centroids, const Tensor* lut, const Tensor* scale,
                   int64_t b, int64_t c, int64_t h, int64_t w, int64_t out_h, int64_t out_w, int64_t ncodebooks, int64_t k, int64_t m,
                   Tensor* output) const;

  struct SimpleConvAttributes {
    std::vector<int64_t> kernel_size;
    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
  } attrs_;
};

ONNX_OPERATOR_KERNEL_EX(
    DPQConv2d,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraints<float>())
        .TypeConstraint("T2", BuildKernelDefConstraints<int8_t>()),
    DPQConv2d);

void DPQConv2d::ComputeImpl(const Tensor* input, const Tensor* bias,
                            const Tensor* centroids, const Tensor* lut, const Tensor* scale,
                            int64_t b, int64_t c, int64_t h, int64_t w, int64_t out_h, int64_t out_w, int64_t ncodebooks, int64_t k, int64_t m,
                            Tensor* output) const {
  float* output_data = output->MutableData<float>();
  for (int64_t i = 0; i < b * m * out_h * out_w; i++) output_data[i] = i;
}

Status DPQConv2d::Compute(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const Tensor* bias = ctx->Input<Tensor>(1);
  const Tensor* centroids = ctx->Input<Tensor>(2);
  const Tensor* lut = ctx->Input<Tensor>(3);
  const Tensor* scale = ctx->Input<Tensor>(4);

  const auto& input_shape = input->Shape();
  const auto& centroids_shape = centroids->Shape();
  const auto& lut_shape = lut->Shape();

  int64_t b = input_shape.GetDims()[0];
  int64_t c = input_shape.GetDims()[1];
  int64_t h = input_shape.GetDims()[2];
  int64_t w = input_shape.GetDims()[3];
  int64_t out_h = (h + 2 * attrs_.padding[0] - attrs_.kernel_size[0]) / attrs_.stride[0] + 1;
  int64_t out_w = (w + 2 * attrs_.padding[1] - attrs_.kernel_size[1]) / attrs_.stride[1] + 1;
  int64_t ncodebooks = centroids_shape.GetDims()[0];
  int64_t k = centroids_shape.GetDims()[1];
  int64_t m = lut_shape.GetDims()[2];

  TensorShape output_shape({b, m, out_h, out_w});
  Tensor* output = ctx->Output(0, output_shape);

  ComputeImpl(input, bias, centroids, lut, scale, b, c, h, w, out_h, out_w, ncodebooks, k, m, output);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime