#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/platform/threadpool.h"
#include "core/util/math_cpuonly.h"

#include "ops/dpq_conv2d.h"
#include "utils/fmt.h"

#include "unsupported/Eigen/CXX11/Tensor"

#include <functional>

namespace onnxruntime {
namespace contrib {
class DPQConv2d final : public OpKernel {
 private:
  BufferUniquePtr packed_y_, packed_lut_;
  int64_t ncodebooks_, output_channels_, subvec_len_;

 public:
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override {
    using Eigen::array;
    using Eigen::RowMajor;
    using Eigen::TensorMap;

    is_packed = false;
    if (input_idx == 2) {
      is_packed = true;

      // centroids
      const auto& centroids_shape = tensor.Shape();
      int64_t k;
      TensorMap<Eigen::Tensor<const float, 3, RowMajor>> centroids_tensor(
          tensor.Data<float>(),
          ncodebooks_ = centroids_shape.GetDims()[0],
          k = centroids_shape.GetDims()[1],
          subvec_len_ = centroids_shape.GetDims()[2]);

      float* buffer = (float*)alloc->Alloc(centroids_tensor.size() * sizeof(float));
      TensorMap<Eigen::Tensor<float, 4, RowMajor>> packed_y_tensor(buffer, ncodebooks_, k / 8, subvec_len_, 8);
      packed_y_ = BufferUniquePtr(buffer, BufferDeleter(alloc));
      packed_y_tensor = centroids_tensor.reshape(array<int64_t, 4>{ncodebooks_, k / 8, 8, subvec_len_})
                            .shuffle(array<int, 4>{0, 1, 3, 2});

      if (prepacked_weights != nullptr) {
        prepacked_weights->buffers_.push_back(std::move(packed_y_));
        prepacked_weights->buffer_sizes_.push_back(centroids_tensor.size());
      }
    } else if (input_idx == 3) {
      is_packed = true;
      // lookup table

      const auto& lut_shape = tensor.Shape();
      const int M = 2;
      int64_t k;
      TensorMap<Eigen::Tensor<const int8_t, 3, RowMajor>> lut_tensor(
          tensor.Data<int8_t>(),
          ncodebooks_ = lut_shape.GetDims()[0],
          k = lut_shape.GetDims()[1],
          output_channels_ = lut_shape.GetDims()[2]);

      int8_t* buffer = (int8_t*)alloc->Alloc(lut_tensor.size() * sizeof(int8_t));
      TensorMap<Eigen::Tensor<int8_t, 4, RowMajor>> packed_lut_tensor(buffer, output_channels_ / M, ncodebooks_, M, k);
      packed_lut_tensor = lut_tensor.reshape(array<int64_t, 4>{ncodebooks_, k, output_channels_ / M, M})
                              .shuffle(array<int, 4>{2, 0, 3, 1});
      packed_lut_ = BufferUniquePtr(buffer, BufferDeleter(alloc));

      if (prepacked_weights != nullptr) {
        prepacked_weights->buffers_.push_back(std::move(packed_lut_));
        prepacked_weights->buffer_sizes_.push_back(lut_tensor.size());
      }
    }
    return Status::OK();
  }

  explicit DPQConv2d(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttrs<int64_t>("kernel_size", attrs_.kernel_size).IsOK(), "");
    ORT_ENFORCE(info.GetAttrs<int64_t>("padding", attrs_.padding).IsOK(), "");
    ORT_ENFORCE(info.GetAttrs<int64_t>("stride", attrs_.stride).IsOK(), "");

    for (const auto& args : dpq_kernels::FusedDistArgmin::available_arguments()) {
      int64_t output_channels = args[1] * attrs_.stride[0];
      ops_[std::array<int64_t, 12>{args[0], args[1], args[2], args[3],
                                   args[4], args[5], args[6], args[7],
                                   args[8], args[9], args[10], output_channels}]
          .reset(new dpq_kernels::DPQConv2d(
              args[0], args[1], args[2], args[3],
              args[4], args[5], args[6], args[7],
              args[8], args[9], args[10], output_channels));
    }
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  void ComputeImpl(const Tensor* input, const Tensor* bias, const Tensor* scale,
                   int64_t b, int64_t c, int64_t h, int64_t w,
                   int8_t* index_data, Tensor* output) const;

  struct SimpleConvAttributes {
    std::vector<int64_t> kernel_size;
    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
  } attrs_;

  static std::map<
      std::array<int64_t, 12>,
      std::unique_ptr<dpq_kernels::DPQConv2d>>
      ops_;
};

std::map<
    std::array<int64_t, 12>,
    std::unique_ptr<dpq_kernels::DPQConv2d>>
    DPQConv2d::ops_ = {};

ONNX_OPERATOR_KERNEL_EX(
    DPQConv2d,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", BuildKernelDefConstraints<float>())
        .TypeConstraint("T2", BuildKernelDefConstraints<int8_t>()),
    DPQConv2d);

void DPQConv2d::ComputeImpl(const Tensor* input, const Tensor* bias, const Tensor* scale,
                            int64_t b, int64_t c, int64_t h, int64_t w,
                            int8_t* index_data, Tensor* output) const {
  int64_t subvec_len = c * attrs_.kernel_size[0] * attrs_.kernel_size[1] / ncodebooks_;
  std::array<int64_t, 12> params = {
      b, c, h, w,
      attrs_.kernel_size[0], attrs_.kernel_size[1],
      attrs_.stride[0], attrs_.stride[1],
      attrs_.padding[0], attrs_.padding[1],
      subvec_len, output_channels_};
  auto op = ops_[params].get();
  CHECK(op != nullptr);
  float bias_data = bias->Data<float>()[0];
  float scale_data = scale->Data<float>()[0];

  op->operator()(
      input->Data<float>(),
      (float*)packed_y_.get(),
      (int8_t*)packed_lut_.get(),
      scale_data, bias_data,
      index_data,
      output->MutableData<float>());
}

Status DPQConv2d::Compute(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const Tensor* bias = ctx->Input<Tensor>(1);
  const Tensor* scale = ctx->Input<Tensor>(4);

  const auto& input_shape = input->Shape();

  int64_t b = input_shape.GetDims()[0];
  int64_t c = input_shape.GetDims()[1];
  int64_t h = input_shape.GetDims()[2];
  int64_t w = input_shape.GetDims()[3];
  int64_t out_h = (h + 2 * attrs_.padding[0] - attrs_.kernel_size[0]) / attrs_.stride[0] + 1;
  int64_t out_w = (w + 2 * attrs_.padding[1] - attrs_.kernel_size[1]) / attrs_.stride[1] + 1;

  TensorShape output_shape({b, output_channels_, out_h, out_w});
  Tensor* output = ctx->Output(0, output_shape);

  AllocatorPtr alloc;
  CHECK(ctx->GetTempSpaceAllocator(&alloc).IsOK());
  int8_t* buffer = (int8_t*)alloc->Alloc(ncodebooks_ * b * h * w);
  auto buffer_holder = BufferUniquePtr(buffer, BufferDeleter(alloc));

  ComputeImpl(input, bias, scale, b, c, h, w, buffer, output);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime