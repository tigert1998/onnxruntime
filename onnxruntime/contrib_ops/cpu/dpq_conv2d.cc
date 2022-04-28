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
      int64_t ncodebooks, k, subvec_len;
      TensorMap<Eigen::Tensor<const float, 3, RowMajor>> centroids_tensor(
          tensor.Data<float>(),
          ncodebooks = centroids_shape.GetDims()[0],
          k = centroids_shape.GetDims()[1],
          subvec_len = centroids_shape.GetDims()[2]);

      float* buffer = (float*)alloc->Alloc(centroids_tensor.size());
      TensorMap<Eigen::Tensor<float, 4, RowMajor>> packed_y_tensor(buffer, ncodebooks, k / 8, subvec_len, 8);
      packed_y_tensor = centroids_tensor.reshape(array<int64_t, 4>{ncodebooks, k / 8, 8, subvec_len})
                            .shuffle(array<int, 4>{0, 1, 3, 2});
      packed_y_ = BufferUniquePtr(buffer, BufferDeleter(alloc));
    } else if (input_idx == 3) {
      is_packed = true;
      // lookup table

      const auto& lut_shape = tensor.Shape();
      const int M = 2;
      int64_t ncodebooks, k, output_channels;
      TensorMap<Eigen::Tensor<const int8_t, 3, RowMajor>> lut_tensor(
          tensor.Data<int8_t>(),
          ncodebooks = lut_shape.GetDims()[0],
          k = lut_shape.GetDims()[1],
          output_channels = lut_shape.GetDims()[2]);

      int8_t* buffer = (int8_t*)alloc->Alloc(lut_tensor.size());
      TensorMap<Eigen::Tensor<int8_t, 4, RowMajor>> packed_lut_tensor(buffer, output_channels / M, ncodebooks, M, k);
      packed_lut_tensor = lut_tensor.reshape(array<int64_t, 4>{ncodebooks, k, output_channels / M, M})
                              .shuffle(array<int, 4>{2, 0, 3, 1});
      packed_lut_ = BufferUniquePtr(buffer, BufferDeleter(alloc));
    }
    return Status::OK();
  }

  explicit DPQConv2d(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttrs<int64_t>("kernel_size", attrs_.kernel_size).IsOK(), "");
    ORT_ENFORCE(info.GetAttrs<int64_t>("padding", attrs_.padding).IsOK(), "");
    ORT_ENFORCE(info.GetAttrs<int64_t>("stride", attrs_.stride).IsOK(), "");

    // initialize dpq functions
    const std::string path = "/data/local/tmp/fused_dist_argmin_libs";

    for (auto tuple : std::vector<std::tuple<int64_t, int64_t>>{{16, 32}, {32, 16}, {64, 8}}) {
      int64_t channels, height;
      std::tie(channels, height) = tuple;
      for (int64_t output_channels : std::vector<int64_t>{channels, channels * 2}) {
        for (int64_t subvec_len : std::vector<int64_t>{4, 9}) {
          DPQConv2dParams params(
              1, channels, height, height,
              attrs_.kernel_size[0], attrs_.kernel_size[1],
              attrs_.stride[0], attrs_.stride[1],
              attrs_.padding[0], attrs_.padding[1],
              subvec_len, output_channels);
          std::string lib_path = path + "/" + params.GetLibraryName() + ".so";
          auto fp = fopen(lib_path.c_str(), "rb");
          if (fp == nullptr) continue;
          fclose(fp);

          if (ops_[params] != nullptr) continue;
          ops_[params].reset(new dpq_kernels::DPQConv2d(
              lib_path,
              std::get<0>(params), std::get<1>(params), std::get<2>(params), std::get<3>(params),
              std::get<4>(params), std::get<5>(params), std::get<6>(params), std::get<7>(params),
              std::get<8>(params), std::get<9>(params), std::get<10>(params), std::get<11>(params)));
        }
      }
    }
  }

  Status Compute(OpKernelContext* ctx) const override;

 private:
  void ComputeImpl(const Tensor* input, const Tensor* bias, const Tensor* scale,
                   int64_t b, int64_t c, int64_t h, int64_t w, int64_t ncodebooks, int64_t m,
                   int8_t* index_data, Tensor* output) const;

  struct SimpleConvAttributes {
    std::vector<int64_t> kernel_size;
    std::vector<int64_t> padding;
    std::vector<int64_t> stride;
  } attrs_;

  using tuple_12_t = std::tuple<int64_t, int64_t, int64_t, int64_t,
                                int64_t, int64_t, int64_t, int64_t,
                                int64_t, int64_t, int64_t, int64_t>;
  struct DPQConv2dParams : public tuple_12_t {
    std::string GetLibraryName() {
      using std::get;
      return dpq_kernels::Format("b%ldc%ldh%ldw%ldkh%ldkw%ldsh%ldsw%ldph%ldpw%ldv%ld",
                                 get<0>(*this), get<1>(*this), get<2>(*this), get<3>(*this),
                                 get<4>(*this), get<5>(*this), get<6>(*this), get<7>(*this),
                                 get<8>(*this), get<9>(*this), get<10>(*this));
    }
    using tuple_12_t::tuple;
  };

  static std::map<
      DPQConv2dParams,
      std::unique_ptr<dpq_kernels::DPQConv2d>>
      ops_;
};

std::map<
    DPQConv2d::DPQConv2dParams,
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
                            int64_t b, int64_t c, int64_t h, int64_t w, int64_t ncodebooks, int64_t m,
                            int8_t* index_data, Tensor* output) const {
  int64_t subvec_len = c * attrs_.kernel_size[0] * attrs_.kernel_size[1] / ncodebooks;
  DPQConv2dParams params(
      b, c, h, w,
      attrs_.kernel_size[0], attrs_.kernel_size[1],
      attrs_.stride[0], attrs_.stride[1],
      attrs_.padding[0], attrs_.padding[1],
      subvec_len, m);
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
  int64_t m = lut_shape.GetDims()[2];

  TensorShape output_shape({b, m, out_h, out_w});
  Tensor* output = ctx->Output(0, output_shape);

  AllocatorPtr alloc;
  CHECK(ctx->GetTempSpaceAllocator(&alloc).IsOK());
  int8_t* buffer = (int8_t*)alloc->Alloc(ncodebooks * b * h * w);
  auto buffer_holder = BufferUniquePtr(buffer, BufferDeleter(alloc));

  ComputeImpl(input, bias, scale, b, c, h, w, ncodebooks, m, buffer, output);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime