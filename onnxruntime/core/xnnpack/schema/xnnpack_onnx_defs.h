// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <onnx/defs/schema.h>
#include <onnx/common/common.h>
#include <onnx/common/status.h>
#include "xnnpack_onnx_schema.h"

namespace onnxruntime {
constexpr const char* kXNNPackDomain = "com.microsoft.xnnpack";
namespace xnnpack {
using OnnxStatus = ::ONNX_NAMESPACE::Common::Status;
// When this function returns OK, *output_size should equals to input_size whenever stride=1.
OnnxStatus ComputeOutputSizeSame(ptrdiff_t input_size, uint32_t stride, ptrdiff_t* output_size);
OnnxStatus ComputeOutputSizeValid(ptrdiff_t input_size, uint32_t stride, ptrdiff_t filter_size, uint32_t dilation_rate,
                                  ptrdiff_t* output_size);
OnnxStatus XnnPackConvShapeInferImpl(const ::ONNX_NAMESPACE::TensorShapeProto& input_shape,
                                     const ::ONNX_NAMESPACE::TensorShapeProto& weight_shape, uint32_t input_padding_top,
                                     uint32_t input_padding_right, uint32_t input_padding_bottom,
                                     uint32_t input_padding_left, uint32_t subsampling_height,
                                     uint32_t subsampling_width, uint32_t dilation_h, uint32_t dilation_w,
                                     int padding_mode, ::ONNX_NAMESPACE::TensorShapeProto* final_output_shape);
OnnxStatus XnnPackDepthwiseConvolution2dShapeInferImpl(const ::ONNX_NAMESPACE::TensorShapeProto& input_shape,
                                                       const ::ONNX_NAMESPACE::TensorShapeProto& weight_shape,
                                                       uint32_t input_padding_top, uint32_t input_padding_right,
                                                       uint32_t input_padding_bottom, uint32_t input_padding_left,
                                                       uint32_t subsampling_height, uint32_t subsampling_width,
                                                       uint32_t dilation_h, uint32_t dilation_w, int padding_mode,
                                                       ::ONNX_NAMESPACE::TensorShapeProto* final_output_shape);
}  // namespace xnnpack
}  // namespace onnxruntime

#define ONNX_XNNPACK_OPERATOR_SET_SCHEMA(name, ver, impl) \
  ONNX_OPERATOR_SET_SCHEMA_EX(name, XnnPack, ::onnxruntime::kXNNPackDomain, ver, true, impl)

#ifndef ONNX_RETURN_IF_ERROR
#define ONNX_RETURN_IF_ERROR(expr) \
  do {                             \
    auto _status = (expr);         \
    if ((!_status.IsOK())) {       \
      return _status;              \
    }                              \
  } while (0)
#endif
