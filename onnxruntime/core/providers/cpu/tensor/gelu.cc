// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

#include "core/platform/threadpool.h"
#include <unsupported/Eigen/SpecialFunctions>
#include "core/providers/cpu/element_wise_ranged_transform.h"
#include "core/providers/cpu/tensor/gelu.h"

#include <cstddef>
#include <memory>

using onnxruntime::narrow;
using namespace onnxruntime::common;

namespace onnxruntime {

// May revisit the implementations to support inplace computation, if needed.

#define ADD_TYPED_GELU_OP(data_type)                                      \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                         \
      Gelu,                                                               \
      20,                                                                 \
      data_type,                                                          \
      KernelDefBuilder()                                                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      Gelu<data_type>)

ADD_TYPED_GELU_OP(float);
ADD_TYPED_GELU_OP(MLFloat16);

#ifndef DISABLE_CONTRIB_OPS
namespace contrib {
ONNX_OPERATOR_KERNEL_EX(
    Gelu,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gelu<float>);
}
#endif

template <typename T>
Status Gelu<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const T* input_data = input->Data<T>();

  Tensor* output = context->Output(0, input->Shape());
  T* output_data = output->MutableData<T>();

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  int64_t elem_count = input->Shape().Size();
  constexpr int64_t length_per_task = 4096;  // this number comes from FastGelu.
  int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;

  if (approximation_algorithm_ == "tanh") {
    // Split the data into chunks of N elements (except the last chunk) and use the thread pool to
    // process chunks in parallel. N = 4096 is selected based on performance test results on input
    // shape 1x128x768. The formula is 0.5 * x * (1 + Tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))).
    concurrency::ThreadPool::TryBatchParallelFor(
        tp, static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          const auto start = task_idx * length_per_task;
          const T* p_input = input_data + start;
          T* p_output = output_data + start;
          int64_t count = std::min(length_per_task, elem_count - start);

          // MlasComputeGeluTanh requires distinct input/output buffers. This
          // call uses disjoint slices from the input and output tensors.
          MlasComputeGeluTanh(p_input, p_output, narrow<size_t>(count));
        },
        0);
    return Status::OK();
  } else if (approximation_algorithm_ == "none") {
    concurrency::ThreadPool::TryBatchParallelFor(
        tp, static_cast<int32_t>(task_count),
        [&](ptrdiff_t task_idx) {
          const auto start = task_idx * length_per_task;
          const T* p_input = input_data + start;
          T* p_output = output_data + start;
          int64_t count = std::min(length_per_task, elem_count - start);

          // MlasComputeGeluErf requires distinct input/output buffers. This
          // call uses disjoint slices from the input and output tensors.
          MlasComputeGeluErf(p_input, p_output, narrow<size_t>(count));
        },
        0);
    return Status::OK();
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported approximation_algorithm: ", approximation_algorithm_);
}

template <>
Status Gelu<MLFloat16>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const MLFloat16* input_data = input->Data<MLFloat16>();
  Tensor* output = context->Output(0, input->Shape());
  MLFloat16* output_data = output->MutableData<MLFloat16>();
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();

  int64_t elem_count = input->Shape().Size();
  constexpr int64_t length_per_task = 4096;
  int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;

  MLAS_GELU_ALGORITHM algo;
  if (approximation_algorithm_ == "tanh") {
    algo = MlasGeluTanh;
  } else if (approximation_algorithm_ == "none") {
    algo = MlasGeluErf;
  } else {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "Unsupported approximation_algorithm: ",
        approximation_algorithm_);
  }

  if (elem_count == 0) {
    return Status::OK();
  }

  // Allocate scratch buffer using ORT temp-space allocator
  size_t buffer_size = static_cast<size_t>(elem_count) * sizeof(MLFloat16);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  void* raw = allocator->Alloc(buffer_size);
  if (!raw) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to allocate temporary buffer.");
  }

  auto deleter = [allocator](MLFloat16* p) {
    if (p) allocator->Free(p);
  };

  std::unique_ptr<MLFloat16, decltype(deleter)> temp_fp16(
      static_cast<MLFloat16*>(raw), deleter);

  concurrency::ThreadPool::TryBatchParallelFor(
      tp,
      static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        const auto start = task_idx * length_per_task;
        const MLFloat16* p_input = input_data + start;
        MLFloat16* p_output = output_data + start;

        int64_t count = std::min(length_per_task, elem_count - start);

        MLFloat16* p_temp = temp_fp16.get() + start;

        MlasComputeFP16Gelu(
            p_input,
            p_output,
            p_temp,
            narrow<size_t>(count),
            algo);
      },
      0);

  return Status::OK();
}

}  // namespace onnxruntime