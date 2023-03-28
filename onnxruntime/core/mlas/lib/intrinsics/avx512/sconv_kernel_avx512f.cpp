/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_kernel_avx512.cpp

Abstract:

;   This module implements the kernels for the single precision
;   sliding window convolution operation.
;
;   This implementation uses AVX512F instructions.

--*/

#include <cassert>

#include "mlasi.h"
#include "sconv_kernel.h"

void
MlasConv1DSlidingKernelK17(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output) {
  constexpr size_t Dimensions = 1;
  constexpr size_t WidthShapeIndex = 0;

  const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
  const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

  const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
  const int64_t PaddingLeft = Parameters->Padding[WidthShapeIndex];
  const int64_t PaddingRight = Parameters->Padding[Dimensions + WidthShapeIndex];

  assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
  MLAS_UNREFERENCED_PARAMETER(PaddingRight);

  constexpr size_t widthStep = 32;

  const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

  const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

  const __mmask64 _leftInputMask = (~0ull << paddingLeftShift);

  const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);

  const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

  const int64_t widthIterations = InputWidthBase / widthStep;
  const int64_t widthInputRemainder = InputWidthBase % widthStep;
  const int64_t widthOutputRemainder = OutputWidth % widthStep;

  const __mmask64 _rightInputMask = ~(~0ull << widthInputRemainder);

  const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
  const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);

  const __mmask64 _rightOutputMask = ~(~0ull << widthOutputRemainder);

  const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
  const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);

  const __m512i _roll_left_1 = _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
  const __m512 _zero = _mm512_setzero_ps();
  const __m512i _one = _mm512_set1_epi32(1);

  float* pOutput0 = Output;

  const float* pInput0 = Input;

  __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

  pInput0 += paddingLeftSeek;

  for (int64_t i = 0; i < widthIterations; i++) {
    __m512 _b0_1 = _mm512_load_ps(pInput0);
    __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

    __m512 _b0 = _b0_1;
    __m512 _b1 = _b0_2;

    __m512i _roll = _roll_left_1;

    __m512 _acc0 = _zero;
    __m512 _acc1 = _zero;

    for (int64_t k = KernelWidth; k-- > 0;) {
      __m512 _ak = _mm512_set1_ps(Filter[k]);

      _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak, _b1, _acc1);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
      _b1 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

      _roll = _mm512_sub_epi32(_roll, _one);
    }

    _b0_0 = _b0_2;

    // Store
    _mm512_store_ps(pOutput0, _acc0);
    _mm512_store_ps(pOutput0 + 16, _acc1);

    pInput0 += widthStep;
    pOutput0 += widthStep;
  }

  // Right Edge
  {
    __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0);
    __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);

    __m512 _b0 = _b0_1;
    __m512 _b1 = _b0_2;

    __m512i _roll = _roll_left_1;

    __m512 _acc0 = _zero;
    __m512 _acc1 = _zero;

    for (int64_t k = KernelWidth; k-- > 0;) {
      __m512 _ak = _mm512_set1_ps(Filter[k]);

      _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak, _b1, _acc1);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
      _b1 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

      _roll = _mm512_sub_epi32(_roll, _one);
    }

    if (widthOutputRemainder < widthInputRemainder) {
      _mm512_store_ps(pOutput0, _acc0);
      _mm512_store_ps(pOutput0 + 16, _acc1);

      pOutput0 += widthStep;

      _b0 = _b1 = _zero;

      _roll = _roll_left_1;

      _acc0 = _acc1 = _zero;

      for (int64_t k = KernelWidth; k-- > 0;) {
        __m512 _ak = _mm512_set1_ps(Filter[k]);

        _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);

        _b0 = _mm512_permutex2var_ps(_b0_2, _roll, _zero);

        _roll = _mm512_sub_epi32(_roll, _one);
      }
    }

    // Store
    _mm512_mask_store_ps(pOutput0, _rightOutputMask1, _acc0);
    _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _acc1);
  }
}

void MlasConv1DSlidingKernelK32(const MLAS_CONV_PARAMETERS* Parameters,
                           const float* Input,
                           const float* Filter,
                           float* Output)
{
  constexpr size_t Dimensions = 1;
  constexpr size_t WidthShapeIndex = 0;

  const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
  const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

  const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
  const int64_t PaddingLeft = Parameters->Padding[WidthShapeIndex];
  const int64_t PaddingRight = Parameters->Padding[Dimensions + WidthShapeIndex];

  assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
  MLAS_UNREFERENCED_PARAMETER(PaddingRight);

  constexpr size_t widthStep = 32;

  const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

  const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

  const __mmask64 _leftInputMask = (~0ull << paddingLeftShift);

  const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMask);
  const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);

  const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

  const int64_t widthIterations = InputWidthBase / widthStep;
  const int64_t widthInputRemainder = InputWidthBase % widthStep;
  const int64_t widthOutputRemainder = OutputWidth % widthStep;

  const int64_t kernelRemainder = KernelWidth - 16;

  const __mmask64 _rightInputMask = ~(~0ull << widthInputRemainder);

  const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
  const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);

  const __mmask64 _rightOutputMask = ~(~0ull << widthOutputRemainder);

  const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
  const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);

  const __m512i _roll_left_1 = _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
  const __m512i _one = _mm512_set1_epi32(1);

  float* pOutput0 = Output;

  const float* pInput0 = Input;

  __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift + 0);
  __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

  pInput0 += paddingLeftSeek;

  for (int64_t i = 0; i < widthIterations; i++) {
    int64_t idx = KernelWidth;

    __m512 _b0_1 = _mm512_load_ps(pInput0);
    __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

    __m512 _aj0 = _mm512_set1_ps(Filter[--idx]);
    __m512 _aj1 = _mm512_set1_ps(Filter[idx - 16]);

    __m512 _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
    __m512 _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);

    __m512 _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
    __m512 _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);

    __m512i _roll = _roll_left_1;

    for (int64_t k = 1ll; k < kernelRemainder; k++) {
      _aj0 = _mm512_set1_ps(Filter[--idx]);
      _aj1 = _mm512_set1_ps(Filter[idx - 16]);

      __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
      __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

      _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

      _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
      _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

      _roll = _mm512_sub_epi32(_roll, _one);
    }

    for (int64_t k = kernelRemainder; k < 16ll; k++) {
      _aj0 = _mm512_set1_ps(Filter[--idx]);

      __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

      _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

      _roll = _mm512_sub_epi32(_roll, _one);
    }

    _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
    _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);

    _b0_f = _b0_1;
    _b0_0 = _b0_2;

    // Store
    _mm512_store_ps(pOutput0 + 0, _acc0_1);
    _mm512_store_ps(pOutput0 + 16, _acc0_2);

    pInput0 += widthStep;
    pOutput0 += widthStep;
  }

  // Right Edge
  {
    __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0 + 0);
    __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);

    int64_t idx = KernelWidth;

    __m512 _aj0 = _mm512_set1_ps(Filter[--idx]);
    __m512 _aj1 = _mm512_set1_ps(Filter[idx - 16]);

    __m512 _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
    __m512 _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);

    __m512 _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
    __m512 _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);

    __m512i _roll = _roll_left_1;

    for (int64_t k = 1ll; k < kernelRemainder; k++) {
      _aj0 = _mm512_set1_ps(Filter[--idx]);
      _aj1 = _mm512_set1_ps(Filter[idx - 16]);

      __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
      __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

      _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

      _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
      _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

      _roll = _mm512_sub_epi32(_roll, _one);
    }

    for (int64_t k = kernelRemainder; k < 16ll; k++) {
      _aj0 = _mm512_set1_ps(Filter[--idx]);

      __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

      _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

      _roll = _mm512_sub_epi32(_roll, _one);
    }

    _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
    _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);

    if (widthOutputRemainder < widthInputRemainder) {
      // Store
      _mm512_store_ps(pOutput0 + 0, _acc0_1);
      _mm512_store_ps(pOutput0 + 16, _acc0_2);

      pOutput0 += widthStep;

      _b0_f = _b0_1;
      _b0_0 = _b0_2;

      _b0_1 = _b0_2 = _mm512_setzero_ps();

      idx = KernelWidth;

      _aj0 = _mm512_set1_ps(Filter[--idx]);
      _aj1 = _mm512_set1_ps(Filter[idx - 16]);

      _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
      _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);

      _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
      _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);

      _roll = _roll_left_1;

      for (int64_t k = 1ll; k < kernelRemainder; k++) {
        _aj0 = _mm512_set1_ps(Filter[--idx]);
        _aj1 = _mm512_set1_ps(Filter[idx - 16]);

        __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
        __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
        __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

        _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
        _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      for (int64_t k = kernelRemainder; k < 16ll; k++) {
        _aj0 = _mm512_set1_ps(Filter[--idx]);

        __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
        __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
      _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);
    }

    // Store
    _mm512_mask_store_ps(pOutput0 + 0, _rightOutputMask1, _acc0_1);
    _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _acc0_2);
  }
}

void MlasConv1DSlidingKernelK48(const MLAS_CONV_PARAMETERS* Parameters,
                           const float* Input,
                           const float* Filter,
                           float* Output)
{
  constexpr size_t Dimensions = 1;
  constexpr size_t WidthShapeIndex = 0;

  const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
  const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

  const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
  const int64_t PaddingLeft = Parameters->Padding[WidthShapeIndex];
  const int64_t PaddingRight = Parameters->Padding[Dimensions + WidthShapeIndex];

  assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
  MLAS_UNREFERENCED_PARAMETER(PaddingRight);

  constexpr size_t widthStep = 48;

  const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

  const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

  const __mmask64 _leftInputMask = (~0ull << paddingLeftShift);

  const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMask);
  const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);
  const __mmask16 _leftInputMask3 = static_cast<__mmask16>(_leftInputMask >> 32);

  const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

  const int64_t widthIterations = InputWidthBase / widthStep;
  const int64_t widthInputRemainder = InputWidthBase % widthStep;
  const int64_t widthOutputRemainder = OutputWidth % widthStep;

  const int64_t kernelRemainder = KernelWidth - 32;

  const __mmask64 _rightInputMask = ~(~0ull << widthInputRemainder);

  const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
  const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);
  const __mmask16 _rightInputMask3 = static_cast<__mmask16>(_rightInputMask >> 32);

  const __mmask64 _rightOutputMask = ~(~0ull << widthOutputRemainder);

  const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
  const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);
  const __mmask16 _rightOutputMask3 = static_cast<__mmask16>(_rightOutputMask >> 32);

  const __m512i _roll_left_1 = _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
  const __m512i _one = _mm512_set1_epi32(1);

  float* pOutput0 = Output;

  const float* pInput0 = Input;

  __m512 _b0_e = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift + 0);
  __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);
  __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask3, pInput0 - paddingLeftShift + 32);

  pInput0 += paddingLeftSeek;

  for (int64_t i = 0; i < widthIterations; i++) {
    int64_t idx = KernelWidth;

    __m512 _b0_1 = _mm512_load_ps(pInput0);
    __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);
    __m512 _b0_3 = _mm512_load_ps(pInput0 + 32);

    __m512 _aj0 = _mm512_set1_ps(Filter[--idx]);
    __m512 _aj1 = _mm512_set1_ps(Filter[idx - 16]);
    __m512 _aj2 = _mm512_set1_ps(Filter[idx - 32]);

    __m512 _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
    __m512 _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);
    __m512 _acc0_3 = _mm512_mul_ps(_aj0, _b0_3);

    __m512 _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
    __m512 _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);
    __m512 _acc1_2 = _mm512_mul_ps(_aj1, _b0_2);

    _acc0_1 = _mm512_fmadd_ps(_aj2, _b0_f, _acc0_1);
    _acc0_2 = _mm512_fmadd_ps(_aj2, _b0_0, _acc0_2);
    _acc0_3 = _mm512_fmadd_ps(_aj2, _b0_1, _acc0_3);

    __m512i _roll = _roll_left_1;

    for (int64_t k = 1ll; k < kernelRemainder; k++) {
      _aj0 = _mm512_set1_ps(Filter[--idx]);
      _aj1 = _mm512_set1_ps(Filter[idx - 16]);
      _aj2 = _mm512_set1_ps(Filter[idx - 32]);

      __m512 _b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
      __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
      __m512 _bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);

      _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);

      _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
      _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
      _acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);

      _acc0_1 = _mm512_fmadd_ps(_aj2, _bf, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj2, _b0, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj2, _b1, _acc0_3);

      _roll = _mm512_sub_epi32(_roll, _one);
    }

    for (int64_t k = kernelRemainder; k < 16ll; k++) {
      _aj0 = _mm512_set1_ps(Filter[--idx]);
      _aj1 = _mm512_set1_ps(Filter[idx - 16]);

      __m512 _b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
      __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

      _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);

      _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
      _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
      _acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);

      _roll = _mm512_sub_epi32(_roll, _one);
    }

    _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
    _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);
    _acc0_3 = _mm512_add_ps(_acc0_3, _acc1_2);

    _b0_e = _b0_1;
    _b0_f = _b0_2;
    _b0_0 = _b0_3;

    // Store
    _mm512_store_ps(pOutput0 + 0, _acc0_1);
    _mm512_store_ps(pOutput0 + 16, _acc0_2);
    _mm512_store_ps(pOutput0 + 32, _acc0_3);

    pInput0 += widthStep;
    pOutput0 += widthStep;
  }

  // Right Edge
  {
    __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0 + 0);
    __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);
    __m512 _b0_3 = _mm512_maskz_load_ps(_rightInputMask3, pInput0 + 32);

    int64_t idx = KernelWidth;

    __m512 _aj0 = _mm512_set1_ps(Filter[--idx]);
    __m512 _aj1 = _mm512_set1_ps(Filter[idx - 16]);
    __m512 _aj2 = _mm512_set1_ps(Filter[idx - 32]);

    __m512 _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
    __m512 _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);
    __m512 _acc0_3 = _mm512_mul_ps(_aj0, _b0_3);

    __m512 _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
    __m512 _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);
    __m512 _acc1_2 = _mm512_mul_ps(_aj1, _b0_2);

    _acc0_1 = _mm512_fmadd_ps(_aj2, _b0_f, _acc0_1);
    _acc0_2 = _mm512_fmadd_ps(_aj2, _b0_0, _acc0_2);
    _acc0_3 = _mm512_fmadd_ps(_aj2, _b0_1, _acc0_3);

    __m512i _roll = _roll_left_1;

    for (int64_t k = 1ll; k < kernelRemainder; k++) {
      _aj0 = _mm512_set1_ps(Filter[--idx]);
      _aj1 = _mm512_set1_ps(Filter[idx - 16]);
      _aj2 = _mm512_set1_ps(Filter[idx - 32]);

      __m512 _b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
      __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
      __m512 _bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);

      _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);

      _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
      _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
      _acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);

      _acc0_1 = _mm512_fmadd_ps(_aj2, _bf, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj2, _b0, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj2, _b1, _acc0_3);

      _roll = _mm512_sub_epi32(_roll, _one);
    }

    for (int64_t k = kernelRemainder; k < 16ll; k++) {
      _aj0 = _mm512_set1_ps(Filter[--idx]);
      _aj1 = _mm512_set1_ps(Filter[idx - 16]);
      _aj2 = _mm512_set1_ps(Filter[idx - 32]);

      __m512 _b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
      __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

      _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);

      _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
      _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
      _acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);

      _roll = _mm512_sub_epi32(_roll, _one);
    }

    _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
    _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);
    _acc0_3 = _mm512_add_ps(_acc0_3, _acc1_2);

    if (widthOutputRemainder < widthInputRemainder) {
      // Store
      _mm512_store_ps(pOutput0 + 0, _acc0_1);
      _mm512_store_ps(pOutput0 + 16, _acc0_2);
      _mm512_store_ps(pOutput0 + 32, _acc0_3);

      pOutput0 += widthStep;

      _b0_e = _b0_1;
      _b0_f = _b0_2;
      _b0_0 = _b0_3;

      _b0_1 = _b0_2 = _b0_3 = _mm512_setzero_ps();

      idx = KernelWidth;

      _aj0 = _mm512_set1_ps(Filter[--idx]);
      _aj1 = _mm512_set1_ps(Filter[idx - 16]);
      _aj2 = _mm512_set1_ps(Filter[idx - 32]);

      _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
      _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);
      _acc0_3 = _mm512_mul_ps(_aj0, _b0_3);

      _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
      _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);
      _acc1_2 = _mm512_mul_ps(_aj1, _b0_2);

      _acc0_1 = _mm512_fmadd_ps(_aj2, _b0_f, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj2, _b0_0, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj2, _b0_1, _acc0_3);

      _roll = _roll_left_1;

      for (int64_t k = 1ll; k < kernelRemainder; k++) {
        _aj0 = _mm512_set1_ps(Filter[--idx]);
        _aj1 = _mm512_set1_ps(Filter[idx - 16]);
        _aj2 = _mm512_set1_ps(Filter[idx - 32]);

        __m512 _b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
        __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
        __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
        __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
        __m512 _bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
        _acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);

        _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
        _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
        _acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);

        _acc0_1 = _mm512_fmadd_ps(_aj2, _bf, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj2, _b0, _acc0_2);
        _acc0_3 = _mm512_fmadd_ps(_aj2, _b1, _acc0_3);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      for (int64_t k = kernelRemainder; k < 16ll; k++) {
        _aj0 = _mm512_set1_ps(Filter[--idx]);
        _aj1 = _mm512_set1_ps(Filter[idx - 16]);

        __m512 _b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
        __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
        __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
        __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
        _acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);

        _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
        _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
        _acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
      _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);
      _acc0_3 = _mm512_add_ps(_acc0_3, _acc1_2);
    }

    // Store
    _mm512_mask_store_ps(pOutput0 + 0, _rightOutputMask1, _acc0_1);
    _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _acc0_2);
    _mm512_mask_store_ps(pOutput0 + 32, _rightOutputMask3, _acc0_3);
  }
}

void MlasConv1DSlidingKernelK64(const MLAS_CONV_PARAMETERS* Parameters,
                           const float* Input,
                           const float* Filter,
                           float* Output)
{
  constexpr size_t Dimensions = 1;
  constexpr size_t WidthShapeIndex = 0;

  const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
  const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

  const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
  const int64_t PaddingLeft = Parameters->Padding[WidthShapeIndex];
  const int64_t PaddingRight = Parameters->Padding[Dimensions + WidthShapeIndex];

  assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
  MLAS_UNREFERENCED_PARAMETER(PaddingRight);

  constexpr size_t widthStep = 64;

  const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

  const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

  const __mmask64 _leftInputMask = (~0ull << paddingLeftShift);

  const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMask);
  const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);
  const __mmask16 _leftInputMask3 = static_cast<__mmask16>(_leftInputMask >> 32);
  const __mmask16 _leftInputMask4 = static_cast<__mmask16>(_leftInputMask >> 48);

  const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

  const int64_t widthIterations = InputWidthBase / widthStep;
  const int64_t widthInputRemainder = InputWidthBase % widthStep;
  const int64_t widthOutputRemainder = OutputWidth % widthStep;

  const int64_t kernelRemainder = KernelWidth - 48;

  const __mmask64 _rightInputMask = ~(~0ull << widthInputRemainder);

  const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
  const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);
  const __mmask16 _rightInputMask3 = static_cast<__mmask16>(_rightInputMask >> 32);
  const __mmask16 _rightInputMask4 = static_cast<__mmask16>(_rightInputMask >> 48);

  const __mmask64 _rightOutputMask = ~(~0ull << widthOutputRemainder);

  const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
  const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);
  const __mmask16 _rightOutputMask3 = static_cast<__mmask16>(_rightOutputMask >> 32);
  const __mmask16 _rightOutputMask4 = static_cast<__mmask16>(_rightOutputMask >> 48);

  const __m512i _roll_left_1 = _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
  const __m512i _one = _mm512_set1_epi32(1);

  float* pOutput0 = Output;

  const float* pInput0 = Input;

  __m512 _b0_d = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift + 0);
  __m512 _b0_e = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);
  __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask3, pInput0 - paddingLeftShift + 32);
  __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask4, pInput0 - paddingLeftShift + 48);

  pInput0 += paddingLeftSeek;

  for (int64_t i = 0; i < widthIterations; i++) {
    int64_t idx = KernelWidth;

    __m512 _b0_1 = _mm512_load_ps(pInput0);
    __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);
    __m512 _b0_3 = _mm512_load_ps(pInput0 + 32);
    __m512 _b0_4 = _mm512_load_ps(pInput0 + 48);

    __m512 _aj0 = _mm512_set1_ps(Filter[--idx]);
    __m512 _aj1 = _mm512_set1_ps(Filter[idx - 16]);
    __m512 _aj2 = _mm512_set1_ps(Filter[idx - 32]);
    __m512 _aj3 = _mm512_set1_ps(Filter[idx - 48]);

    __m512 _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
    __m512 _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);
    __m512 _acc0_3 = _mm512_mul_ps(_aj0, _b0_3);
    __m512 _acc0_4 = _mm512_mul_ps(_aj0, _b0_4);

    __m512 _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
    __m512 _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);
    __m512 _acc1_2 = _mm512_mul_ps(_aj1, _b0_2);
    __m512 _acc1_3 = _mm512_mul_ps(_aj1, _b0_3);

    _acc0_1 = _mm512_fmadd_ps(_aj2, _b0_f, _acc0_1);
    _acc0_2 = _mm512_fmadd_ps(_aj2, _b0_0, _acc0_2);
    _acc0_3 = _mm512_fmadd_ps(_aj2, _b0_1, _acc0_3);
    _acc0_4 = _mm512_fmadd_ps(_aj2, _b0_2, _acc0_4);

    _acc1_0 = _mm512_fmadd_ps(_aj3, _b0_e, _acc1_0);
    _acc1_1 = _mm512_fmadd_ps(_aj3, _b0_f, _acc1_1);
    _acc1_2 = _mm512_fmadd_ps(_aj3, _b0_0, _acc1_2);
    _acc1_3 = _mm512_fmadd_ps(_aj3, _b0_1, _acc1_3);

    __m512i _roll = _roll_left_1;

    for (int64_t k = 1ll; k < kernelRemainder; k++) {
      _aj0 = _mm512_set1_ps(Filter[--idx]);
      _aj1 = _mm512_set1_ps(Filter[idx - 16]);
      _aj2 = _mm512_set1_ps(Filter[idx - 32]);
      _aj3 = _mm512_set1_ps(Filter[idx - 48]);

      __m512 _b4 = _mm512_permutex2var_ps(_b0_3, _roll, _b0_4);
      __m512 _b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
      __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
      __m512 _bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);
      __m512 _be = _mm512_permutex2var_ps(_b0_d, _roll, _b0_e);

      _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);
      _acc0_4 = _mm512_fmadd_ps(_aj0, _b4, _acc0_4);

      _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
      _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
      _acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);
      _acc1_3 = _mm512_fmadd_ps(_aj1, _b3, _acc1_3);

      _acc0_1 = _mm512_fmadd_ps(_aj2, _bf, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj2, _b0, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj2, _b1, _acc0_3);
      _acc0_4 = _mm512_fmadd_ps(_aj2, _b2, _acc0_4);

      _acc1_0 = _mm512_fmadd_ps(_aj3, _be, _acc1_0);
      _acc1_1 = _mm512_fmadd_ps(_aj3, _bf, _acc1_1);
      _acc1_2 = _mm512_fmadd_ps(_aj3, _b0, _acc1_2);
      _acc1_3 = _mm512_fmadd_ps(_aj3, _b1, _acc1_3);

      _roll = _mm512_sub_epi32(_roll, _one);
    }

    for (int64_t k = kernelRemainder; k < 16ll; k++) {
      _aj0 = _mm512_set1_ps(Filter[--idx]);
      _aj1 = _mm512_set1_ps(Filter[idx - 16]);
      _aj2 = _mm512_set1_ps(Filter[idx - 32]);

      __m512 _b4 = _mm512_permutex2var_ps(_b0_3, _roll, _b0_4);
      __m512 _b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
      __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
      __m512 _bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);

      _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);
      _acc0_4 = _mm512_fmadd_ps(_aj0, _b4, _acc0_4);

      _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
      _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
      _acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);
      _acc1_3 = _mm512_fmadd_ps(_aj1, _b3, _acc1_3);

      _acc0_1 = _mm512_fmadd_ps(_aj2, _bf, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj2, _b0, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj2, _b1, _acc0_3);
      _acc0_4 = _mm512_fmadd_ps(_aj2, _b2, _acc0_4);

      _roll = _mm512_sub_epi32(_roll, _one);
    }

    _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
    _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);
    _acc0_3 = _mm512_add_ps(_acc0_3, _acc1_2);
    _acc0_4 = _mm512_add_ps(_acc0_4, _acc1_3);

    _b0_d = _b0_1;
    _b0_e = _b0_2;
    _b0_f = _b0_3;
    _b0_0 = _b0_4;

    // Store
    _mm512_store_ps(pOutput0 + 0, _acc0_1);
    _mm512_store_ps(pOutput0 + 16, _acc0_2);
    _mm512_store_ps(pOutput0 + 32, _acc0_3);
    _mm512_store_ps(pOutput0 + 48, _acc0_4);

    pInput0 += widthStep;
    pOutput0 += widthStep;
  }

  // Right Edge
  {
    __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0 + 0);
    __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);
    __m512 _b0_3 = _mm512_maskz_load_ps(_rightInputMask3, pInput0 + 32);
    __m512 _b0_4 = _mm512_maskz_load_ps(_rightInputMask4, pInput0 + 48);

    int64_t idx = KernelWidth;

    __m512 _aj0 = _mm512_set1_ps(Filter[--idx]);
    __m512 _aj1 = _mm512_set1_ps(Filter[idx - 16]);
    __m512 _aj2 = _mm512_set1_ps(Filter[idx - 32]);
    __m512 _aj3 = _mm512_set1_ps(Filter[idx - 48]);

    __m512 _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
    __m512 _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);
    __m512 _acc0_3 = _mm512_mul_ps(_aj0, _b0_3);
    __m512 _acc0_4 = _mm512_mul_ps(_aj0, _b0_4);

    __m512 _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
    __m512 _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);
    __m512 _acc1_2 = _mm512_mul_ps(_aj1, _b0_2);
    __m512 _acc1_3 = _mm512_mul_ps(_aj1, _b0_3);

    _acc0_1 = _mm512_fmadd_ps(_aj2, _b0_f, _acc0_1);
    _acc0_2 = _mm512_fmadd_ps(_aj2, _b0_0, _acc0_2);
    _acc0_3 = _mm512_fmadd_ps(_aj2, _b0_1, _acc0_3);
    _acc0_4 = _mm512_fmadd_ps(_aj2, _b0_2, _acc0_4);

    _acc1_0 = _mm512_fmadd_ps(_aj3, _b0_e, _acc1_0);
    _acc1_1 = _mm512_fmadd_ps(_aj3, _b0_f, _acc1_1);
    _acc1_2 = _mm512_fmadd_ps(_aj3, _b0_0, _acc1_2);
    _acc1_3 = _mm512_fmadd_ps(_aj3, _b0_1, _acc1_3);

    __m512i _roll = _roll_left_1;

    for (int64_t k = 1ll; k < kernelRemainder; k++) {
      _aj0 = _mm512_set1_ps(Filter[--idx]);
      _aj1 = _mm512_set1_ps(Filter[idx - 16]);
      _aj2 = _mm512_set1_ps(Filter[idx - 32]);
      _aj3 = _mm512_set1_ps(Filter[idx - 48]);

      __m512 _b4 = _mm512_permutex2var_ps(_b0_3, _roll, _b0_4);
      __m512 _b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
      __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
      __m512 _bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);
      __m512 _be = _mm512_permutex2var_ps(_b0_d, _roll, _b0_e);

      _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);
      _acc0_4 = _mm512_fmadd_ps(_aj0, _b4, _acc0_4);

      _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
      _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
      _acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);
      _acc1_3 = _mm512_fmadd_ps(_aj1, _b3, _acc1_3);

      _acc0_1 = _mm512_fmadd_ps(_aj2, _bf, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj2, _b0, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj2, _b1, _acc0_3);
      _acc0_4 = _mm512_fmadd_ps(_aj2, _b2, _acc0_4);

      _acc1_0 = _mm512_fmadd_ps(_aj3, _be, _acc1_0);
      _acc1_1 = _mm512_fmadd_ps(_aj3, _bf, _acc1_1);
      _acc1_2 = _mm512_fmadd_ps(_aj3, _b0, _acc1_2);
      _acc1_3 = _mm512_fmadd_ps(_aj3, _b1, _acc1_3);

      _roll = _mm512_sub_epi32(_roll, _one);
    }

    for (int64_t k = kernelRemainder; k < 16ll; k++) {
      _aj0 = _mm512_set1_ps(Filter[--idx]);
      _aj1 = _mm512_set1_ps(Filter[idx - 16]);
      _aj2 = _mm512_set1_ps(Filter[idx - 32]);

      __m512 _b4 = _mm512_permutex2var_ps(_b0_3, _roll, _b0_4);
      __m512 _b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
      __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
      __m512 _bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);

      _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);
      _acc0_4 = _mm512_fmadd_ps(_aj0, _b4, _acc0_4);

      _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
      _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
      _acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);
      _acc1_3 = _mm512_fmadd_ps(_aj1, _b3, _acc1_3);

      _acc0_1 = _mm512_fmadd_ps(_aj2, _bf, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj2, _b0, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj2, _b1, _acc0_3);
      _acc0_4 = _mm512_fmadd_ps(_aj2, _b2, _acc0_4);

      _roll = _mm512_sub_epi32(_roll, _one);
    }

    _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
    _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);
    _acc0_3 = _mm512_add_ps(_acc0_3, _acc1_2);
    _acc0_4 = _mm512_add_ps(_acc0_4, _acc1_3);

    if (widthOutputRemainder < widthInputRemainder) {
      // Store
      _mm512_store_ps(pOutput0 + 0, _acc0_1);
      _mm512_store_ps(pOutput0 + 16, _acc0_2);
      _mm512_store_ps(pOutput0 + 32, _acc0_3);
      _mm512_store_ps(pOutput0 + 48, _acc0_4);

      pOutput0 += widthStep;

      _b0_d = _b0_1;
      _b0_e = _b0_2;
      _b0_f = _b0_3;
      _b0_0 = _b0_4;

      _b0_1 = _b0_2 = _b0_3 = _b0_4 = _mm512_setzero_ps();

      idx = KernelWidth;

      _aj0 = _mm512_set1_ps(Filter[--idx]);
      _aj1 = _mm512_set1_ps(Filter[idx - 16]);
      _aj2 = _mm512_set1_ps(Filter[idx - 32]);
      _aj3 = _mm512_set1_ps(Filter[idx - 48]);

      _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
      _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);
      _acc0_3 = _mm512_mul_ps(_aj0, _b0_3);
      _acc0_4 = _mm512_mul_ps(_aj0, _b0_4);

      _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
      _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);
      _acc1_2 = _mm512_mul_ps(_aj1, _b0_2);
      _acc1_3 = _mm512_mul_ps(_aj1, _b0_3);

      _acc0_1 = _mm512_fmadd_ps(_aj2, _b0_f, _acc0_1);
      _acc0_2 = _mm512_fmadd_ps(_aj2, _b0_0, _acc0_2);
      _acc0_3 = _mm512_fmadd_ps(_aj2, _b0_1, _acc0_3);
      _acc0_4 = _mm512_fmadd_ps(_aj2, _b0_2, _acc0_4);

      _acc1_0 = _mm512_fmadd_ps(_aj3, _b0_e, _acc1_0);
      _acc1_1 = _mm512_fmadd_ps(_aj3, _b0_f, _acc1_1);
      _acc1_2 = _mm512_fmadd_ps(_aj3, _b0_0, _acc1_2);
      _acc1_3 = _mm512_fmadd_ps(_aj3, _b0_1, _acc1_3);

      _roll = _roll_left_1;

      for (int64_t k = 1ll; k < kernelRemainder; k++) {
        _aj0 = _mm512_set1_ps(Filter[--idx]);
        _aj1 = _mm512_set1_ps(Filter[idx - 16]);
        _aj2 = _mm512_set1_ps(Filter[idx - 32]);
        _aj3 = _mm512_set1_ps(Filter[idx - 48]);

        __m512 _b4 = _mm512_permutex2var_ps(_b0_3, _roll, _b0_4);
        __m512 _b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
        __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
        __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
        __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
        __m512 _bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);
        __m512 _be = _mm512_permutex2var_ps(_b0_d, _roll, _b0_e);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
        _acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);
        _acc0_4 = _mm512_fmadd_ps(_aj0, _b4, _acc0_4);

        _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
        _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
        _acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);
        _acc1_3 = _mm512_fmadd_ps(_aj1, _b3, _acc1_3);

        _acc0_1 = _mm512_fmadd_ps(_aj2, _bf, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj2, _b0, _acc0_2);
        _acc0_3 = _mm512_fmadd_ps(_aj2, _b1, _acc0_3);
        _acc0_4 = _mm512_fmadd_ps(_aj2, _b2, _acc0_4);

        _acc1_0 = _mm512_fmadd_ps(_aj3, _be, _acc1_0);
        _acc1_1 = _mm512_fmadd_ps(_aj3, _bf, _acc1_1);
        _acc1_2 = _mm512_fmadd_ps(_aj3, _b0, _acc1_2);
        _acc1_3 = _mm512_fmadd_ps(_aj3, _b1, _acc1_3);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      for (int64_t k = kernelRemainder; k < 16ll; k++) {
        _aj0 = _mm512_set1_ps(Filter[--idx]);
        _aj1 = _mm512_set1_ps(Filter[idx - 16]);
        _aj2 = _mm512_set1_ps(Filter[idx - 32]);

        __m512 _b4 = _mm512_permutex2var_ps(_b0_3, _roll, _b0_4);
        __m512 _b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
        __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
        __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
        __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
        __m512 _bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
        _acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);
        _acc0_4 = _mm512_fmadd_ps(_aj0, _b4, _acc0_4);

        _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
        _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
        _acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);
        _acc1_3 = _mm512_fmadd_ps(_aj1, _b3, _acc1_3);

        _acc0_1 = _mm512_fmadd_ps(_aj2, _bf, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj2, _b0, _acc0_2);
        _acc0_3 = _mm512_fmadd_ps(_aj2, _b1, _acc0_3);
        _acc0_4 = _mm512_fmadd_ps(_aj2, _b2, _acc0_4);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
      _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);
      _acc0_3 = _mm512_add_ps(_acc0_3, _acc1_2);
      _acc0_4 = _mm512_add_ps(_acc0_4, _acc1_3);
    }

    // Store
    _mm512_mask_store_ps(pOutput0 + 0, _rightOutputMask1, _acc0_1);
    _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _acc0_2);
    _mm512_mask_store_ps(pOutput0 + 32, _rightOutputMask3, _acc0_3);
    _mm512_mask_store_ps(pOutput0 + 48, _rightOutputMask4, _acc0_4);
  }
}


void
MlasConv1DSlidingKernel(const MLAS_CONV_PARAMETERS* Parameters,
                        const float* Input,
                        const float* Filter,
                        float* Output)
{
  constexpr size_t WidthShapeIndex = 0;

  const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];

 if (KernelWidth <= 16) {
    MlasConv1DSlidingKernelK17(Parameters, Input, Filter, Output);
 } else if (KernelWidth <= 32) {
    MlasConv1DSlidingKernelK32(Parameters, Input, Filter, Output);
 } else if (KernelWidth <= 48) {
    MlasConv1DSlidingKernelK48(Parameters, Input, Filter, Output);
 } else if (KernelWidth <= 64) {
    MlasConv1DSlidingKernelK64(Parameters, Input, Filter, Output);
 } else
    return;
}

void MlasConv2DSlidingKernelK17(const MLAS_CONV_PARAMETERS* Parameters,
                           const float* Input,
                           const float* Filter,
                           float* Output)
{
 constexpr size_t Dimensions = 2;
  constexpr size_t HeightShapeIndex = 0;
  constexpr size_t WidthShapeIndex = 1;

  const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
  const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
  const size_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
  const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

  const int64_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
  const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
  const int64_t PaddingTop = Parameters->Padding[HeightShapeIndex];
  const int64_t PaddingLeft = Parameters->Padding[WidthShapeIndex];
  const int64_t PaddingBottom = Parameters->Padding[Dimensions + HeightShapeIndex];
  const int64_t PaddingRight = Parameters->Padding[Dimensions + WidthShapeIndex];

  assert(OutputHeight == (InputHeight - KernelHeight + PaddingTop + PaddingBottom + 1));
  assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
  MLAS_UNREFERENCED_PARAMETER(InputHeight);
  MLAS_UNREFERENCED_PARAMETER(PaddingRight);

  constexpr size_t widthStep = 32;

  const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

  const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

  const __mmask64 _leftInputMask = (~0ull << paddingLeftShift);

  const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);

  const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

  const int64_t widthIterations = InputWidthBase / widthStep;
  const int64_t widthInputRemainder = InputWidthBase % widthStep;
  const int64_t widthOutputRemainder = OutputWidth % widthStep;

  const __mmask64 _rightInputMask = ~(~0ull << widthInputRemainder);

  const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
  const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);

  const __mmask64 _rightOutputMask = ~(~0ull << widthOutputRemainder);

  const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
  const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);

  const __m512i _roll_left_1 = _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
  const __m512i _one = _mm512_set1_epi32(1);

  const float* pInputRow = Input;

  float* pOutputRow = Output;

  // Outer Loop Prologue
  for (int64_t i = 0; i < PaddingTop; i++) {
    __m512 _zero = _mm512_setzero_ps();

    float* pOutput0 = pOutputRow;

    for (int64_t j = 0; j < widthIterations; j++) {
      _mm512_store_ps(pOutput0, _zero);

      _mm512_store_ps(pOutput0 + 16, _zero);

      pOutput0 += widthStep;
    }

    if (widthOutputRemainder < widthInputRemainder) {
      _mm512_store_ps(pOutput0, _zero);

      _mm512_store_ps(pOutput0 + 16, _zero);

      pOutput0 += widthStep;
    }

    _mm512_mask_store_ps(pOutput0, _rightOutputMask1, _zero);

    _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _zero);

    pOutputRow += OutputWidth;
  }

  for (int64_t i = PaddingTop; i < KernelHeight; i++) {
    const float* pInput0 = pInputRow;

    float* pOutput0 = pOutputRow;

    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

    pInput0 += paddingLeftSeek;

    for (int64_t j = 0; j < widthIterations; j++) {
      const float* pFilterRow = Filter;

      __m512 _b0_1 = _mm512_load_ps(pInput0);
      __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

      __m512 _b0 = _b0_1;
      __m512 _b1 = _b0_2;

      __m512i _roll = _roll_left_1;

      __m512 _acc0 = _mm512_setzero_ps();
      __m512 _acc1 = _mm512_setzero_ps();

      for (int64_t l = KernelWidth; l-- > 0;) {
        __m512 _ak = _mm512_set1_ps(pFilterRow[l]);

        _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak, _b1, _acc1);

        _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
        _b1 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      // Store
      _mm512_store_ps(pOutput0, _acc0);
      _mm512_store_ps(pOutput0 + 16, _acc1);

      float* pOutput1 = pOutput0;

      for (int64_t k = i; k-- > 0;) {
        pFilterRow += KernelWidth;

        pOutput1 -= OutputWidth;

        _b0 = _b0_1;
        _b1 = _b0_2;

        _roll = _roll_left_1;

        _acc0 = _mm512_load_ps(pOutput1);
        _acc1 = _mm512_load_ps(pOutput1 + 16);

        for (int64_t l = KernelWidth; l-- > 0;) {
          __m512 _ak = _mm512_set1_ps(pFilterRow[l]);

          _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);
          _acc1 = _mm512_fmadd_ps(_ak, _b1, _acc1);

          _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
          _b1 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        // Store
        _mm512_store_ps(pOutput1, _acc0);
        _mm512_store_ps(pOutput1 + 16, _acc1);
      }

      _b0_0 = _b0_2;

      pInput0 += widthStep;
      pOutput0 += widthStep;
    }

    // Right Edge
    {
      const float* pFilterRow = Filter;

      float* pOutput1 = pOutput0;

      __m512 _zero = _mm512_setzero_ps();

      __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0);
      __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);

      __m512 _b0 = _b0_1;
      __m512 _b1 = _b0_2;

      __m512i _roll = _roll_left_1;

      __m512 _acc0 = _zero;
      __m512 _acc1 = _zero;

      for (int64_t l = KernelWidth; l-- > 0;) {
        __m512 _ak = _mm512_set1_ps(pFilterRow[l]);

        _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak, _b1, _acc1);

        _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
        _b1 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      float* pOutput2 = pOutput1;

      if (widthOutputRemainder < widthInputRemainder) {
        _mm512_store_ps(pOutput2, _acc0);
        _mm512_store_ps(pOutput2 + 16, _acc1);

        pOutput2 += widthStep;

        _b0 = _b1 = _zero;

        _roll = _roll_left_1;

        _acc0 = _acc1 = _zero;

        for (int64_t l = KernelWidth; l-- > 0;) {
          __m512 _ak = _mm512_set1_ps(pFilterRow[l]);

          _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);

          _b0 = _mm512_permutex2var_ps(_b0_2, _roll, _zero);

          _roll = _mm512_sub_epi32(_roll, _one);
        }
      }

      _mm512_mask_store_ps(pOutput2, _rightOutputMask1, _acc0);
      _mm512_mask_store_ps(pOutput2 + 16, _rightOutputMask2, _acc1);

      for (int64_t k = i; k-- > 0;) {
        pFilterRow += KernelWidth;

        pOutput1 -= OutputWidth;

        _b0 = _b0_1;
        _b1 = _b0_2;

        _roll = _roll_left_1;

        _acc0 = _acc1 = _zero;

        for (int64_t l = KernelWidth; l-- > 0;) {
          __m512 _ak = _mm512_set1_ps(pFilterRow[l]);

          _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);
          _acc1 = _mm512_fmadd_ps(_ak, _b1, _acc1);

          _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
          _b1 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        pOutput2 = pOutput1;

        if (widthOutputRemainder < widthInputRemainder) {
          __m512 _acc0_x = _mm512_load_ps(pOutput2);
          __m512 _acc1_x = _mm512_load_ps(pOutput2 + 16);

          _acc0_x = _mm512_add_ps(_acc0_x, _acc0);
          _acc1_x = _mm512_add_ps(_acc0_x, _acc1);

          _mm512_store_ps(pOutput2, _acc0_x);
          _mm512_store_ps(pOutput2 + 16, _acc1_x);

          pOutput2 += widthStep;

          _b0 = _zero;

          _roll = _roll_left_1;

          _acc0 = _zero;
          _acc1 = _zero;

          for (int64_t l = KernelWidth; l-- > 0;) {
            __m512 _ak = _mm512_set1_ps(pFilterRow[l]);

            _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);

            _b0 = _mm512_permutex2var_ps(_b0_1, _roll, _zero);

            _roll = _mm512_sub_epi32(_roll, _one);
          }
        }

        __m512 _acc0_x = _mm512_maskz_load_ps(_rightOutputMask1, pOutput2);
        __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask2, pOutput2 + 16);

        _acc0_x = _mm512_add_ps(_acc0_x, _acc0);
        _acc1_x = _mm512_add_ps(_acc1_x, _acc1);

        _mm512_mask_store_ps(pOutput2, _rightOutputMask1, _acc0_x);
        _mm512_mask_store_ps(pOutput2 + 16, _rightOutputMask2, _acc1_x);
      }
    }

    pInputRow += InputWidth;

    pOutputRow += OutputWidth;
  }

  // Outer loop
  for (size_t i = KernelHeight; i < OutputHeight; i++) {
    const float* pInput0 = pInputRow;

    float* pOutput0 = pOutputRow;

    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

    pInput0 += paddingLeftSeek;

    for (int64_t j = 0; j < widthIterations; j++) {
      const float* pFilterRow = Filter;

      float* pOutput1 = pOutput0;

      __m512 _b0_1 = _mm512_load_ps(pInput0);
      __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

      __m512 _b0 = _b0_1;
      __m512 _b1 = _b0_2;

      __m512i _roll = _roll_left_1;

      __m512 _acc0 = _mm512_setzero_ps();
      __m512 _acc1 = _mm512_setzero_ps();

      for (int64_t l = KernelWidth; l-- > 0;) {
        __m512 _ak = _mm512_set1_ps(pFilterRow[l]);

        _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak, _b1, _acc1);

        _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
        _b1 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      // Store
      _mm512_store_ps(pOutput0, _acc0);
      _mm512_store_ps(pOutput0 + 16, _acc1);

      for (int64_t k = KernelHeight; --k > 0;) {
        pFilterRow += KernelWidth;

        pOutput1 -= OutputWidth;

        _b0 = _b0_1;
        _b1 = _b0_2;

        _roll = _roll_left_1;

        _acc0 = _mm512_load_ps(pOutput1);
        _acc1 = _mm512_load_ps(pOutput1 + 16);

        for (int64_t l = KernelWidth; l-- > 0;) {
          __m512 _ak = _mm512_set1_ps(pFilterRow[l]);

          _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);
          _acc1 = _mm512_fmadd_ps(_ak, _b1, _acc1);

          _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
          _b1 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        // Store
        _mm512_store_ps(pOutput1, _acc0);
        _mm512_store_ps(pOutput1 + 16, _acc1);
      }

      _b0_0 = _b0_2;

      pInput0 += widthStep;
      pOutput0 += widthStep;
    }

    // Right Edge
    {
      const float* pFilterRow = Filter;

      float* pOutput1 = pOutput0;

      __m512 _zero = _mm512_setzero_ps();

      __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0);
      __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);

      __m512 _b0 = _b0_1;
      __m512 _b1 = _b0_2;

      __m512i _roll = _roll_left_1;

      __m512 _acc0 = _zero;
      __m512 _acc1 = _zero;

      for (int64_t l = KernelWidth; l-- > 0;) {
        __m512 _ak = _mm512_set1_ps(pFilterRow[l]);

        _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak, _b1, _acc1);

        _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
        _b1 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      float* pOutput2 = pOutput1;

      if (widthOutputRemainder < widthInputRemainder) {
        _mm512_store_ps(pOutput2, _acc0);
        _mm512_store_ps(pOutput2 + 16, _acc1);

        pOutput2 += widthStep;

        _b0 = _b1 = _zero;

        _roll = _roll_left_1;

        _acc0 = _acc1 = _zero;

        for (int64_t l = KernelWidth; l-- > 0;) {
          __m512 _ak = _mm512_set1_ps(pFilterRow[l]);

          _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);

          _b0 = _mm512_permutex2var_ps(_b0_2, _roll, _zero);

          _roll = _mm512_sub_epi32(_roll, _one);
        }
      }

      _mm512_mask_store_ps(pOutput2, _rightOutputMask1, _acc0);
      _mm512_mask_store_ps(pOutput2 + 16, _rightOutputMask2, _acc1);

      for (int64_t k = KernelHeight; --k > 0;) {
        pFilterRow += KernelWidth;

        pOutput1 -= OutputWidth;

        _b0 = _b0_1;
        _b1 = _b0_2;

        _roll = _roll_left_1;

        _acc0 = _acc1 = _zero;

        for (int64_t l = KernelWidth; l-- > 0;) {
          __m512 _ak = _mm512_set1_ps(pFilterRow[l]);

          _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);
          _acc1 = _mm512_fmadd_ps(_ak, _b1, _acc1);

          _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
          _b1 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        pOutput2 = pOutput1;

        if (widthOutputRemainder < widthInputRemainder) {
          __m512 _acc0_x = _mm512_load_ps(pOutput2);
          __m512 _acc1_x = _mm512_load_ps(pOutput2 + 16);

          _acc0_x = _mm512_add_ps(_acc0_x, _acc0);
          _acc1_x = _mm512_add_ps(_acc0_x, _acc1);

          _mm512_store_ps(pOutput2, _acc0_x);
          _mm512_store_ps(pOutput2 + 16, _acc1_x);

          pOutput2 += widthStep;

          _b0 = _zero;

          _roll = _roll_left_1;

          _acc0 = _zero;
          _acc1 = _zero;

          for (int64_t l = KernelWidth; l-- > 0;) {
            __m512 _ak = _mm512_set1_ps(pFilterRow[l]);

            _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);

            _b0 = _mm512_permutex2var_ps(_b0_1, _roll, _zero);

            _roll = _mm512_sub_epi32(_roll, _one);
          }
        }

        __m512 _acc0_x = _mm512_maskz_load_ps(_rightOutputMask1, pOutput2);
        __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask2, pOutput2 + 16);

        _acc0_x = _mm512_add_ps(_acc0_x, _acc0);
        _acc1_x = _mm512_add_ps(_acc1_x, _acc1);

        _mm512_mask_store_ps(pOutput2, _rightOutputMask1, _acc0_x);
        _mm512_mask_store_ps(pOutput2 + 16, _rightOutputMask2, _acc1_x);
      }
    }

    pInputRow += InputWidth;

    pOutputRow += OutputWidth;
  }

  // Outer Loop Epilogue
  for (int64_t i = 1; i < KernelHeight - PaddingBottom; i++) {
    const float* pInput0 = pInputRow;

    float* pOutput0 = pOutputRow;

    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

    pInput0 += paddingLeftSeek;

    for (int64_t j = 0; j < widthIterations; j++) {
      const float* pFilterRow = Filter + i * KernelWidth;

      float* pOutput1 = pOutput0;

      __m512 _b0_1 = _mm512_load_ps(pInput0);
      __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

      for (int64_t k = KernelHeight; k-- > i;) {
        pOutput1 -= OutputWidth;

        __m512 _b0 = _b0_1;
        __m512 _b1 = _b0_2;

        __m512i _roll = _roll_left_1;

        __m512 _acc0 = _mm512_load_ps(pOutput1);
        __m512 _acc1 = _mm512_load_ps(pOutput1 + 16);

        for (int64_t l = KernelWidth; l-- > 0;) {
          __m512 _ak = _mm512_set1_ps(pFilterRow[l]);

          _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);
          _acc1 = _mm512_fmadd_ps(_ak, _b1, _acc1);

          _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
          _b1 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        // Store
        _mm512_store_ps(pOutput1, _acc0);
        _mm512_store_ps(pOutput1 + 16, _acc1);

        pFilterRow += KernelWidth;
      }

      _b0_0 = _b0_2;

      pInput0 += widthStep;
      pOutput0 += widthStep;
    }

    // Right Edge
    {
      const float* pFilterRow = Filter + i * KernelWidth;

      float* pOutput1 = pOutput0;

      __m512 _zero = _mm512_setzero_ps();

      __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0);
      __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);

      for (int64_t k = KernelHeight; k-- > i;) {
        pOutput1 -= OutputWidth;

        __m512 _b0 = _b0_1;
        __m512 _b1 = _b0_2;

        __m512i _roll = _roll_left_1;

        __m512 _acc0 = _zero;
        __m512 _acc1 = _zero;

        for (int64_t l = KernelWidth; l-- > 0;) {
          __m512 _ak = _mm512_set1_ps(pFilterRow[l]);

          _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);
          _acc1 = _mm512_fmadd_ps(_ak, _b1, _acc1);

          _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
          _b1 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        float* pOutput2 = pOutput1;

        if (widthOutputRemainder < widthInputRemainder) {
          __m512 _acc0_x = _mm512_load_ps(pOutput2);
          __m512 _acc1_x = _mm512_load_ps(pOutput2 + 16);

          _acc0_x = _mm512_add_ps(_acc0_x, _acc0);
          _acc1_x = _mm512_add_ps(_acc0_x, _acc1);

          _mm512_store_ps(pOutput2, _acc0_x);
          _mm512_store_ps(pOutput2 + 16, _acc1_x);

          _b0 = _zero;

          _roll = _roll_left_1;

          _acc0 = _zero;

          for (int64_t l = KernelWidth; l-- > 0;) {
            __m512 _ak = _mm512_set1_ps(pFilterRow[k]);

            _acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);

            _b0 = _mm512_permutex2var_ps(_b0_1, _roll, _zero);

            _roll = _mm512_sub_epi32(_roll, _one);
          }
        }

        __m512 _acc0_x = _mm512_maskz_load_ps(_rightOutputMask1, pOutput2);
        __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask2, pOutput2 + 16);

        _acc0_x = _mm512_add_ps(_acc0_x, _acc0);
        _acc1_x = _mm512_add_ps(_acc1_x, _acc1);

        _mm512_mask_store_ps(pOutput2, _rightOutputMask1, _acc0_x);
        _mm512_mask_store_ps(pOutput2 + 16, _rightOutputMask2, _acc1_x);

        pFilterRow += KernelWidth;
      }
    }

    pInputRow += InputWidth;
  }
}

void MlasConv2DSlidingKernelK3x3(const MLAS_CONV_PARAMETERS* Parameters,
                            const float* Input,
                            const float* Filter,
                            float* Output)
{
  constexpr size_t Dimensions = 2;
  constexpr size_t HeightShapeIndex = 0;
  constexpr size_t WidthShapeIndex = 1;

  const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
  const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
  const size_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
  const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

  const int64_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
  const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
  const int64_t PaddingTop = Parameters->Padding[HeightShapeIndex];
  const int64_t PaddingLeft = Parameters->Padding[WidthShapeIndex];
  const int64_t PaddingBottom = Parameters->Padding[Dimensions + HeightShapeIndex];
  const int64_t PaddingRight = Parameters->Padding[Dimensions + WidthShapeIndex];

  assert(KernelHeight == 3);
  assert(KernelWidth == 3);
  assert(OutputHeight == (InputHeight - KernelHeight + PaddingTop + PaddingBottom + 1));
  assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
  MLAS_UNREFERENCED_PARAMETER(InputHeight);
  MLAS_UNREFERENCED_PARAMETER(PaddingRight);

  constexpr size_t heightStep = 4;

  const int64_t OutputHeightBase = OutputHeight - KernelHeight;

  const size_t heightIterations = OutputHeightBase / heightStep;
  const size_t heightRemainder = OutputHeightBase % heightStep;

  constexpr size_t widthStep = 16;

  const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

  const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

  const __mmask16 _leftInputMask = static_cast<__mmask16>(~0 << paddingLeftShift);

  const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

  const int64_t widthIterations = InputWidthBase / widthStep;
  const int64_t widthInputRemainder = InputWidthBase % widthStep;
  const int64_t widthOutputRemainder = OutputWidth % widthStep;

  const __mmask16 _rightInputMask = static_cast<__mmask16>(~(~0 << widthInputRemainder));
  const __mmask16 _rightOutputMask = static_cast<__mmask16>(~(~0 << widthOutputRemainder));

  const __m512 _a0 = _mm512_set1_ps(Filter[0]);
  const __m512 _a1 = _mm512_set1_ps(Filter[1]);
  const __m512 _a2 = _mm512_set1_ps(Filter[2]);
  const __m512 _a3 = _mm512_set1_ps(Filter[3]);
  const __m512 _a4 = _mm512_set1_ps(Filter[4]);
  const __m512 _a5 = _mm512_set1_ps(Filter[5]);
  const __m512 _a6 = _mm512_set1_ps(Filter[6]);
  const __m512 _a7 = _mm512_set1_ps(Filter[7]);
  const __m512 _a8 = _mm512_set1_ps(Filter[8]);

  const __m512i _roll_left_1 = _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
  const __m512i _roll_left_2 = _mm512_set_epi32(29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14);

  float* pOutputRow = Output;

  const float* pInputRow = Input;

  // Outer Loop Prologue
  for (int64_t i = 0; i < PaddingTop; i++) {
    __m512 _zero = _mm512_setzero_ps();

    float* pOutput0 = pOutputRow;

    for (int64_t j = 0; j < widthIterations; j++) {
      _mm512_store_ps(pOutput0, _zero);

      pOutput0 += widthStep;
    }

    if (widthOutputRemainder < widthInputRemainder) {
      _mm512_store_ps(pOutput0, _zero);

      pOutput0 += widthStep;
    }

    _mm512_mask_store_ps(pOutput0, _rightOutputMask, _zero);

    pOutputRow += OutputWidth;
  }

  for (int64_t i = PaddingTop; i < KernelHeight; i++) {
    const float* pInput0 = pInputRow;

    float* pOutput0 = pOutputRow;

    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);

    pInput0 += paddingLeftSeek;

    for (int64_t j = 0; j < widthIterations; j++) {
      const float* pFilterRow = Filter;

      __m512 _b0_1 = _mm512_load_ps(pInput0);

      __m512 _aj2 = _mm512_set1_ps(pFilterRow[2]);
      __m512 _acc0 = _mm512_mul_ps(_aj2, _b0_1);

      __m512 _aj1 = _mm512_set1_ps(pFilterRow[1]);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
      _acc0 = _mm512_fmadd_ps(_aj1, _b1, _acc0);

      __m512 _aj0 = _mm512_set1_ps(pFilterRow[0]);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
      _acc0 = _mm512_fmadd_ps(_aj0, _b2, _acc0);

      // Store
      _mm512_store_ps(pOutput0, _acc0);

      float* pOutput1 = pOutput0;

      for (int64_t k = i; k-- > 0;) {
        pFilterRow += KernelWidth;

        pOutput1 -= OutputWidth;

        _acc0 = _mm512_load_ps(pOutput1);

        _aj2 = _mm512_set1_ps(pFilterRow[2]);
        _acc0 = _mm512_fmadd_ps(_aj2, _b0_1, _acc0);

        _aj1 = _mm512_set1_ps(pFilterRow[1]);
        _acc0 = _mm512_fmadd_ps(_aj1, _b1, _acc0);

        _aj0 = _mm512_set1_ps(pFilterRow[0]);
        _acc0 = _mm512_fmadd_ps(_aj0, _b2, _acc0);

        // Store
        _mm512_store_ps(pOutput1, _acc0);
      }

      _b0_0 = _b0_1;

      pInput0 += widthStep;
      pOutput0 += widthStep;
    }

    // Right Edge
    {
      const float* pFilterRow = Filter;

      float* pOutput1 = pOutput0;

      __m512 _zero = _mm512_setzero_ps();

      __m512 _b1_1 = _zero, _b1_2 = _zero;

      __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);

      __m512 _aj2 = _mm512_set1_ps(pFilterRow[2]);
      __m512 _acc0 = _mm512_mul_ps(_aj2, _b0_1);

      __m512 _aj1 = _mm512_set1_ps(pFilterRow[1]);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
      _acc0 = _mm512_fmadd_ps(_aj1, _b1, _acc0);

      __m512 _aj0 = _mm512_set1_ps(pFilterRow[0]);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
      _acc0 = _mm512_fmadd_ps(_aj0, _b2, _acc0);

      float* pOutput2 = pOutput1;

      if (widthOutputRemainder < widthInputRemainder) {
        _mm512_store_ps(pOutput2, _acc0);

        pOutput2 += widthStep;

        _b1_1 = _mm512_permutex2var_ps(_b0_1, _roll_left_1, _zero);
        _acc0 = _mm512_mul_ps(_aj1, _b1_1);

        _b1_2 = _mm512_permutex2var_ps(_b0_1, _roll_left_2, _zero);
        _acc0 = _mm512_fmadd_ps(_aj0, _b1_2, _acc0);
      }

      _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc0);

      for (int64_t k = i; k-- > 0;) {
        pFilterRow += KernelWidth;

        pOutput1 -= OutputWidth;

        _aj2 = _mm512_set1_ps(pFilterRow[2]);
        _acc0 = _mm512_mul_ps(_aj2, _b0_1);

        _aj1 = _mm512_set1_ps(pFilterRow[1]);
        _acc0 = _mm512_fmadd_ps(_aj1, _b1, _acc0);

        _aj0 = _mm512_set1_ps(pFilterRow[0]);
        _acc0 = _mm512_fmadd_ps(_aj0, _b2, _acc0);

        pOutput2 = pOutput1;

        if (widthOutputRemainder < widthInputRemainder) {
          __m512 _acc1 = _mm512_load_ps(pOutput2);

          _acc1 = _mm512_add_ps(_acc1, _acc0);

          _mm512_store_ps(pOutput2, _acc1);

          pOutput2 += widthStep;

          _acc0 = _mm512_mul_ps(_aj1, _b1_1);

          _acc0 = _mm512_fmadd_ps(_aj0, _b1_2, _acc0);
        }

        __m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput2);

        _acc1 = _mm512_add_ps(_acc1, _acc0);

        _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc1);
      }
    }

    pInputRow += InputWidth;

    pOutputRow += OutputWidth;
  }

  // Outer Loop
  for (size_t i = 0; i < heightIterations; i++) {
    const float* pInput0 = pInputRow;
    const float* pInput1 = pInput0 + InputWidth;
    const float* pInput2 = pInput1 + InputWidth;
    const float* pInput3 = pInput2 + InputWidth;

    float* pOutput0 = pOutputRow;
    float* pOutput1 = pOutput0 + OutputWidth;
    float* pOutput2 = pOutput1 + OutputWidth;
    float* pOutput3 = pOutput2 + OutputWidth;
    float* pOutput_1 = pOutput0 - OutputWidth;
    float* pOutput_2 = pOutput_1 - OutputWidth;

    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);
    __m512 _b1_0 = _mm512_maskz_load_ps(_leftInputMask, pInput1 - paddingLeftShift);
    __m512 _b2_0 = _mm512_maskz_load_ps(_leftInputMask, pInput2 - paddingLeftShift);
    __m512 _b3_0 = _mm512_maskz_load_ps(_leftInputMask, pInput3 - paddingLeftShift);

    pInput0 += paddingLeftSeek;
    pInput1 += paddingLeftSeek;
    pInput2 += paddingLeftSeek;
    pInput3 += paddingLeftSeek;

    for (int64_t j = 0; j < widthIterations; j++) {
      __m512 _acce = _mm512_load_ps(pOutput_2);
      __m512 _accf = _mm512_load_ps(pOutput_1);

      __m512 _b0_1 = _mm512_load_ps(pInput0);
      __m512 _b1_1 = _mm512_load_ps(pInput1);
      __m512 _b2_1 = _mm512_load_ps(pInput2);
      __m512 _b3_1 = _mm512_load_ps(pInput3);

      _acce = _mm512_fmadd_ps(_a8, _b0_1, _acce);
      _accf = _mm512_fmadd_ps(_a5, _b0_1, _accf);
      __m512 _acc0 = _mm512_mul_ps(_a2, _b0_1);

      _accf = _mm512_fmadd_ps(_a8, _b1_1, _accf);
      _acc0 = _mm512_fmadd_ps(_a5, _b1_1, _acc0);
      __m512 _acc1 = _mm512_mul_ps(_a2, _b1_1);

      _acc0 = _mm512_fmadd_ps(_a8, _b2_1, _acc0);
      _acc1 = _mm512_fmadd_ps(_a5, _b2_1, _acc1);
      __m512 _acc2 = _mm512_mul_ps(_a2, _b2_1);

      _acc1 = _mm512_fmadd_ps(_a8, _b3_1, _acc1);
      _acc2 = _mm512_fmadd_ps(_a5, _b3_1, _acc2);
      __m512 _acc3 = _mm512_mul_ps(_a2, _b3_1);

      __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
      __m512 _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_1, _b1_1);
      __m512 _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_1, _b2_1);
      __m512 _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_1, _b3_1);

      _acce = _mm512_fmadd_ps(_a7, _b0, _acce);
      _accf = _mm512_fmadd_ps(_a4, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_a1, _b0, _acc0);

      _accf = _mm512_fmadd_ps(_a7, _b1, _accf);
      _acc0 = _mm512_fmadd_ps(_a4, _b1, _acc0);
      _acc1 = _mm512_fmadd_ps(_a1, _b1, _acc1);

      _acc0 = _mm512_fmadd_ps(_a7, _b2, _acc0);
      _acc1 = _mm512_fmadd_ps(_a4, _b2, _acc1);
      _acc2 = _mm512_fmadd_ps(_a1, _b2, _acc2);

      _acc1 = _mm512_fmadd_ps(_a7, _b3, _acc1);
      _acc2 = _mm512_fmadd_ps(_a4, _b3, _acc2);
      _acc3 = _mm512_fmadd_ps(_a1, _b3, _acc3);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
      _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
      _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
      _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

      _acce = _mm512_fmadd_ps(_a6, _b0, _acce);
      _accf = _mm512_fmadd_ps(_a3, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_a0, _b0, _acc0);

      _accf = _mm512_fmadd_ps(_a6, _b1, _accf);
      _acc0 = _mm512_fmadd_ps(_a3, _b1, _acc0);
      _acc1 = _mm512_fmadd_ps(_a0, _b1, _acc1);

      _acc0 = _mm512_fmadd_ps(_a6, _b2, _acc0);
      _acc1 = _mm512_fmadd_ps(_a3, _b2, _acc1);
      _acc2 = _mm512_fmadd_ps(_a0, _b2, _acc2);

      _acc1 = _mm512_fmadd_ps(_a6, _b3, _acc1);
      _acc2 = _mm512_fmadd_ps(_a3, _b3, _acc2);
      _acc3 = _mm512_fmadd_ps(_a0, _b3, _acc3);

      // Store
      _mm512_store_ps(pOutput_2, _acce);
      _mm512_store_ps(pOutput_1, _accf);

      _mm512_store_ps(pOutput0, _acc0);
      _mm512_store_ps(pOutput1, _acc1);
      _mm512_store_ps(pOutput2, _acc2);
      _mm512_store_ps(pOutput3, _acc3);

      _b0_0 = _b0_1;
      _b1_0 = _b1_1;
      _b2_0 = _b2_1;
      _b3_0 = _b3_1;

      pInput0 += widthStep;
      pInput1 += widthStep;
      pInput2 += widthStep;
      pInput3 += widthStep;

      pOutput_2 += widthStep;
      pOutput_1 += widthStep;
      pOutput0 += widthStep;
      pOutput1 += widthStep;
      pOutput2 += widthStep;
      pOutput3 += widthStep;
    }

    // Right Edge
    {
      __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);
      __m512 _b1_1 = _mm512_maskz_load_ps(_rightInputMask, pInput1);
      __m512 _b2_1 = _mm512_maskz_load_ps(_rightInputMask, pInput2);
      __m512 _b3_1 = _mm512_maskz_load_ps(_rightInputMask, pInput3);

      __m512 _acce = _mm512_mul_ps(_a8, _b0_1);
      __m512 _accf = _mm512_mul_ps(_a5, _b0_1);
      __m512 _acc0 = _mm512_mul_ps(_a2, _b0_1);

      _accf = _mm512_fmadd_ps(_a8, _b1_1, _accf);
      _acc0 = _mm512_fmadd_ps(_a5, _b1_1, _acc0);
      __m512 _acc1 = _mm512_mul_ps(_a2, _b1_1);

      _acc0 = _mm512_fmadd_ps(_a8, _b2_1, _acc0);
      _acc1 = _mm512_fmadd_ps(_a5, _b2_1, _acc1);
      __m512 _acc2 = _mm512_mul_ps(_a2, _b2_1);

      _acc1 = _mm512_fmadd_ps(_a8, _b3_1, _acc1);
      _acc2 = _mm512_fmadd_ps(_a5, _b3_1, _acc2);
      __m512 _acc3 = _mm512_mul_ps(_a2, _b3_1);

      __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
      __m512 _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_1, _b1_1);
      __m512 _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_1, _b2_1);
      __m512 _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_1, _b3_1);

      _acce = _mm512_fmadd_ps(_a7, _b0, _acce);
      _accf = _mm512_fmadd_ps(_a4, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_a1, _b0, _acc0);

      _accf = _mm512_fmadd_ps(_a7, _b1, _accf);
      _acc0 = _mm512_fmadd_ps(_a4, _b1, _acc0);
      _acc1 = _mm512_fmadd_ps(_a1, _b1, _acc1);

      _acc0 = _mm512_fmadd_ps(_a7, _b2, _acc0);
      _acc1 = _mm512_fmadd_ps(_a4, _b2, _acc1);
      _acc2 = _mm512_fmadd_ps(_a1, _b2, _acc2);

      _acc1 = _mm512_fmadd_ps(_a7, _b3, _acc1);
      _acc2 = _mm512_fmadd_ps(_a4, _b3, _acc2);
      _acc3 = _mm512_fmadd_ps(_a1, _b3, _acc3);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
      _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
      _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
      _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

      _acce = _mm512_fmadd_ps(_a6, _b0, _acce);
      _accf = _mm512_fmadd_ps(_a3, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_a0, _b0, _acc0);

      _accf = _mm512_fmadd_ps(_a6, _b1, _accf);
      _acc0 = _mm512_fmadd_ps(_a3, _b1, _acc0);
      _acc1 = _mm512_fmadd_ps(_a0, _b1, _acc1);

      _acc0 = _mm512_fmadd_ps(_a6, _b2, _acc0);
      _acc1 = _mm512_fmadd_ps(_a3, _b2, _acc1);
      _acc2 = _mm512_fmadd_ps(_a0, _b2, _acc2);

      _acc1 = _mm512_fmadd_ps(_a6, _b3, _acc1);
      _acc2 = _mm512_fmadd_ps(_a3, _b3, _acc2);
      _acc3 = _mm512_fmadd_ps(_a0, _b3, _acc3);

      if (widthOutputRemainder < widthInputRemainder) {
        __m512 _zero = _mm512_setzero_ps();

        __m512 _acc2_x = _mm512_load_ps(pOutput_2);
        __m512 _acc1_x = _mm512_load_ps(pOutput_1);

        _acc2_x = _mm512_add_ps(_acc2_x, _acce);
        _acc1_x = _mm512_add_ps(_acc1_x, _accf);

        _mm512_store_ps(pOutput_2, _acc2_x);
        _mm512_store_ps(pOutput_1, _acc1_x);

        _mm512_store_ps(pOutput0, _acc0);
        _mm512_store_ps(pOutput1, _acc1);
        _mm512_store_ps(pOutput2, _acc2);
        _mm512_store_ps(pOutput3, _acc3);

        pOutput_2 += widthStep;
        pOutput_1 += widthStep;
        pOutput0 += widthStep;
        pOutput1 += widthStep;
        pOutput2 += widthStep;
        pOutput3 += widthStep;

        _b1 = _mm512_permutex2var_ps(_b0_1, _roll_left_1, _zero);

        _acc0 = _mm512_mul_ps(_a1, _b1);
        _accf = _mm512_mul_ps(_a4, _b1);
        _acce = _mm512_mul_ps(_a7, _b1);

        _b2 = _mm512_permutex2var_ps(_b0_1, _roll_left_2, _zero);

        _acc0 = _mm512_fmadd_ps(_a0, _b2, _acc0);
        _accf = _mm512_fmadd_ps(_a3, _b2, _accf);
        _acce = _mm512_fmadd_ps(_a6, _b2, _acce);

        _b0 = _mm512_permutex2var_ps(_b0_1, _roll_left_1, _zero);
        _b1 = _mm512_permutex2var_ps(_b1_1, _roll_left_1, _zero);
        _b2 = _mm512_permutex2var_ps(_b2_1, _roll_left_1, _zero);
        _b3 = _mm512_permutex2var_ps(_b3_1, _roll_left_1, _zero);

        _acce = _mm512_mul_ps(_a7, _b0);
        _accf = _mm512_mul_ps(_a4, _b0);
        _acc0 = _mm512_mul_ps(_a1, _b0);

        _accf = _mm512_fmadd_ps(_a7, _b1, _accf);
        _acc0 = _mm512_fmadd_ps(_a4, _b1, _acc0);
        _acc1 = _mm512_mul_ps(_a1, _b1);

        _acc0 = _mm512_fmadd_ps(_a7, _b2, _acc0);
        _acc1 = _mm512_fmadd_ps(_a4, _b2, _acc1);
        _acc2 = _mm512_mul_ps(_a1, _b2);

        _acc1 = _mm512_fmadd_ps(_a7, _b3, _acc1);
        _acc2 = _mm512_fmadd_ps(_a4, _b3, _acc2);
        _acc3 = _mm512_mul_ps(_a1, _b3);

        _b0 = _mm512_permutex2var_ps(_b0_1, _roll_left_2, _zero);
        _b1 = _mm512_permutex2var_ps(_b1_1, _roll_left_2, _zero);
        _b2 = _mm512_permutex2var_ps(_b2_1, _roll_left_2, _zero);
        _b3 = _mm512_permutex2var_ps(_b3_1, _roll_left_2, _zero);

        _acce = _mm512_fmadd_ps(_a6, _b0, _acce);
        _accf = _mm512_fmadd_ps(_a3, _b0, _accf);
        _acc0 = _mm512_fmadd_ps(_a0, _b0, _acc0);

        _accf = _mm512_fmadd_ps(_a6, _b1, _accf);
        _acc0 = _mm512_fmadd_ps(_a3, _b1, _acc0);
        _acc1 = _mm512_fmadd_ps(_a0, _b1, _acc1);

        _acc0 = _mm512_fmadd_ps(_a6, _b2, _acc0);
        _acc1 = _mm512_fmadd_ps(_a3, _b2, _acc1);
        _acc2 = _mm512_fmadd_ps(_a0, _b2, _acc2);

        _acc1 = _mm512_fmadd_ps(_a6, _b3, _acc1);
        _acc2 = _mm512_fmadd_ps(_a3, _b3, _acc2);
        _acc3 = _mm512_fmadd_ps(_a0, _b3, _acc3);
      }

      __m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_2);
      __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_1);

      _acc2_x = _mm512_add_ps(_acc2_x, _acce);
      _acc1_x = _mm512_add_ps(_acc1_x, _accf);

      _mm512_mask_store_ps(pOutput_2, _rightOutputMask, _acc2_x);
      _mm512_mask_store_ps(pOutput_1, _rightOutputMask, _acc1_x);

      _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);
      _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
      _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc2);
      _mm512_mask_store_ps(pOutput3, _rightOutputMask, _acc3);
    }

    pInputRow += heightStep * InputWidth;

    pOutputRow += heightStep * OutputWidth;
  }

  // Outer Short Tail
  for (size_t i = 0; i < heightRemainder; i++) {
    const float* pInput0 = pInputRow;

    float* pOutput0 = pOutputRow;
    float* pOutput_1 = pOutput0 - OutputWidth;
    float* pOutput_2 = pOutput_1 - OutputWidth;

    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);

    pInput0 += paddingLeftSeek;

    for (int64_t j = 0; j < widthIterations; j++) {
      __m512 _acce = _mm512_load_ps(pOutput_2);
      __m512 _accf = _mm512_load_ps(pOutput_1);

      __m512 _b0_1 = _mm512_load_ps(pInput0);

      __m512 _acc0 = _mm512_mul_ps(_a2, _b0_1);
      _accf = _mm512_fmadd_ps(_a5, _b0_1, _accf);
      _acce = _mm512_fmadd_ps(_a8, _b0_1, _acce);

      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

      _acc0 = _mm512_fmadd_ps(_a1, _b1, _acc0);
      _accf = _mm512_fmadd_ps(_a4, _b1, _accf);
      _acce = _mm512_fmadd_ps(_a7, _b1, _acce);

      __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

      _acc0 = _mm512_fmadd_ps(_a0, _b2, _acc0);
      _accf = _mm512_fmadd_ps(_a3, _b2, _accf);
      _acce = _mm512_fmadd_ps(_a6, _b2, _acce);

      // Store
      _mm512_store_ps(pOutput_2, _acce);
      _mm512_store_ps(pOutput_1, _accf);

      _mm512_store_ps(pOutput0, _acc0);

      _b0_0 = _b0_1;

      pInput0 += widthStep;

      pOutput_2 += widthStep;
      pOutput_1 += widthStep;
      pOutput0 += widthStep;
    }

    // Right Edge
    {
      __m512 _zero = _mm512_setzero_ps();

      __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);

      __m512 _acc0 = _mm512_mul_ps(_a2, _b0_1);
      __m512 _accf = _mm512_mul_ps(_a5, _b0_1);
      __m512 _acce = _mm512_mul_ps(_a8, _b0_1);

      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

      _acc0 = _mm512_fmadd_ps(_a1, _b1, _acc0);
      _accf = _mm512_fmadd_ps(_a4, _b1, _accf);
      _acce = _mm512_fmadd_ps(_a7, _b1, _acce);

      __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

      _acc0 = _mm512_fmadd_ps(_a0, _b2, _acc0);
      _accf = _mm512_fmadd_ps(_a3, _b2, _accf);
      _acce = _mm512_fmadd_ps(_a6, _b2, _acce);

      if (widthOutputRemainder < widthInputRemainder) {
        __m512 _acc2_x = _mm512_load_ps(pOutput_2);
        __m512 _acc1_x = _mm512_load_ps(pOutput_1);

        _acc2_x = _mm512_add_ps(_acc2_x, _acce);
        _acc1_x = _mm512_add_ps(_acc1_x, _accf);

        _mm512_store_ps(pOutput_2, _acc2_x);
        _mm512_store_ps(pOutput_1, _acc1_x);

        _mm512_store_ps(pOutput0, _acc0);

        pOutput_2 += widthStep;
        pOutput_1 += widthStep;
        pOutput0 += widthStep;

        _b1 = _mm512_permutex2var_ps(_b0_1, _roll_left_1, _zero);

        _acc0 = _mm512_mul_ps(_a1, _b1);
        _accf = _mm512_mul_ps(_a4, _b1);
        _acce = _mm512_mul_ps(_a7, _b1);

        _b2 = _mm512_permutex2var_ps(_b0_1, _roll_left_2, _zero);

        _acc0 = _mm512_fmadd_ps(_a0, _b2, _acc0);
        _accf = _mm512_fmadd_ps(_a3, _b2, _accf);
        _acce = _mm512_fmadd_ps(_a6, _b2, _acce);
      }

      __m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_2);
      __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_1);

      _acc2_x = _mm512_add_ps(_acc2_x, _acce);
      _acc1_x = _mm512_add_ps(_acc1_x, _accf);

      _mm512_mask_store_ps(pOutput_2, _rightOutputMask, _acc2_x);
      _mm512_mask_store_ps(pOutput_1, _rightOutputMask, _acc1_x);

      _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);
    }

    pInputRow += InputWidth;

    pOutputRow += OutputWidth;
  }

  // Outer Loop Epilogue
  for (int64_t i = 1; i < KernelHeight - PaddingBottom; i++) {
    const float* pInput0 = pInputRow;

    float* pOutput0 = pOutputRow;

    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);

    pInput0 += paddingLeftSeek;

    for (int64_t j = 0; j < widthIterations; j++) {
      const float* pFilterRow = Filter + i * KernelWidth;

      float* pOutput1 = pOutput0;

      __m512 _b0_1 = _mm512_load_ps(pInput0);

      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

      for (int64_t k = KernelHeight; k-- > i;) {
        pOutput1 -= OutputWidth;

        __m512 _acc0 = _mm512_load_ps(pOutput1);

        __m512 _aj2 = _mm512_set1_ps(pFilterRow[2]);
        _acc0 = _mm512_fmadd_ps(_aj2, _b0_1, _acc0);

        __m512 _aj1 = _mm512_set1_ps(pFilterRow[1]);
        _acc0 = _mm512_fmadd_ps(_aj1, _b1, _acc0);

        __m512 _aj0 = _mm512_set1_ps(pFilterRow[0]);
        _acc0 = _mm512_fmadd_ps(_aj0, _b2, _acc0);

        // Store
        _mm512_store_ps(pOutput1, _acc0);

        pFilterRow += KernelWidth;
      }

      _b0_0 = _b0_1;

      pInput0 += widthStep;
      pOutput0 += widthStep;
    }

    // Right Edge
    {
      const float* pFilterRow = Filter + i * KernelWidth;

      float* pOutput1 = pOutput0;

      __m512 _zero = _mm512_setzero_ps();

      __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);

      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

      __m512 _b1_1 = _zero, _b1_2 = _zero;

      if (widthOutputRemainder < widthInputRemainder) {
        _b1_1 = _mm512_permutex2var_ps(_b0_1, _roll_left_1, _zero);
        _b1_2 = _mm512_permutex2var_ps(_b0_1, _roll_left_2, _zero);
      }

      for (int64_t k = KernelHeight; k-- > i;) {
        pOutput1 -= OutputWidth;

        __m512 _aj2 = _mm512_set1_ps(pFilterRow[2]);
        __m512 _acc0 = _mm512_mul_ps(_aj2, _b0_1);

        __m512 _aj1 = _mm512_set1_ps(pFilterRow[1]);
        _acc0 = _mm512_fmadd_ps(_aj1, _b1, _acc0);

        __m512 _aj0 = _mm512_set1_ps(pFilterRow[0]);
        _acc0 = _mm512_fmadd_ps(_aj0, _b2, _acc0);

        float* pOutput2 = pOutput1;

        if (widthOutputRemainder < widthInputRemainder) {
          __m512 _acc1 = _mm512_load_ps(pOutput2);

          _acc1 = _mm512_add_ps(_acc1, _acc0);

          _mm512_store_ps(pOutput2, _acc1);

          pOutput2 += widthStep;

          _acc0 = _mm512_mul_ps(_aj1, _b1_1);

          _acc0 = _mm512_fmadd_ps(_aj0, _b1_2, _acc0);
        }

        __m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput2);

        _acc1 = _mm512_add_ps(_acc1, _acc0);

        _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc1);

        pFilterRow += KernelWidth;
      }
    }

    pInputRow += InputWidth;
  }
}

void MlasConv2DSlidingKernelK5x5(const MLAS_CONV_PARAMETERS* Parameters,
                            const float* Input,
                            const float* Filter,
                            float* Output)
{
  constexpr size_t Dimensions = 2;
  constexpr size_t HeightShapeIndex = 0;
  constexpr size_t WidthShapeIndex = 1;

  const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
  const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
  const size_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
  const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

  const int64_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
  const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
  const int64_t PaddingTop = Parameters->Padding[HeightShapeIndex];
  const int64_t PaddingLeft = Parameters->Padding[WidthShapeIndex];
  const int64_t PaddingBottom = Parameters->Padding[Dimensions + HeightShapeIndex];
  const int64_t PaddingRight = Parameters->Padding[Dimensions + WidthShapeIndex];

  assert(KernelHeight == 5);
  assert(KernelWidth == 5);
  assert(OutputHeight == (InputHeight - KernelHeight + PaddingTop + PaddingBottom + 1));
  assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
  MLAS_UNREFERENCED_PARAMETER(InputHeight);
  MLAS_UNREFERENCED_PARAMETER(PaddingRight);

  constexpr size_t heightStep = 4;

  const int64_t OutputHeightBase = OutputHeight - KernelHeight;

  const size_t heightIterations = OutputHeightBase / heightStep;
  const size_t heightRemainder = OutputHeightBase % heightStep;

  constexpr size_t widthStep = 16;

  const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

  const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

  const __mmask16 _leftInputMask = static_cast<__mmask16>(~0 << paddingLeftShift);

  const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

  const int64_t widthIterations = InputWidthBase / widthStep;
  const int64_t widthInputRemainder = InputWidthBase % widthStep;
  const int64_t widthOutputRemainder = OutputWidth % widthStep;

  const __mmask16 _rightInputMask = static_cast<__mmask16>(~(~0 << widthInputRemainder));
  const __mmask16 _rightOutputMask = static_cast<__mmask16>(~(~0 << widthOutputRemainder));

  const __m512i _roll_left_1 = _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
  const __m512i _roll_left_2 = _mm512_set_epi32(29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14);
  const __m512i _roll_left_3 = _mm512_set_epi32(28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13);
  const __m512i _roll_left_4 = _mm512_set_epi32(27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12);

  float* pOutputRow = Output;

  const float* pInputRow = Input;

  // Outer Loop Prologue
  for (int64_t i = 0; i < PaddingTop; i++) {
    __m512 _zero = _mm512_setzero_ps();

    float* pOutput0 = pOutputRow;

    for (int64_t j = 0; j < widthIterations; j++) {
      _mm512_store_ps(pOutput0, _zero);

      pOutput0 += widthStep;
    }

    if (widthOutputRemainder < widthInputRemainder) {
      _mm512_store_ps(pOutput0, _zero);

      pOutput0 += widthStep;
    }

    _mm512_mask_store_ps(pOutput0, _rightOutputMask, _zero);

    pOutputRow += OutputWidth;
  }

  for (int64_t i = PaddingTop; i < KernelHeight; i++) {
    const float* pInput0 = pInputRow;

    float* pOutput0 = pOutputRow;

    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);

    pInput0 += paddingLeftSeek;

    for (int64_t j = 0; j < widthIterations; j++) {
      const float* pFilterRow = Filter;

      float* pOutput1 = pOutput0;

      __m512 _b0_1 = _mm512_load_ps(pInput0);

      __m512 _aj4 = _mm512_set1_ps(pFilterRow[4]);
      __m512 _acc0 = _mm512_mul_ps(_aj4, _b0_1);

      __m512 _aj3 = _mm512_set1_ps(pFilterRow[3]);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
      _acc0 = _mm512_fmadd_ps(_aj3, _b1, _acc0);

      __m512 _aj2 = _mm512_set1_ps(pFilterRow[2]);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
      _acc0 = _mm512_fmadd_ps(_aj2, _b2, _acc0);

      __m512 _aj1 = _mm512_set1_ps(pFilterRow[1]);
      __m512 _b3 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
      _acc0 = _mm512_fmadd_ps(_aj1, _b3, _acc0);

      __m512 _aj0 = _mm512_set1_ps(pFilterRow[0]);
      __m512 _b4 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);
      _acc0 = _mm512_fmadd_ps(_aj0, _b4, _acc0);

      // Store
      _mm512_store_ps(pOutput1, _acc0);

      for (int64_t k = i; k-- > 0;) {
        pFilterRow += KernelWidth;

        pOutput1 -= OutputWidth;

        _acc0 = _mm512_load_ps(pOutput1);

        _aj4 = _mm512_set1_ps(pFilterRow[4]);
        _acc0 = _mm512_fmadd_ps(_aj4, _b0_1, _acc0);

        _aj3 = _mm512_set1_ps(pFilterRow[3]);
        _acc0 = _mm512_fmadd_ps(_aj3, _b1, _acc0);

        _aj2 = _mm512_set1_ps(pFilterRow[2]);
        _acc0 = _mm512_fmadd_ps(_aj2, _b2, _acc0);

        _aj1 = _mm512_set1_ps(pFilterRow[1]);
        _acc0 = _mm512_fmadd_ps(_aj1, _b3, _acc0);

        _aj0 = _mm512_set1_ps(pFilterRow[0]);
        _acc0 = _mm512_fmadd_ps(_aj0, _b4, _acc0);

        // Store
        _mm512_store_ps(pOutput1, _acc0);
      }

      _b0_0 = _b0_1;

      pInput0 += widthStep;
      pOutput0 += widthStep;
    }

    // Right Edge
    {
      const float* pFilterRow = Filter;

      float* pOutput1 = pOutput0;

      __m512 _zero = _mm512_setzero_ps();

      __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);

      __m512 _aj4 = _mm512_set1_ps(pFilterRow[4]);
      __m512 _acc0 = _mm512_mul_ps(_aj4, _b0_1);

      __m512 _aj3 = _mm512_set1_ps(pFilterRow[3]);
      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
      _acc0 = _mm512_fmadd_ps(_aj3, _b1, _acc0);

      __m512 _aj2 = _mm512_set1_ps(pFilterRow[2]);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
      _acc0 = _mm512_fmadd_ps(_aj2, _b2, _acc0);

      __m512 _aj1 = _mm512_set1_ps(pFilterRow[1]);
      __m512 _b3 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
      _acc0 = _mm512_fmadd_ps(_aj1, _b3, _acc0);

      __m512 _aj0 = _mm512_set1_ps(pFilterRow[0]);
      __m512 _b4 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);
      _acc0 = _mm512_fmadd_ps(_aj0, _b4, _acc0);

      float* pOutput2 = pOutput1;

      if (widthOutputRemainder < widthInputRemainder) {
        _mm512_store_ps(pOutput2, _acc0);

        pOutput2 += widthStep;

        __m512 _b1_1 = _mm512_permutex2var_ps(_b0_1, _roll_left_1, _zero);
        _acc0 = _mm512_mul_ps(_aj3, _b1_1);

        __m512 _b1_2 = _mm512_permutex2var_ps(_b0_1, _roll_left_2, _zero);
        _acc0 = _mm512_fmadd_ps(_aj2, _b1_2, _acc0);

        __m512 _b1_3 = _mm512_permutex2var_ps(_b0_1, _roll_left_3, _zero);
        _acc0 = _mm512_fmadd_ps(_aj1, _b1_3, _acc0);

        __m512 _b1_4 = _mm512_permutex2var_ps(_b0_1, _roll_left_4, _zero);
        _acc0 = _mm512_fmadd_ps(_aj0, _b1_4, _acc0);
      }

      _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc0);

      for (int64_t k = i; k-- > 0;) {
        pFilterRow += KernelWidth;

        pOutput1 -= OutputWidth;

        _aj4 = _mm512_set1_ps(pFilterRow[4]);
        _acc0 = _mm512_mul_ps(_aj4, _b0_1);

        _aj3 = _mm512_set1_ps(pFilterRow[3]);
        _acc0 = _mm512_fmadd_ps(_aj3, _b1, _acc0);

        _aj2 = _mm512_set1_ps(pFilterRow[2]);
        _acc0 = _mm512_fmadd_ps(_aj2, _b2, _acc0);

        _aj1 = _mm512_set1_ps(pFilterRow[1]);
        _acc0 = _mm512_fmadd_ps(_aj1, _b3, _acc0);

        _aj0 = _mm512_set1_ps(pFilterRow[0]);
        _acc0 = _mm512_fmadd_ps(_aj0, _b4, _acc0);

        pOutput2 = pOutput1;

        if (widthOutputRemainder < widthInputRemainder) {
          __m512 _acc1 = _mm512_load_ps(pOutput2);

          _acc1 = _mm512_add_ps(_acc1, _acc0);

          _mm512_store_ps(pOutput2, _acc1);

          pOutput2 += widthStep;

          __m512 _b1_1 = _mm512_permutex2var_ps(_b0_1, _roll_left_1, _zero);
          _acc0 = _mm512_mul_ps(_aj3, _b1_1);

          __m512 _b1_2 = _mm512_permutex2var_ps(_b0_1, _roll_left_2, _zero);
          _acc0 = _mm512_fmadd_ps(_aj2, _b1_2, _acc0);

          __m512 _b1_3 = _mm512_permutex2var_ps(_b0_1, _roll_left_3, _zero);
          _acc0 = _mm512_fmadd_ps(_aj1, _b1_3, _acc0);

          __m512 _b1_4 = _mm512_permutex2var_ps(_b0_1, _roll_left_4, _zero);
          _acc0 = _mm512_fmadd_ps(_aj0, _b1_4, _acc0);
        }

        __m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput2);

        _acc1 = _mm512_add_ps(_acc1, _acc0);

        _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc1);
      }
    }

    pInputRow += InputWidth;

    pOutputRow += OutputWidth;
  }

  // Outer loop
  for (size_t i = 0; i < heightIterations; i++) {
    const float* pInput0 = pInputRow;
    const float* pInput1 = pInput0 + InputWidth;
    const float* pInput2 = pInput1 + InputWidth;
    const float* pInput3 = pInput2 + InputWidth;

    float* pOutput0 = pOutputRow;
    float* pOutput1 = pOutput0 + OutputWidth;
    float* pOutput2 = pOutput1 + OutputWidth;
    float* pOutput3 = pOutput2 + OutputWidth;
    float* pOutput_1 = pOutput0 - OutputWidth;
    float* pOutput_2 = pOutput_1 - OutputWidth;
    float* pOutput_3 = pOutput_2 - OutputWidth;
    float* pOutput_4 = pOutput_3 - OutputWidth;

    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);
    __m512 _b1_0 = _mm512_maskz_load_ps(_leftInputMask, pInput1 - paddingLeftShift);
    __m512 _b2_0 = _mm512_maskz_load_ps(_leftInputMask, pInput2 - paddingLeftShift);
    __m512 _b3_0 = _mm512_maskz_load_ps(_leftInputMask, pInput3 - paddingLeftShift);

    pInput0 += paddingLeftSeek;
    pInput1 += paddingLeftSeek;
    pInput2 += paddingLeftSeek;
    pInput3 += paddingLeftSeek;

    for (int64_t j = 0; j < widthIterations; j++) {
      __m512 _b0_1 = _mm512_load_ps(pInput0);
      __m512 _b1_1 = _mm512_load_ps(pInput1);
      __m512 _b2_1 = _mm512_load_ps(pInput2);
      __m512 _b3_1 = _mm512_load_ps(pInput3);

      __m512 _accc = _mm512_load_ps(pOutput_4);
      __m512 _accd = _mm512_load_ps(pOutput_3);
      __m512 _acce = _mm512_load_ps(pOutput_2);
      __m512 _accf = _mm512_load_ps(pOutput_1);

      __m512 _ak4 = _mm512_set1_ps(Filter[24]);
      __m512 _ak3 = _mm512_set1_ps(Filter[19]);
      __m512 _ak2 = _mm512_set1_ps(Filter[14]);
      __m512 _ak1 = _mm512_set1_ps(Filter[9]);
      __m512 _ak0 = _mm512_set1_ps(Filter[4]);

      _accc = _mm512_fmadd_ps(_ak4, _b0_1, _accc);
      _accd = _mm512_fmadd_ps(_ak3, _b0_1, _accd);
      _acce = _mm512_fmadd_ps(_ak2, _b0_1, _acce);
      _accf = _mm512_fmadd_ps(_ak1, _b0_1, _accf);
      __m512 _acc0 = _mm512_mul_ps(_ak0, _b0_1);

      _accd = _mm512_fmadd_ps(_ak4, _b1_1, _accd);
      _acce = _mm512_fmadd_ps(_ak3, _b1_1, _acce);
      _accf = _mm512_fmadd_ps(_ak2, _b1_1, _accf);
      _acc0 = _mm512_fmadd_ps(_ak1, _b1_1, _acc0);
      __m512 _acc1 = _mm512_mul_ps(_ak0, _b1_1);

      _acce = _mm512_fmadd_ps(_ak4, _b2_1, _acce);
      _accf = _mm512_fmadd_ps(_ak3, _b2_1, _accf);
      _acc0 = _mm512_fmadd_ps(_ak2, _b2_1, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak1, _b2_1, _acc1);
      __m512 _acc2 = _mm512_mul_ps(_ak0, _b2_1);

      _accf = _mm512_fmadd_ps(_ak4, _b3_1, _accf);
      _acc0 = _mm512_fmadd_ps(_ak3, _b3_1, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak2, _b3_1, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak1, _b3_1, _acc2);
      __m512 _acc3 = _mm512_mul_ps(_ak0, _b3_1);

      __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
      __m512 _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_1, _b1_1);
      __m512 _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_1, _b2_1);
      __m512 _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_1, _b3_1);

      _ak4 = _mm512_set1_ps(Filter[23]);
      _ak3 = _mm512_set1_ps(Filter[18]);
      _ak2 = _mm512_set1_ps(Filter[13]);
      _ak1 = _mm512_set1_ps(Filter[8]);
      _ak0 = _mm512_set1_ps(Filter[3]);

      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

      _accd = _mm512_fmadd_ps(_ak4, _b1, _accd);
      _acce = _mm512_fmadd_ps(_ak3, _b1, _acce);
      _accf = _mm512_fmadd_ps(_ak2, _b1, _accf);
      _acc0 = _mm512_fmadd_ps(_ak1, _b1, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak0, _b1, _acc1);

      _acce = _mm512_fmadd_ps(_ak4, _b2, _acce);
      _accf = _mm512_fmadd_ps(_ak3, _b2, _accf);
      _acc0 = _mm512_fmadd_ps(_ak2, _b2, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak1, _b2, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak0, _b2, _acc2);

      _accf = _mm512_fmadd_ps(_ak4, _b3, _accf);
      _acc0 = _mm512_fmadd_ps(_ak3, _b3, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak2, _b3, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak1, _b3, _acc2);
      _acc3 = _mm512_fmadd_ps(_ak0, _b3, _acc3);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
      _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
      _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
      _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

      _ak4 = _mm512_set1_ps(Filter[22]);
      _ak3 = _mm512_set1_ps(Filter[17]);
      _ak2 = _mm512_set1_ps(Filter[12]);
      _ak1 = _mm512_set1_ps(Filter[7]);
      _ak0 = _mm512_set1_ps(Filter[2]);

      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

      _accd = _mm512_fmadd_ps(_ak4, _b1, _accd);
      _acce = _mm512_fmadd_ps(_ak3, _b1, _acce);
      _accf = _mm512_fmadd_ps(_ak2, _b1, _accf);
      _acc0 = _mm512_fmadd_ps(_ak1, _b1, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak0, _b1, _acc1);

      _acce = _mm512_fmadd_ps(_ak4, _b2, _acce);
      _accf = _mm512_fmadd_ps(_ak3, _b2, _accf);
      _acc0 = _mm512_fmadd_ps(_ak2, _b2, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak1, _b2, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak0, _b2, _acc2);

      _accf = _mm512_fmadd_ps(_ak4, _b3, _accf);
      _acc0 = _mm512_fmadd_ps(_ak3, _b3, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak2, _b3, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak1, _b3, _acc2);
      _acc3 = _mm512_fmadd_ps(_ak0, _b3, _acc3);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
      _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_3, _b1_1);
      _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_3, _b2_1);
      _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_3, _b3_1);

      _ak4 = _mm512_set1_ps(Filter[21]);
      _ak3 = _mm512_set1_ps(Filter[16]);
      _ak2 = _mm512_set1_ps(Filter[11]);
      _ak1 = _mm512_set1_ps(Filter[6]);
      _ak0 = _mm512_set1_ps(Filter[1]);

      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

      _accd = _mm512_fmadd_ps(_ak4, _b1, _accd);
      _acce = _mm512_fmadd_ps(_ak3, _b1, _acce);
      _accf = _mm512_fmadd_ps(_ak2, _b1, _accf);
      _acc0 = _mm512_fmadd_ps(_ak1, _b1, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak0, _b1, _acc1);

      _acce = _mm512_fmadd_ps(_ak4, _b2, _acce);
      _accf = _mm512_fmadd_ps(_ak3, _b2, _accf);
      _acc0 = _mm512_fmadd_ps(_ak2, _b2, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak1, _b2, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak0, _b2, _acc2);

      _accf = _mm512_fmadd_ps(_ak4, _b3, _accf);
      _acc0 = _mm512_fmadd_ps(_ak3, _b3, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak2, _b3, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak1, _b3, _acc2);
      _acc3 = _mm512_fmadd_ps(_ak0, _b3, _acc3);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);
      _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_4, _b1_1);
      _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_4, _b2_1);
      _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_4, _b3_1);

      _ak4 = _mm512_set1_ps(Filter[20]);
      _ak3 = _mm512_set1_ps(Filter[15]);
      _ak2 = _mm512_set1_ps(Filter[10]);
      _ak1 = _mm512_set1_ps(Filter[5]);
      _ak0 = _mm512_set1_ps(Filter[0]);

      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

      _accd = _mm512_fmadd_ps(_ak4, _b1, _accd);
      _acce = _mm512_fmadd_ps(_ak3, _b1, _acce);
      _accf = _mm512_fmadd_ps(_ak2, _b1, _accf);
      _acc0 = _mm512_fmadd_ps(_ak1, _b1, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak0, _b1, _acc1);

      _acce = _mm512_fmadd_ps(_ak4, _b2, _acce);
      _accf = _mm512_fmadd_ps(_ak3, _b2, _accf);
      _acc0 = _mm512_fmadd_ps(_ak2, _b2, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak1, _b2, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak0, _b2, _acc2);

      _accf = _mm512_fmadd_ps(_ak4, _b3, _accf);
      _acc0 = _mm512_fmadd_ps(_ak3, _b3, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak2, _b3, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak1, _b3, _acc2);
      _acc3 = _mm512_fmadd_ps(_ak0, _b3, _acc3);

      // Store
      _mm512_store_ps(pOutput_4, _accc);
      _mm512_store_ps(pOutput_3, _accd);
      _mm512_store_ps(pOutput_2, _acce);
      _mm512_store_ps(pOutput_1, _accf);

      _mm512_store_ps(pOutput0, _acc0);
      _mm512_store_ps(pOutput1, _acc1);
      _mm512_store_ps(pOutput2, _acc2);
      _mm512_store_ps(pOutput3, _acc3);

      _b0_0 = _b0_1;
      _b1_0 = _b1_1;
      _b2_0 = _b2_1;
      _b3_0 = _b3_1;

      pInput0 += widthStep;
      pInput1 += widthStep;
      pInput2 += widthStep;
      pInput3 += widthStep;

      pOutput_4 += widthStep;
      pOutput_3 += widthStep;
      pOutput_2 += widthStep;
      pOutput_1 += widthStep;
      pOutput0 += widthStep;
      pOutput1 += widthStep;
      pOutput2 += widthStep;
      pOutput3 += widthStep;
    }

    // Right Edge
    {
      __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);
      __m512 _b1_1 = _mm512_maskz_load_ps(_rightInputMask, pInput1);
      __m512 _b2_1 = _mm512_maskz_load_ps(_rightInputMask, pInput2);
      __m512 _b3_1 = _mm512_maskz_load_ps(_rightInputMask, pInput3);

      __m512 _ak0 = _mm512_set1_ps(Filter[4]);
      __m512 _ak1 = _mm512_set1_ps(Filter[9]);
      __m512 _ak2 = _mm512_set1_ps(Filter[14]);
      __m512 _ak3 = _mm512_set1_ps(Filter[19]);
      __m512 _ak4 = _mm512_set1_ps(Filter[24]);

      __m512 _accc = _mm512_mul_ps(_ak4, _b0_1);
      __m512 _accd = _mm512_mul_ps(_ak3, _b0_1);
      __m512 _acce = _mm512_mul_ps(_ak2, _b0_1);
      __m512 _accf = _mm512_mul_ps(_ak1, _b0_1);
      __m512 _acc0 = _mm512_mul_ps(_ak0, _b0_1);

      _accd = _mm512_fmadd_ps(_ak4, _b1_1, _accd);
      _acce = _mm512_fmadd_ps(_ak3, _b1_1, _acce);
      _accf = _mm512_fmadd_ps(_ak2, _b1_1, _accf);
      _acc0 = _mm512_fmadd_ps(_ak1, _b1_1, _acc0);
      __m512 _acc1 = _mm512_mul_ps(_ak0, _b1_1);

      _acce = _mm512_fmadd_ps(_ak4, _b2_1, _acce);
      _accf = _mm512_fmadd_ps(_ak3, _b2_1, _accf);
      _acc0 = _mm512_fmadd_ps(_ak2, _b2_1, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak1, _b2_1, _acc1);
      __m512 _acc2 = _mm512_mul_ps(_ak0, _b2_1);

      _accf = _mm512_fmadd_ps(_ak4, _b3_1, _accf);
      _acc0 = _mm512_fmadd_ps(_ak3, _b3_1, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak2, _b3_1, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak1, _b3_1, _acc2);
      __m512 _acc3 = _mm512_mul_ps(_ak0, _b3_1);

      __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
      __m512 _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_1, _b1_1);
      __m512 _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_1, _b2_1);
      __m512 _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_1, _b3_1);

      _ak4 = _mm512_set1_ps(Filter[23]);
      _ak3 = _mm512_set1_ps(Filter[18]);
      _ak2 = _mm512_set1_ps(Filter[13]);
      _ak1 = _mm512_set1_ps(Filter[8]);
      _ak0 = _mm512_set1_ps(Filter[3]);

      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

      _accd = _mm512_fmadd_ps(_ak4, _b1, _accd);
      _acce = _mm512_fmadd_ps(_ak3, _b1, _acce);
      _accf = _mm512_fmadd_ps(_ak2, _b1, _accf);
      _acc0 = _mm512_fmadd_ps(_ak1, _b1, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak0, _b1, _acc1);

      _acce = _mm512_fmadd_ps(_ak4, _b2, _acce);
      _accf = _mm512_fmadd_ps(_ak3, _b2, _accf);
      _acc0 = _mm512_fmadd_ps(_ak2, _b2, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak1, _b2, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak0, _b2, _acc2);

      _accf = _mm512_fmadd_ps(_ak4, _b3, _accf);
      _acc0 = _mm512_fmadd_ps(_ak3, _b3, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak2, _b3, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak1, _b3, _acc2);
      _acc3 = _mm512_fmadd_ps(_ak0, _b3, _acc3);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
      _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
      _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
      _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

      _ak4 = _mm512_set1_ps(Filter[22]);
      _ak3 = _mm512_set1_ps(Filter[17]);
      _ak2 = _mm512_set1_ps(Filter[12]);
      _ak1 = _mm512_set1_ps(Filter[7]);
      _ak0 = _mm512_set1_ps(Filter[2]);

      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

      _accd = _mm512_fmadd_ps(_ak4, _b1, _accd);
      _acce = _mm512_fmadd_ps(_ak3, _b1, _acce);
      _accf = _mm512_fmadd_ps(_ak2, _b1, _accf);
      _acc0 = _mm512_fmadd_ps(_ak1, _b1, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak0, _b1, _acc1);

      _acce = _mm512_fmadd_ps(_ak4, _b2, _acce);
      _accf = _mm512_fmadd_ps(_ak3, _b2, _accf);
      _acc0 = _mm512_fmadd_ps(_ak2, _b2, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak1, _b2, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak0, _b2, _acc2);

      _accf = _mm512_fmadd_ps(_ak4, _b3, _accf);
      _acc0 = _mm512_fmadd_ps(_ak3, _b3, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak2, _b3, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak1, _b3, _acc2);
      _acc3 = _mm512_fmadd_ps(_ak0, _b3, _acc3);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
      _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_3, _b1_1);
      _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_3, _b2_1);
      _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_3, _b3_1);

      _ak4 = _mm512_set1_ps(Filter[21]);
      _ak3 = _mm512_set1_ps(Filter[16]);
      _ak2 = _mm512_set1_ps(Filter[11]);
      _ak1 = _mm512_set1_ps(Filter[6]);
      _ak0 = _mm512_set1_ps(Filter[1]);

      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

      _accd = _mm512_fmadd_ps(_ak4, _b1, _accd);
      _acce = _mm512_fmadd_ps(_ak3, _b1, _acce);
      _accf = _mm512_fmadd_ps(_ak2, _b1, _accf);
      _acc0 = _mm512_fmadd_ps(_ak1, _b1, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak0, _b1, _acc1);

      _acce = _mm512_fmadd_ps(_ak4, _b2, _acce);
      _accf = _mm512_fmadd_ps(_ak3, _b2, _accf);
      _acc0 = _mm512_fmadd_ps(_ak2, _b2, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak1, _b2, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak0, _b2, _acc2);

      _accf = _mm512_fmadd_ps(_ak4, _b3, _accf);
      _acc0 = _mm512_fmadd_ps(_ak3, _b3, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak2, _b3, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak1, _b3, _acc2);
      _acc3 = _mm512_fmadd_ps(_ak0, _b3, _acc3);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);
      _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_4, _b1_1);
      _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_4, _b2_1);
      _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_4, _b3_1);

      _ak4 = _mm512_set1_ps(Filter[20]);
      _ak3 = _mm512_set1_ps(Filter[15]);
      _ak2 = _mm512_set1_ps(Filter[10]);
      _ak1 = _mm512_set1_ps(Filter[5]);
      _ak0 = _mm512_set1_ps(Filter[0]);

      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

      _accd = _mm512_fmadd_ps(_ak4, _b1, _accd);
      _acce = _mm512_fmadd_ps(_ak3, _b1, _acce);
      _accf = _mm512_fmadd_ps(_ak2, _b1, _accf);
      _acc0 = _mm512_fmadd_ps(_ak1, _b1, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak0, _b1, _acc1);

      _acce = _mm512_fmadd_ps(_ak4, _b2, _acce);
      _accf = _mm512_fmadd_ps(_ak3, _b2, _accf);
      _acc0 = _mm512_fmadd_ps(_ak2, _b2, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak1, _b2, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak0, _b2, _acc2);

      _accf = _mm512_fmadd_ps(_ak4, _b3, _accf);
      _acc0 = _mm512_fmadd_ps(_ak3, _b3, _acc0);
      _acc1 = _mm512_fmadd_ps(_ak2, _b3, _acc1);
      _acc2 = _mm512_fmadd_ps(_ak1, _b3, _acc2);
      _acc3 = _mm512_fmadd_ps(_ak0, _b3, _acc3);

      if (widthOutputRemainder < widthInputRemainder) {
        __m512 _zero = _mm512_setzero_ps();

        __m512 _acc4_x = _mm512_load_ps(pOutput_4);
        __m512 _acc3_x = _mm512_load_ps(pOutput_3);
        __m512 _acc2_x = _mm512_load_ps(pOutput_2);
        __m512 _acc1_x = _mm512_load_ps(pOutput_1);

        _acc4_x = _mm512_add_ps(_acc4_x, _accc);
        _acc3_x = _mm512_add_ps(_acc3_x, _accd);
        _acc2_x = _mm512_add_ps(_acc2_x, _acce);
        _acc1_x = _mm512_add_ps(_acc1_x, _accf);

        _mm512_store_ps(pOutput_4, _acc4_x);
        _mm512_store_ps(pOutput_3, _acc3_x);
        _mm512_store_ps(pOutput_2, _acc2_x);
        _mm512_store_ps(pOutput_1, _acc1_x);

        _mm512_store_ps(pOutput0, _acc0);
        _mm512_store_ps(pOutput1, _acc1);
        _mm512_store_ps(pOutput2, _acc2);
        _mm512_store_ps(pOutput3, _acc3);

        pOutput_4 += widthStep;
        pOutput_3 += widthStep;
        pOutput_2 += widthStep;
        pOutput_1 += widthStep;
        pOutput0 += widthStep;

        _b0 = _mm512_permutex2var_ps(_b0_1, _roll_left_1, _zero);
        _b1 = _mm512_permutex2var_ps(_b1_1, _roll_left_1, _zero);
        _b2 = _mm512_permutex2var_ps(_b2_1, _roll_left_1, _zero);
        _b3 = _mm512_permutex2var_ps(_b3_1, _roll_left_1, _zero);

        _ak0 = _mm512_set1_ps(Filter[3]);
        _ak1 = _mm512_set1_ps(Filter[8]);
        _ak2 = _mm512_set1_ps(Filter[13]);
        _ak3 = _mm512_set1_ps(Filter[18]);
        _ak4 = _mm512_set1_ps(Filter[23]);

        _accc = _mm512_mul_ps(_ak4, _b0);
        _accd = _mm512_mul_ps(_ak3, _b0);
        _acce = _mm512_mul_ps(_ak2, _b0);
        _accf = _mm512_mul_ps(_ak1, _b0);
        _acc0 = _mm512_mul_ps(_ak0, _b0);

        _accd = _mm512_fmadd_ps(_ak4, _b1, _accd);
        _acce = _mm512_fmadd_ps(_ak3, _b1, _acce);
        _accf = _mm512_fmadd_ps(_ak2, _b1, _accf);
        _acc0 = _mm512_fmadd_ps(_ak1, _b1, _acc0);
        _acc1 = _mm512_mul_ps(_ak0, _b1);

        _acce = _mm512_fmadd_ps(_ak4, _b2, _acce);
        _accf = _mm512_fmadd_ps(_ak3, _b2, _accf);
        _acc0 = _mm512_fmadd_ps(_ak2, _b2, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak1, _b2, _acc1);
        _acc2 = _mm512_mul_ps(_ak0, _b2);

        _accf = _mm512_fmadd_ps(_ak4, _b3, _accf);
        _acc0 = _mm512_fmadd_ps(_ak3, _b3, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak2, _b3, _acc1);
        _acc2 = _mm512_fmadd_ps(_ak1, _b3, _acc2);
        _acc3 = _mm512_mul_ps(_ak0, _b3);

        _b0 = _mm512_permutex2var_ps(_b0_1, _roll_left_2, _zero);
        _b1 = _mm512_permutex2var_ps(_b1_1, _roll_left_2, _zero);
        _b2 = _mm512_permutex2var_ps(_b2_1, _roll_left_2, _zero);
        _b3 = _mm512_permutex2var_ps(_b3_1, _roll_left_2, _zero);

        _ak0 = _mm512_set1_ps(Filter[2]);
        _ak1 = _mm512_set1_ps(Filter[7]);
        _ak2 = _mm512_set1_ps(Filter[12]);
        _ak3 = _mm512_set1_ps(Filter[17]);
        _ak4 = _mm512_set1_ps(Filter[22]);

        _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
        _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
        _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
        _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
        _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

        _accd = _mm512_fmadd_ps(_ak4, _b1, _accd);
        _acce = _mm512_fmadd_ps(_ak3, _b1, _acce);
        _accf = _mm512_fmadd_ps(_ak2, _b1, _accf);
        _acc0 = _mm512_fmadd_ps(_ak1, _b1, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak0, _b1, _acc1);

        _acce = _mm512_fmadd_ps(_ak4, _b2, _acce);
        _accf = _mm512_fmadd_ps(_ak3, _b2, _accf);
        _acc0 = _mm512_fmadd_ps(_ak2, _b2, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak1, _b2, _acc1);
        _acc2 = _mm512_fmadd_ps(_ak0, _b2, _acc2);

        _accf = _mm512_fmadd_ps(_ak4, _b3, _accf);
        _acc0 = _mm512_fmadd_ps(_ak3, _b3, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak2, _b3, _acc1);
        _acc2 = _mm512_fmadd_ps(_ak1, _b3, _acc2);
        _acc3 = _mm512_fmadd_ps(_ak0, _b3, _acc3);

        _b0 = _mm512_permutex2var_ps(_b0_1, _roll_left_3, _zero);
        _b1 = _mm512_permutex2var_ps(_b1_1, _roll_left_3, _zero);
        _b2 = _mm512_permutex2var_ps(_b2_1, _roll_left_3, _zero);
        _b3 = _mm512_permutex2var_ps(_b3_1, _roll_left_3, _zero);

        _ak0 = _mm512_set1_ps(Filter[1]);
        _ak1 = _mm512_set1_ps(Filter[6]);
        _ak2 = _mm512_set1_ps(Filter[11]);
        _ak3 = _mm512_set1_ps(Filter[16]);
        _ak4 = _mm512_set1_ps(Filter[21]);

        _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
        _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
        _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
        _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
        _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

        _accd = _mm512_fmadd_ps(_ak4, _b1, _accd);
        _acce = _mm512_fmadd_ps(_ak3, _b1, _acce);
        _accf = _mm512_fmadd_ps(_ak2, _b1, _accf);
        _acc0 = _mm512_fmadd_ps(_ak1, _b1, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak0, _b1, _acc1);

        _acce = _mm512_fmadd_ps(_ak4, _b2, _acce);
        _accf = _mm512_fmadd_ps(_ak3, _b2, _accf);
        _acc0 = _mm512_fmadd_ps(_ak2, _b2, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak1, _b2, _acc1);
        _acc2 = _mm512_fmadd_ps(_ak0, _b2, _acc2);

        _accf = _mm512_fmadd_ps(_ak4, _b3, _accf);
        _acc0 = _mm512_fmadd_ps(_ak3, _b3, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak2, _b3, _acc1);
        _acc2 = _mm512_fmadd_ps(_ak1, _b3, _acc2);
        _acc3 = _mm512_fmadd_ps(_ak0, _b3, _acc3);

        _b0 = _mm512_permutex2var_ps(_b0_1, _roll_left_4, _zero);
        _b1 = _mm512_permutex2var_ps(_b1_1, _roll_left_4, _zero);
        _b2 = _mm512_permutex2var_ps(_b2_1, _roll_left_4, _zero);
        _b3 = _mm512_permutex2var_ps(_b3_1, _roll_left_4, _zero);

        _ak0 = _mm512_set1_ps(Filter[0]);
        _ak1 = _mm512_set1_ps(Filter[5]);
        _ak2 = _mm512_set1_ps(Filter[10]);
        _ak3 = _mm512_set1_ps(Filter[15]);
        _ak4 = _mm512_set1_ps(Filter[20]);

        _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
        _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
        _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
        _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
        _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

        _accd = _mm512_fmadd_ps(_ak4, _b1, _accd);
        _acce = _mm512_fmadd_ps(_ak3, _b1, _acce);
        _accf = _mm512_fmadd_ps(_ak2, _b1, _accf);
        _acc0 = _mm512_fmadd_ps(_ak1, _b1, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak0, _b1, _acc1);

        _acce = _mm512_fmadd_ps(_ak4, _b2, _acce);
        _accf = _mm512_fmadd_ps(_ak3, _b2, _accf);
        _acc0 = _mm512_fmadd_ps(_ak2, _b2, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak1, _b2, _acc1);
        _acc2 = _mm512_fmadd_ps(_ak0, _b2, _acc2);

        _accf = _mm512_fmadd_ps(_ak4, _b3, _accf);
        _acc0 = _mm512_fmadd_ps(_ak3, _b3, _acc0);
        _acc1 = _mm512_fmadd_ps(_ak2, _b3, _acc1);
        _acc2 = _mm512_fmadd_ps(_ak1, _b3, _acc2);
        _acc3 = _mm512_fmadd_ps(_ak0, _b3, _acc3);
      }

      __m512 _acc4_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_4);
      __m512 _acc3_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_3);
      __m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_2);
      __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_1);

      _acc4_x = _mm512_add_ps(_acc4_x, _accc);
      _acc3_x = _mm512_add_ps(_acc3_x, _accd);
      _acc2_x = _mm512_add_ps(_acc2_x, _acce);
      _acc1_x = _mm512_add_ps(_acc1_x, _accf);

      _mm512_mask_store_ps(pOutput_4, _rightOutputMask, _acc4_x);
      _mm512_mask_store_ps(pOutput_3, _rightOutputMask, _acc3_x);
      _mm512_mask_store_ps(pOutput_2, _rightOutputMask, _acc2_x);
      _mm512_mask_store_ps(pOutput_1, _rightOutputMask, _acc1_x);

      _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);
      _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
      _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc2);
      _mm512_mask_store_ps(pOutput3, _rightOutputMask, _acc3);
    }

    pInputRow += heightStep * InputWidth;

    pOutputRow += heightStep * OutputWidth;
  }

  // Outer Short Tail
  for (size_t i = 0; i < heightRemainder; i++) {
    const float* pInput0 = pInputRow;

    float* pOutput0 = pOutputRow;
    float* pOutput_1 = pOutput0 - OutputWidth;
    float* pOutput_2 = pOutput_1 - OutputWidth;
    float* pOutput_3 = pOutput_2 - OutputWidth;
    float* pOutput_4 = pOutput_3 - OutputWidth;

    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);

    pInput0 += paddingLeftSeek;

    for (int64_t j = 0; j < widthIterations; j++) {
      __m512 _b0_1 = _mm512_load_ps(pInput0);

      __m512 _accc = _mm512_load_ps(pOutput_4);
      __m512 _accd = _mm512_load_ps(pOutput_3);
      __m512 _acce = _mm512_load_ps(pOutput_2);
      __m512 _accf = _mm512_load_ps(pOutput_1);

      __m512 _ak4 = _mm512_set1_ps(Filter[24]);
      __m512 _ak3 = _mm512_set1_ps(Filter[19]);
      __m512 _ak2 = _mm512_set1_ps(Filter[14]);
      __m512 _ak1 = _mm512_set1_ps(Filter[9]);
      __m512 _ak0 = _mm512_set1_ps(Filter[4]);

      _accc = _mm512_fmadd_ps(_ak4, _b0_1, _accc);
      _accd = _mm512_fmadd_ps(_ak3, _b0_1, _accd);
      _acce = _mm512_fmadd_ps(_ak2, _b0_1, _acce);
      _accf = _mm512_fmadd_ps(_ak1, _b0_1, _accf);
      __m512 _acc0 = _mm512_mul_ps(_ak0, _b0_1);

      __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

      _ak4 = _mm512_set1_ps(Filter[23]);
      _ak3 = _mm512_set1_ps(Filter[18]);
      _ak2 = _mm512_set1_ps(Filter[13]);
      _ak1 = _mm512_set1_ps(Filter[8]);
      _ak0 = _mm512_set1_ps(Filter[3]);

      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

      _ak4 = _mm512_set1_ps(Filter[22]);
      _ak3 = _mm512_set1_ps(Filter[17]);
      _ak2 = _mm512_set1_ps(Filter[12]);
      _ak1 = _mm512_set1_ps(Filter[7]);
      _ak0 = _mm512_set1_ps(Filter[2]);

      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);

      _ak4 = _mm512_set1_ps(Filter[21]);
      _ak3 = _mm512_set1_ps(Filter[16]);
      _ak2 = _mm512_set1_ps(Filter[11]);
      _ak1 = _mm512_set1_ps(Filter[6]);
      _ak0 = _mm512_set1_ps(Filter[1]);

      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

      _ak4 = _mm512_set1_ps(Filter[20]);
      _ak3 = _mm512_set1_ps(Filter[15]);
      _ak2 = _mm512_set1_ps(Filter[10]);
      _ak1 = _mm512_set1_ps(Filter[5]);
      _ak0 = _mm512_set1_ps(Filter[0]);

      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);

      // Store
      _mm512_store_ps(pOutput_4, _accc);
      _mm512_store_ps(pOutput_3, _accd);
      _mm512_store_ps(pOutput_2, _acce);
      _mm512_store_ps(pOutput_1, _accf);

      _mm512_store_ps(pOutput0, _acc0);

      _b0_0 = _b0_1;

      pInput0 += widthStep;

      pOutput_4 += widthStep;
      pOutput_3 += widthStep;
      pOutput_2 += widthStep;
      pOutput_1 += widthStep;
      pOutput0 += widthStep;
    }

    // Right Edge
    {
      __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);

      __m512 _ak0 = _mm512_set1_ps(Filter[4]);
      __m512 _ak1 = _mm512_set1_ps(Filter[9]);
      __m512 _ak2 = _mm512_set1_ps(Filter[14]);
      __m512 _ak3 = _mm512_set1_ps(Filter[19]);
      __m512 _ak4 = _mm512_set1_ps(Filter[24]);

      __m512 _acc0 = _mm512_mul_ps(_ak0, _b0_1);
      __m512 _accf = _mm512_mul_ps(_ak1, _b0_1);
      __m512 _acce = _mm512_mul_ps(_ak2, _b0_1);
      __m512 _accd = _mm512_mul_ps(_ak3, _b0_1);
      __m512 _accc = _mm512_mul_ps(_ak4, _b0_1);

      __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

      _ak0 = _mm512_set1_ps(Filter[3]);
      _ak1 = _mm512_set1_ps(Filter[8]);
      _ak2 = _mm512_set1_ps(Filter[13]);
      _ak3 = _mm512_set1_ps(Filter[18]);
      _ak4 = _mm512_set1_ps(Filter[23]);

      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

      _ak0 = _mm512_set1_ps(Filter[2]);
      _ak1 = _mm512_set1_ps(Filter[7]);
      _ak2 = _mm512_set1_ps(Filter[12]);
      _ak3 = _mm512_set1_ps(Filter[17]);
      _ak4 = _mm512_set1_ps(Filter[22]);

      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);

      _ak0 = _mm512_set1_ps(Filter[1]);
      _ak1 = _mm512_set1_ps(Filter[6]);
      _ak2 = _mm512_set1_ps(Filter[11]);
      _ak3 = _mm512_set1_ps(Filter[16]);
      _ak4 = _mm512_set1_ps(Filter[21]);

      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);

      _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

      _ak0 = _mm512_set1_ps(Filter[0]);
      _ak1 = _mm512_set1_ps(Filter[5]);
      _ak2 = _mm512_set1_ps(Filter[10]);
      _ak3 = _mm512_set1_ps(Filter[15]);
      _ak4 = _mm512_set1_ps(Filter[20]);

      _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);
      _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
      _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
      _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
      _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);

      if (widthOutputRemainder < widthInputRemainder) {
        __m512 _zero = _mm512_setzero_ps();

        __m512 _acc4_x = _mm512_load_ps(pOutput_4);
        __m512 _acc3_x = _mm512_load_ps(pOutput_3);
        __m512 _acc2_x = _mm512_load_ps(pOutput_2);
        __m512 _acc1_x = _mm512_load_ps(pOutput_1);

        _acc4_x = _mm512_add_ps(_acc4_x, _accc);
        _acc3_x = _mm512_add_ps(_acc3_x, _accd);
        _acc2_x = _mm512_add_ps(_acc2_x, _acce);
        _acc1_x = _mm512_add_ps(_acc1_x, _accf);

        _mm512_store_ps(pOutput_4, _acc4_x);
        _mm512_store_ps(pOutput_3, _acc3_x);
        _mm512_store_ps(pOutput_2, _acc2_x);
        _mm512_store_ps(pOutput_1, _acc1_x);

        _mm512_store_ps(pOutput0, _acc0);

        pOutput_4 += widthStep;
        pOutput_3 += widthStep;
        pOutput_2 += widthStep;
        pOutput_1 += widthStep;
        pOutput0 += widthStep;

        _b0 = _mm512_permutex2var_ps(_b0_1, _roll_left_1, _zero);

        _ak0 = _mm512_set1_ps(Filter[3]);
        _ak1 = _mm512_set1_ps(Filter[8]);
        _ak2 = _mm512_set1_ps(Filter[13]);
        _ak3 = _mm512_set1_ps(Filter[18]);
        _ak4 = _mm512_set1_ps(Filter[23]);

        _acc0 = _mm512_mul_ps(_ak0, _b0);
        _accf = _mm512_mul_ps(_ak1, _b0);
        _acce = _mm512_mul_ps(_ak2, _b0);
        _accd = _mm512_mul_ps(_ak3, _b0);
        _accc = _mm512_mul_ps(_ak4, _b0);

        _b0 = _mm512_permutex2var_ps(_b0_1, _roll_left_2, _zero);

        _ak0 = _mm512_set1_ps(Filter[2]);
        _ak1 = _mm512_set1_ps(Filter[7]);
        _ak2 = _mm512_set1_ps(Filter[12]);
        _ak3 = _mm512_set1_ps(Filter[17]);
        _ak4 = _mm512_set1_ps(Filter[22]);

        _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);
        _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
        _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
        _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
        _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);

        _b0 = _mm512_permutex2var_ps(_b0_1, _roll_left_3, _zero);

        _ak0 = _mm512_set1_ps(Filter[1]);
        _ak1 = _mm512_set1_ps(Filter[6]);
        _ak2 = _mm512_set1_ps(Filter[11]);
        _ak3 = _mm512_set1_ps(Filter[16]);
        _ak4 = _mm512_set1_ps(Filter[21]);

        _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);
        _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
        _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
        _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
        _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);

        _b0 = _mm512_permutex2var_ps(_b0_1, _roll_left_4, _zero);

        _ak0 = _mm512_set1_ps(Filter[0]);
        _ak1 = _mm512_set1_ps(Filter[5]);
        _ak2 = _mm512_set1_ps(Filter[10]);
        _ak3 = _mm512_set1_ps(Filter[15]);
        _ak4 = _mm512_set1_ps(Filter[20]);

        _acc0 = _mm512_fmadd_ps(_ak0, _b0, _acc0);
        _accf = _mm512_fmadd_ps(_ak1, _b0, _accf);
        _acce = _mm512_fmadd_ps(_ak2, _b0, _acce);
        _accd = _mm512_fmadd_ps(_ak3, _b0, _accd);
        _accc = _mm512_fmadd_ps(_ak4, _b0, _accc);
      }

      __m512 _acc4_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_4);
      __m512 _acc3_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_3);
      __m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_2);
      __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_1);

      _acc4_x = _mm512_add_ps(_acc4_x, _accc);
      _acc3_x = _mm512_add_ps(_acc3_x, _accd);
      _acc2_x = _mm512_add_ps(_acc2_x, _acce);
      _acc1_x = _mm512_add_ps(_acc1_x, _accf);

      _mm512_mask_store_ps(pOutput_4, _rightOutputMask, _acc4_x);
      _mm512_mask_store_ps(pOutput_3, _rightOutputMask, _acc3_x);
      _mm512_mask_store_ps(pOutput_2, _rightOutputMask, _acc2_x);
      _mm512_mask_store_ps(pOutput_1, _rightOutputMask, _acc1_x);

      _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);
    }

    pInputRow += InputWidth;

    pOutputRow += OutputWidth;
  }

  // Outer Loop Epilogue
  for (int64_t i = 1; i < KernelHeight - PaddingBottom; i++) {
    const float* pInput0 = pInputRow;

    float* pOutput0 = pOutputRow;

    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);

    pInput0 += paddingLeftSeek;

    for (int64_t j = 0; j < widthIterations; j++) {
      const float* pFilterRow = Filter + i * KernelWidth;

      float* pOutput1 = pOutput0;

      __m512 _b0_1 = _mm512_load_ps(pInput0);

      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
      __m512 _b3 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
      __m512 _b4 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

      for (int64_t k = KernelHeight; k-- > i;) {
        pOutput1 -= OutputWidth;

        __m512 _acc0 = _mm512_load_ps(pOutput1);

        __m512 _aj4 = _mm512_set1_ps(pFilterRow[4]);
        _acc0 = _mm512_fmadd_ps(_aj4, _b0_1, _acc0);

        __m512 _aj3 = _mm512_set1_ps(pFilterRow[3]);
        _acc0 = _mm512_fmadd_ps(_aj3, _b1, _acc0);

        __m512 _aj2 = _mm512_set1_ps(pFilterRow[2]);
        _acc0 = _mm512_fmadd_ps(_aj2, _b2, _acc0);

        __m512 _aj1 = _mm512_set1_ps(pFilterRow[1]);
        _acc0 = _mm512_fmadd_ps(_aj1, _b3, _acc0);

        __m512 _aj0 = _mm512_set1_ps(pFilterRow[0]);
        _acc0 = _mm512_fmadd_ps(_aj0, _b4, _acc0);

        // Store
        _mm512_store_ps(pOutput1, _acc0);

        pFilterRow += KernelWidth;
      }

      _b0_0 = _b0_1;

      pInput0 += widthStep;
      pOutput0 += widthStep;
    }

    // Right Edge
    {
      const float* pFilterRow = Filter + i * KernelWidth;

      float* pOutput1 = pOutput0;

      __m512 _zero = _mm512_setzero_ps();

      __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);

      __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
      __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
      __m512 _b3 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
      __m512 _b4 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

      for (int64_t k = KernelHeight; k-- > i;) {
        pOutput1 -= OutputWidth;

        __m512 _aj4 = _mm512_set1_ps(pFilterRow[4]);
        __m512 _acc0 = _mm512_mul_ps(_aj4, _b0_1);

        __m512 _aj3 = _mm512_set1_ps(pFilterRow[3]);
        _acc0 = _mm512_fmadd_ps(_aj3, _b1, _acc0);

        __m512 _aj2 = _mm512_set1_ps(pFilterRow[2]);
        _acc0 = _mm512_fmadd_ps(_aj2, _b2, _acc0);

        __m512 _aj1 = _mm512_set1_ps(pFilterRow[1]);
        _acc0 = _mm512_fmadd_ps(_aj1, _b3, _acc0);

        __m512 _aj0 = _mm512_set1_ps(pFilterRow[0]);
        _acc0 = _mm512_fmadd_ps(_aj0, _b4, _acc0);

        float* pOutput2 = pOutput1;

        if (widthOutputRemainder < widthInputRemainder) {
          __m512 _acc1 = _mm512_load_ps(pOutput2);

          _acc1 = _mm512_add_ps(_acc1, _acc0);

          _mm512_store_ps(pOutput2, _acc1);

          pOutput2 += widthStep;

          __m512 _b1_1 = _mm512_permutex2var_ps(_b0_1, _roll_left_1, _zero);
          _acc0 = _mm512_mul_ps(_aj3, _b1_1);

          __m512 _b1_2 = _mm512_permutex2var_ps(_b0_1, _roll_left_2, _zero);
          _acc0 = _mm512_fmadd_ps(_aj2, _b1_2, _acc0);

          __m512 _b1_3 = _mm512_permutex2var_ps(_b0_1, _roll_left_3, _zero);
          _acc0 = _mm512_fmadd_ps(_aj1, _b1_3, _acc0);

          __m512 _b1_4 = _mm512_permutex2var_ps(_b0_1, _roll_left_4, _zero);
          _acc0 = _mm512_fmadd_ps(_aj0, _b1_4, _acc0);
        }

        __m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput2);

        _acc1 = _mm512_add_ps(_acc1, _acc0);

        _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc1);

        pFilterRow += KernelWidth;
      }
    }

    pInputRow += InputWidth;
  }
}

void MlasConv2DSlidingKernelK32(const MLAS_CONV_PARAMETERS* Parameters,
                           const float* Input,
                           const float* Filter,
                           float* Output)
{
  constexpr size_t Dimensions = 2;
  constexpr size_t HeightShapeIndex = 0;
  constexpr size_t WidthShapeIndex = 1;

  const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
  const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
  const size_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
  const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

  const int64_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
  const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
  const int64_t PaddingTop = Parameters->Padding[HeightShapeIndex];
  const int64_t PaddingLeft = Parameters->Padding[WidthShapeIndex];
  const int64_t PaddingBottom = Parameters->Padding[Dimensions + HeightShapeIndex];
  const int64_t PaddingRight = Parameters->Padding[Dimensions + WidthShapeIndex];

  assert(OutputHeight == (InputHeight - KernelHeight + PaddingTop + PaddingBottom + 1));
  assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
  MLAS_UNREFERENCED_PARAMETER(InputHeight);
  MLAS_UNREFERENCED_PARAMETER(PaddingRight);

  constexpr size_t widthStep = 32;

  const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

  const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

  const __mmask64 _leftInputMask = (~0ull << paddingLeftShift);

  const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMask);
  const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);

  const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

  const int64_t widthIterations = InputWidthBase / widthStep;
  const int64_t widthInputRemainder = InputWidthBase % widthStep;
  const int64_t widthOutputRemainder = OutputWidth % widthStep;

  const int64_t kernelRemainder = KernelWidth - 16;

  const __mmask64 _rightInputMask = ~(~0ull << widthInputRemainder);

  const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
  const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);

  const __mmask64 _rightOutputMask = ~(~0ull << widthOutputRemainder);

  const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
  const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);

  const __m512i _roll_left_1 = _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
  const __m512i _one = _mm512_set1_epi32(1);

  const float* pInputRow = Input;

  float* pOutputRow = Output;

  // Outer Loop Prologue
  for (int64_t i = 0; i < PaddingTop; i++) {
    __m512 _zero = _mm512_setzero_ps();

    float* pOutput0 = pOutputRow;

    for (int64_t j = 0; j < widthIterations; j++) {
      _mm512_store_ps(pOutput0, _zero);

      _mm512_store_ps(pOutput0 + 16, _zero);

      pOutput0 += widthStep;
    }

    if (widthOutputRemainder < widthInputRemainder) {
      _mm512_store_ps(pOutput0, _zero);

      _mm512_store_ps(pOutput0 + 16, _zero);

      pOutput0 += widthStep;
    }

    _mm512_mask_store_ps(pOutput0, _rightOutputMask1, _zero);

    _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _zero);

    pOutputRow += OutputWidth;
  }

  for (int64_t i = PaddingTop; i < KernelHeight; i++) {
    const float* pInput0 = pInputRow;

    float* pOutput0 = pOutputRow;

    __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift + 0);
    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

    pInput0 += paddingLeftSeek;

    for (int64_t j = 0; j < widthIterations; j++) {
      int64_t idx = KernelWidth;

      const float* pFilterRow = Filter;

      float* pOutput1 = pOutput0;

      __m512 _b0_1 = _mm512_load_ps(pInput0);
      __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

      __m512 _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
      __m512 _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

      __m512 _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
      __m512 _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);

      __m512 _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
      __m512 _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);

      __m512i _roll = _roll_left_1;

      for (int64_t l = 1ll; l < kernelRemainder; l++) {
        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

        __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
        __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
        __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

        _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
        _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      for (int64_t l = kernelRemainder; l < 16ll; l++) {
        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

        __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
        __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
      _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);

      // Store
      _mm512_store_ps(pOutput1 + 0, _acc0_1);
      _mm512_store_ps(pOutput1 + 16, _acc0_2);

      for (int64_t k = i; k-- > 0;) {
        idx = KernelWidth;

        pFilterRow += KernelWidth;

        pOutput1 -= OutputWidth;

        _acc0_1 = _mm512_load_ps(pOutput1);
        _acc0_2 = _mm512_load_ps(pOutput1 + 16);

        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b0_1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b0_2, _acc0_2);

        _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
        _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);

        _roll = _roll_left_1;

        for (int64_t l = 1ll; l < kernelRemainder; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
          _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

          __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
          __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
          __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
          _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

          _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
          _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        for (int64_t l = kernelRemainder; l < 16ll; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

          __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
          __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
          _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
        _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);

        // Store
        _mm512_store_ps(pOutput1 + 0, _acc0_1);
        _mm512_store_ps(pOutput1 + 16, _acc0_2);
      }

      _b0_f = _b0_1;
      _b0_0 = _b0_2;

      pInput0 += widthStep;
      pOutput0 += widthStep;
    }

    // Right Edge
    {
      int64_t idx = KernelWidth;

      const float* pFilterRow = Filter;

      float* pOutput1 = pOutput0;

      __m512 _zero = _mm512_setzero_ps();

      __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0);
      __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);

      __m512 _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
      __m512 _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

      __m512 _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
      __m512 _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);

      __m512 _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
      __m512 _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);

      __m512i _roll = _roll_left_1;

      for (int64_t l = 1ll; l < kernelRemainder; l++) {
        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

        __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
        __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
        __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

        _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
        _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      for (int64_t l = kernelRemainder; l < 16ll; l++) {
        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

        __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
        __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
      _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);

      float* pOutput2 = pOutput1;

      if (widthOutputRemainder < widthInputRemainder) {
        _mm512_store_ps(pOutput2 + 0, _acc0_1);
        _mm512_store_ps(pOutput2 + 16, _acc0_2);

        pOutput2 += widthStep;

        _acc0_1 = _zero;

        _acc1_0 = _mm512_mul_ps(_aj1, _b0_2);
        _acc1_1 = _zero;

        _roll = _roll_left_1;

        for (int64_t l = 1ll; l < kernelRemainder; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
          _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

          __m512 _b1 = _mm512_permutex2var_ps(_b0_2, _roll, _zero);
          __m512 _b0 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);

          _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
          _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        for (int64_t l = kernelRemainder; l < 16ll; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

          __m512 _b1 = _mm512_permutex2var_ps(_b0_2, _roll, _zero);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
        _acc0_2 = _acc1_1;
      }

      _mm512_mask_store_ps(pOutput2, _rightOutputMask1, _acc0_1);
      _mm512_mask_store_ps(pOutput2 + 16, _rightOutputMask2, _acc0_2);

      for (int64_t k = i; k-- > 0;) {
        idx = KernelWidth;

        pFilterRow += KernelWidth;

        pOutput1 -= OutputWidth;

        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

        _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
        _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);

        _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
        _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);

        _roll = _roll_left_1;

        for (int64_t l = 1ll; l < kernelRemainder; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
          _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

          __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
          __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
          __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
          _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

          _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
          _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        for (int64_t l = kernelRemainder; l < 16ll; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

          __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
          __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
          _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
        _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);

        pOutput2 = pOutput1;

        if (widthOutputRemainder < widthInputRemainder) {
          __m512 _acc2_1 = _mm512_load_ps(pOutput2);
          __m512 _acc2_2 = _mm512_load_ps(pOutput2 + 16);

          _acc2_1 = _mm512_add_ps(_acc2_1, _acc0_1);
          _acc2_2 = _mm512_add_ps(_acc2_2, _acc0_2);

          _mm512_store_ps(pOutput2, _acc2_1);
          _mm512_store_ps(pOutput2 + 16, _acc2_1);

          pOutput2 += widthStep;

          _acc0_1 = _zero;

          _acc1_0 = _mm512_mul_ps(_aj1, _b0_2);
          _acc1_1 = _zero;

          _roll = _roll_left_1;

          for (int64_t l = 1ll; l < kernelRemainder; l++) {
            _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
            _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

            __m512 _b1 = _mm512_permutex2var_ps(_b0_2, _roll, _zero);
            __m512 _b0 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

            _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);

            _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
            _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

            _roll = _mm512_sub_epi32(_roll, _one);
          }

          for (int64_t l = kernelRemainder; l < 16ll; l++) {
            _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

            __m512 _b1 = _mm512_permutex2var_ps(_b0_2, _roll, _zero);

            _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);

            _roll = _mm512_sub_epi32(_roll, _one);
          }

          _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
          _acc0_2 = _acc1_1;
        }

        __m512 _acc2_1 = _mm512_maskz_load_ps(_rightOutputMask1, pOutput2);
        __m512 _acc2_2 = _mm512_maskz_load_ps(_rightOutputMask2, pOutput2 + 16);

        _acc2_1 = _mm512_add_ps(_acc2_1, _acc0_1);
        _acc2_2 = _mm512_add_ps(_acc2_2, _acc0_2);

        _mm512_mask_store_ps(pOutput2, _rightOutputMask1, _acc2_1);
        _mm512_mask_store_ps(pOutput2 + 16, _rightOutputMask2, _acc2_2);
      }
    }

    pInputRow += InputWidth;

    pOutputRow += OutputWidth;
  }

  // Outer loop
  for (size_t i = KernelHeight; i < OutputHeight; i++) {
    const float* pInput0 = pInputRow;

    float* pOutput0 = pOutputRow;

    __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift + 0);
    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

    pInput0 += paddingLeftSeek;

    for (int64_t j = 0; j < widthIterations; j++) {
      int64_t idx = KernelWidth;

      const float* pFilterRow = Filter;

      float* pOutput1 = pOutput0;

      __m512 _b0_1 = _mm512_load_ps(pInput0);
      __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

      __m512 _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
      __m512 _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

      __m512 _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
      __m512 _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);

      __m512 _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
      __m512 _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);

      __m512i _roll = _roll_left_1;

      for (int64_t l = 1ll; l < kernelRemainder; l++) {
        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

        __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
        __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
        __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

        _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
        _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      for (int64_t l = kernelRemainder; l < 16ll; l++) {
        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

        __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
        __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
      _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);

      // Store
      _mm512_store_ps(pOutput1 + 0, _acc0_1);
      _mm512_store_ps(pOutput1 + 16, _acc0_2);

      for (int64_t k = KernelHeight; --k > 0;) {
        idx = KernelWidth;

        pFilterRow += KernelWidth;

        pOutput1 -= OutputWidth;

        _acc0_1 = _mm512_load_ps(pOutput1);
        _acc0_2 = _mm512_load_ps(pOutput1 + 16);

        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b0_1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b0_2, _acc0_2);

        _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
        _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);

        _roll = _roll_left_1;

        for (int64_t l = 1ll; l < kernelRemainder; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
          _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

          __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
          __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
          __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
          _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

          _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
          _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        for (int64_t l = kernelRemainder; l < 16ll; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

          __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
          __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
          _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
        _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);

        // Store
        _mm512_store_ps(pOutput1 + 0, _acc0_1);
        _mm512_store_ps(pOutput1 + 16, _acc0_2);
      }

      _b0_f = _b0_1;
      _b0_0 = _b0_2;

      pInput0 += widthStep;
      pOutput0 += widthStep;
    }

    // Right Edge
    {
      int64_t idx = KernelWidth;

      const float* pFilterRow = Filter;

      float* pOutput1 = pOutput0;

      __m512 _zero = _mm512_setzero_ps();

      __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0);
      __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);

      __m512 _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
      __m512 _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

      __m512 _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
      __m512 _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);

      __m512 _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
      __m512 _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);

      __m512i _roll = _roll_left_1;

      for (int64_t l = 1ll; l < kernelRemainder; l++) {
        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

        __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
        __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
        __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

        _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
        _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      for (int64_t l = kernelRemainder; l < 16ll; l++) {
        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

        __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
        __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

        _roll = _mm512_sub_epi32(_roll, _one);
      }

      _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
      _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);

      float* pOutput2 = pOutput1;

      if (widthOutputRemainder < widthInputRemainder) {
        _mm512_store_ps(pOutput2 + 0, _acc0_1);
        _mm512_store_ps(pOutput2 + 16, _acc0_2);

        pOutput2 += widthStep;

        _acc0_1 = _zero;

        _acc1_0 = _mm512_mul_ps(_aj1, _b0_2);
        _acc1_1 = _zero;

        _roll = _roll_left_1;

        for (int64_t l = 1ll; l < kernelRemainder; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
          _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

          __m512 _b1 = _mm512_permutex2var_ps(_b0_2, _roll, _zero);
          __m512 _b0 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);

          _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
          _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        for (int64_t l = kernelRemainder; l < 16ll; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

          __m512 _b1 = _mm512_permutex2var_ps(_b0_2, _roll, _zero);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
        _acc0_2 = _acc1_1;
      }

      _mm512_mask_store_ps(pOutput2, _rightOutputMask1, _acc0_1);
      _mm512_mask_store_ps(pOutput2 + 16, _rightOutputMask2, _acc0_2);

      for (int64_t k = KernelHeight; --k > 0;) {
        idx = KernelWidth;

        pFilterRow += KernelWidth;

        pOutput1 -= OutputWidth;

        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

        _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
        _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);

        _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
        _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);

        _roll = _roll_left_1;

        for (int64_t l = 1ll; l < kernelRemainder; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
          _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

          __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
          __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
          __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
          _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

          _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
          _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        for (int64_t l = kernelRemainder; l < 16ll; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

          __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
          __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
          _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
        _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);

        pOutput2 = pOutput1;

        if (widthOutputRemainder < widthInputRemainder) {
          __m512 _acc2_1 = _mm512_load_ps(pOutput2);
          __m512 _acc2_2 = _mm512_load_ps(pOutput2 + 16);

          _acc2_1 = _mm512_add_ps(_acc2_1, _acc0_1);
          _acc2_2 = _mm512_add_ps(_acc2_2, _acc0_2);

          _mm512_store_ps(pOutput2, _acc2_1);
          _mm512_store_ps(pOutput2 + 16, _acc2_1);

          pOutput2 += widthStep;

          _acc0_1 = _zero;

          _acc1_0 = _mm512_mul_ps(_aj1, _b0_2);
          _acc1_1 = _zero;

          _roll = _roll_left_1;

          for (int64_t l = 1ll; l < kernelRemainder; l++) {
            _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
            _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

            __m512 _b1 = _mm512_permutex2var_ps(_b0_2, _roll, _zero);
            __m512 _b0 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

            _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);

            _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
            _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

            _roll = _mm512_sub_epi32(_roll, _one);
          }

          for (int64_t l = kernelRemainder; l < 16ll; l++) {
            _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

            __m512 _b1 = _mm512_permutex2var_ps(_b0_2, _roll, _zero);

            _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);

            _roll = _mm512_sub_epi32(_roll, _one);
          }

          _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
          _acc0_2 = _acc1_1;
        }

        __m512 _acc2_1 = _mm512_maskz_load_ps(_rightOutputMask1, pOutput2);
        __m512 _acc2_2 = _mm512_maskz_load_ps(_rightOutputMask2, pOutput2 + 16);

        _acc2_1 = _mm512_add_ps(_acc2_1, _acc0_1);
        _acc2_2 = _mm512_add_ps(_acc2_2, _acc0_2);

        _mm512_mask_store_ps(pOutput2, _rightOutputMask1, _acc2_1);
        _mm512_mask_store_ps(pOutput2 + 16, _rightOutputMask2, _acc2_2);
      }
    }

    pInputRow += InputWidth;

    pOutputRow += OutputWidth;
  }

  // Outer Loop Epilogue
  for (int64_t i = 1; i < KernelHeight - PaddingBottom; i++) {
    const float* pInput0 = pInputRow;

    float* pOutput0 = pOutputRow;

    __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift + 0);
    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

    pInput0 += paddingLeftSeek;

    for (int64_t j = 0; j < widthIterations; j++) {
      const float* pFilterRow = Filter + i * KernelWidth;

      float* pOutput1 = pOutput0;

      __m512 _b0_1 = _mm512_load_ps(pInput0);
      __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

      for (int64_t k = KernelHeight; k-- > i;) {
        int64_t idx = KernelWidth;

        pOutput1 -= OutputWidth;

        __m512 _acc0_1 = _mm512_load_ps(pOutput1);
        __m512 _acc0_2 = _mm512_load_ps(pOutput1 + 16);

        __m512 _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
        __m512 _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b0_1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b0_2, _acc0_2);

        __m512 _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
        __m512 _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);

        __m512i _roll = _roll_left_1;

        for (int64_t l = 1ll; l < kernelRemainder; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
          _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

          __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
          __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
          __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
          _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

          _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
          _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        for (int64_t l = kernelRemainder; l < 16ll; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

          __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
          __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
          _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
        _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);

        // Store
        _mm512_store_ps(pOutput1 + 0, _acc0_1);
        _mm512_store_ps(pOutput1 + 16, _acc0_2);

        pFilterRow += KernelWidth;
      }

      _b0_f = _b0_1;
      _b0_0 = _b0_2;

      pInput0 += widthStep;
      pOutput0 += widthStep;
    }

    // Right Edge
    {
      const float* pFilterRow = Filter + i * KernelWidth;

      float* pOutput1 = pOutput0;

      __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0);
      __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);

      for (int64_t k = KernelHeight; k-- > i;) {
        int64_t idx = KernelWidth;

        pOutput1 -= OutputWidth;

        __m512 _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
        __m512 _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

        __m512 _acc0_1 = _mm512_mul_ps(_aj0, _b0_1);
        __m512 _acc0_2 = _mm512_mul_ps(_aj0, _b0_2);

        __m512 _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
        __m512 _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);

        __m512i _roll = _roll_left_1;

        for (int64_t l = 1ll; l < kernelRemainder; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
          _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

          __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
          __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
          __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
          _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

          _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
          _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        for (int64_t l = kernelRemainder; l < 16ll; l++) {
          _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

          __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
          __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

          _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
          _acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

          _roll = _mm512_sub_epi32(_roll, _one);
        }

        _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
        _acc0_2 = _mm512_add_ps(_acc0_2, _acc1_1);

        float* pOutput2 = pOutput1;

        if (widthOutputRemainder < widthInputRemainder) {
          __m512 _zero = _mm512_setzero_ps();

          __m512 _acc2_1 = _mm512_load_ps(pOutput2);
          __m512 _acc2_2 = _mm512_load_ps(pOutput2 + 16);

          _acc2_1 = _mm512_add_ps(_acc2_1, _acc0_1);
          _acc2_2 = _mm512_add_ps(_acc2_2, _acc0_2);

          _mm512_store_ps(pOutput2, _acc2_1);
          _mm512_store_ps(pOutput2 + 16, _acc2_1);

          pOutput2 += widthStep;

          _acc0_1 = _zero;

          _acc1_0 = _mm512_mul_ps(_aj1, _b0_2);
          _acc1_1 = _zero;

          _roll = _roll_left_1;

          for (int64_t l = 1ll; l < kernelRemainder; l++) {
            _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
            _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

            __m512 _b1 = _mm512_permutex2var_ps(_b0_2, _roll, _zero);
            __m512 _b0 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

            _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);

            _acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
            _acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

            _roll = _mm512_sub_epi32(_roll, _one);
          }

          for (int64_t l = kernelRemainder; l < 16ll; l++) {
            _aj0 = _mm512_set1_ps(pFilterRow[--idx]);

            __m512 _b1 = _mm512_permutex2var_ps(_b0_2, _roll, _zero);

            _acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);

            _roll = _mm512_sub_epi32(_roll, _one);
          }

          _acc0_1 = _mm512_add_ps(_acc0_1, _acc1_0);
          _acc0_2 = _acc1_1;
        }

        __m512 _acc2_1 = _mm512_maskz_load_ps(_rightOutputMask1, pOutput2);
        __m512 _acc2_2 = _mm512_maskz_load_ps(_rightOutputMask2, pOutput2 + 16);

        _acc2_1 = _mm512_add_ps(_acc2_1, _acc0_1);
        _acc2_2 = _mm512_add_ps(_acc2_2, _acc0_2);

        _mm512_mask_store_ps(pOutput2, _rightOutputMask1, _acc2_1);
        _mm512_mask_store_ps(pOutput2 + 16, _rightOutputMask2, _acc2_2);

        pFilterRow += KernelWidth;
      }
    }

    pInputRow += InputWidth;
  }
}

void
MlasConv2DSlidingKernel(const MLAS_CONV_PARAMETERS* Parameters,
                           const float* Input,
                           const float* Filter,
                           float* Output)
{
  constexpr size_t HeightShapeIndex = 0;
  constexpr size_t WidthShapeIndex = 1;

  const int64_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
  const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];

  if ((KernelHeight == 3) && (KernelWidth == 3)) {
    MlasConv2DSlidingKernelK3x3(Parameters, Input, Filter, Output);
  } else if ((KernelHeight == 5) && (KernelWidth == 5)) {
    MlasConv2DSlidingKernelK5x5(Parameters, Input, Filter, Output);
  } else if (KernelWidth <= 16) {
    MlasConv2DSlidingKernelK17(Parameters, Input, Filter, Output);
  } else if (KernelWidth <= 32) {
    MlasConv2DSlidingKernelK32(Parameters, Input, Filter, Output);
  } else
    return;
}