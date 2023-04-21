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
MlasConvPointwiseKernel(const MLAS_CONV_PARAMETERS* Parameters,
                        const float* Input,
                        const float* Filter,
                        float* Output)
{
    const size_t InputChannels = Parameters->InputChannels;

    const size_t InputSize = Parameters->InputSize;
    const size_t OutputSize = Parameters->OutputSize;
    const size_t KernelSize = Parameters->KernelSize;

    assert(OutputSize == InputSize);

    constexpr size_t inputStep = 4;

    const size_t inputIterations = InputChannels / inputStep;
    const size_t inputRemainder = InputChannels % inputStep;

    constexpr size_t widthStep = 16;

    const size_t widthIterations = InputSize / widthStep;
    const size_t widthRemainder = InputSize % widthStep;

    const __mmask16 _rightMask = static_cast<__mmask16>(~(~0 << widthRemainder));

    const float* pInput = Input;

    for (size_t ic = 0; ic < inputIterations; ic++) {
        float* pOutput0 = Output;

        const float* pInput0 = pInput;
        const float* pInput1 = pInput0 + InputSize;
        const float* pInput2 = pInput1 + InputSize;
        const float* pInput3 = pInput2 + InputSize;

        __m512 _a0 = _mm512_set1_ps(Filter[0]);
        __m512 _a1 = _mm512_set1_ps(Filter[1]);
        __m512 _a2 = _mm512_set1_ps(Filter[2]);
        __m512 _a3 = _mm512_set1_ps(Filter[3]);

        for (size_t i = 0; i < widthIterations; i++) {
            __m512 _acc0 = _mm512_load_ps(pOutput0);

            _acc0 = _mm512_fmadd_ps(_a0, _mm512_load_ps(pInput0), _acc0);
            _acc0 = _mm512_fmadd_ps(_a1, _mm512_load_ps(pInput1), _acc0);
            _acc0 = _mm512_fmadd_ps(_a2, _mm512_load_ps(pInput2), _acc0);
            _acc0 = _mm512_fmadd_ps(_a3, _mm512_load_ps(pInput3), _acc0);

            _mm512_store_ps(pOutput0, _acc0);

            pInput0 += widthStep;
            pInput1 += widthStep;
            pInput2 += widthStep;
            pInput3 += widthStep;

            pOutput0 += widthStep;
        }

        // Right Edge
        {
            __m512 _acc0 = _mm512_maskz_load_ps(_rightMask, pOutput0);

            _acc0 = _mm512_fmadd_ps(_a0, _mm512_maskz_load_ps(_rightMask, pInput0), _acc0);
            _acc0 = _mm512_fmadd_ps(_a1, _mm512_maskz_load_ps(_rightMask, pInput1), _acc0);
            _acc0 = _mm512_fmadd_ps(_a2, _mm512_maskz_load_ps(_rightMask, pInput2), _acc0);
            _acc0 = _mm512_fmadd_ps(_a3, _mm512_maskz_load_ps(_rightMask, pInput3), _acc0);

            _mm512_mask_store_ps(pOutput0, _rightMask, _acc0);
        }

        pInput += inputStep * InputSize;
        Filter += inputStep * KernelSize;
    }

    for (size_t ic = 0; ic < inputRemainder; ic++) {
        float* pOutput0 = Output;

        const float* pInput0 = pInput;

        __m512 _a0 = _mm512_set1_ps(Filter[0]);

        for (size_t i = 0; i < widthIterations; i++) {
            __m512 _out0 = _mm512_load_ps(pOutput0);
            __m512 _acc0 = _mm512_fmadd_ps(_a0, _mm512_load_ps(pInput0), _out0);
            _mm512_store_ps(pOutput0, _acc0);

            pInput0 += widthStep;
            pOutput0 += widthStep;
        }

        // Right Edge
        {
            __m512 _out0 = _mm512_maskz_load_ps(_rightMask, pOutput0);
            __m512 _acc0 = _mm512_fmadd_ps(_a0, _mm512_maskz_load_ps(_rightMask, pInput0), _out0);
            _mm512_mask_store_ps(pOutput0, _rightMask, _acc0);
        }

        pInput += InputSize;
        Filter += KernelSize;
    }
}

void
MlasConv1DSlidingKernelK17(const MLAS_CONV_PARAMETERS* Parameters,
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

    constexpr int64_t widthStep = 32;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const int64_t paddingRightShift = std::max<int64_t>(paddingLeftSeek - InputWidth, 0ll);

    const __mmask64 _leftInputMask =
        (~0ull << paddingLeftShift) & (0xffffffffull >> paddingRightShift);

    const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek + paddingRightShift;

    const int64_t widthIterations = InputWidthBase / widthStep;
    const int64_t widthInputRemainder = InputWidthBase % widthStep;
    const int64_t widthOutputRemainder = OutputWidth % widthStep;

    const __mmask64 _rightInputMask = ~(~0ull << widthInputRemainder);

    const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
    const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);

    const __mmask64 _rightOutputMask = ~(~0ull << widthOutputRemainder);

    const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
    const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
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

        __m512 _acc0 = _mm512_load_ps(pOutput0);
        __m512 _acc1 = _mm512_load_ps(pOutput0 + 16);

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
            _acc0 = _mm512_add_ps(_acc0, _mm512_load_ps(pOutput0));
            _acc1 = _mm512_add_ps(_acc1, _mm512_load_ps(pOutput0 + 16));

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
        _acc0 = _mm512_add_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask1, pOutput0));
        _acc1 = _mm512_add_ps(_acc1, _mm512_maskz_load_ps(_rightOutputMask2, pOutput0 + 16));

        _mm512_mask_store_ps(pOutput0, _rightOutputMask1, _acc0);
        _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _acc1);
    }
}

void
MlasConv1DSlidingKernelK32(const MLAS_CONV_PARAMETERS* Parameters,
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

    const int64_t paddingRightShift = std::max<int64_t>(paddingLeftSeek - InputWidth, 0ll);

    const __mmask64 _leftInputMask =
        (~0ull << paddingLeftShift) & (0xffffffffull >> paddingRightShift);

    const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMask);
    const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek + paddingRightShift;

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

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _one = _mm512_set1_epi32(1);

    float* pOutput0 = Output;

    const float* pInput0 = Input;

    __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift + 0);
    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

    pInput0 += paddingLeftSeek;

    for (int64_t i = 0; i < widthIterations; i++) {
        int64_t idx = KernelWidth;

        __m512 _acc0_1 = _mm512_load_ps(pOutput0);
        __m512 _acc0_2 = _mm512_load_ps(pOutput0 + 16);

        __m512 _b0_1 = _mm512_load_ps(pInput0);
        __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

        __m512 _aj0 = _mm512_set1_ps(Filter[--idx]);
        __m512 _aj1 = _mm512_set1_ps(Filter[idx - 16]);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b0_1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b0_2, _acc0_2);

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
            _acc0_1 = _mm512_add_ps(_acc0_1, _mm512_load_ps(pOutput0));
            _acc0_2 = _mm512_add_ps(_acc0_2, _mm512_load_ps(pOutput0 + 16));

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
        _acc0_1 = _mm512_add_ps(_acc0_1, _mm512_maskz_load_ps(_rightOutputMask1, pOutput0));
        _acc0_2 = _mm512_add_ps(_acc0_2, _mm512_maskz_load_ps(_rightOutputMask2, pOutput0 + 16));

        _mm512_mask_store_ps(pOutput0 + 0, _rightOutputMask1, _acc0_1);
        _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _acc0_2);
    }
}

void
MlasConv1DSlidingKernelK48(const MLAS_CONV_PARAMETERS* Parameters,
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

    const int64_t paddingRightShift = std::max<int64_t>(paddingLeftSeek - InputWidth, 0ll);

    const __mmask64 _leftInputMask =
        (~0ull << paddingLeftShift) & (0xffffffffffffull >> paddingRightShift);

    const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMask);
    const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);
    const __mmask16 _leftInputMask3 = static_cast<__mmask16>(_leftInputMask >> 32);

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek + paddingRightShift;

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

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _one = _mm512_set1_epi32(1);

    float* pOutput0 = Output;

    const float* pInput0 = Input;

    __m512 _b0_e = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift + 0);
    __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);
    __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask3, pInput0 - paddingLeftShift + 32);

    pInput0 += paddingLeftSeek;

    for (int64_t i = 0; i < widthIterations; i++) {
        int64_t idx = KernelWidth;

        __m512 _acc0_1 = _mm512_load_ps(pOutput0);
        __m512 _acc0_2 = _mm512_load_ps(pOutput0 + 16);
        __m512 _acc0_3 = _mm512_load_ps(pOutput0 + 32);

        __m512 _b0_1 = _mm512_load_ps(pInput0);
        __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);
        __m512 _b0_3 = _mm512_load_ps(pInput0 + 32);

        __m512 _aj0 = _mm512_set1_ps(Filter[--idx]);
        __m512 _aj1 = _mm512_set1_ps(Filter[idx - 16]);
        __m512 _aj2 = _mm512_set1_ps(Filter[idx - 32]);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b0_1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b0_2, _acc0_2);
        _acc0_3 = _mm512_fmadd_ps(_aj0, _b0_3, _acc0_3);

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
            _acc0_1 = _mm512_add_ps(_acc0_1, _mm512_load_ps(pOutput0));
            _acc0_2 = _mm512_add_ps(_acc0_2, _mm512_load_ps(pOutput0 + 16));
            _acc0_3 = _mm512_add_ps(_acc0_3, _mm512_load_ps(pOutput0 + 32));

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
        _acc0_1 = _mm512_add_ps(_acc0_1, _mm512_maskz_load_ps(_rightOutputMask1, pOutput0));
        _acc0_2 = _mm512_add_ps(_acc0_2, _mm512_maskz_load_ps(_rightOutputMask2, pOutput0 + 16));
        _acc0_3 = _mm512_add_ps(_acc0_3, _mm512_maskz_load_ps(_rightOutputMask3, pOutput0 + 32));

        _mm512_mask_store_ps(pOutput0 + 0, _rightOutputMask1, _acc0_1);
        _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _acc0_2);
        _mm512_mask_store_ps(pOutput0 + 32, _rightOutputMask3, _acc0_3);
    }
}

void
MlasConv1DSlidingKernelK64(const MLAS_CONV_PARAMETERS* Parameters,
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

    const int64_t paddingRightShift = std::max<int64_t>(paddingLeftSeek - InputWidth, 0ll);

    const __mmask64 _leftInputMask = (~0ull << paddingLeftShift) & (~0ull >> paddingRightShift);

    const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMask);
    const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);
    const __mmask16 _leftInputMask3 = static_cast<__mmask16>(_leftInputMask >> 32);
    const __mmask16 _leftInputMask4 = static_cast<__mmask16>(_leftInputMask >> 48);

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek + paddingRightShift;

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

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
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

        __m512 _acc0_1 = _mm512_load_ps(pOutput0);
        __m512 _acc0_2 = _mm512_load_ps(pOutput0 + 16);
        __m512 _acc0_3 = _mm512_load_ps(pOutput0 + 32);
        __m512 _acc0_4 = _mm512_load_ps(pOutput0 + 48);

        __m512 _b0_1 = _mm512_load_ps(pInput0);
        __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);
        __m512 _b0_3 = _mm512_load_ps(pInput0 + 32);
        __m512 _b0_4 = _mm512_load_ps(pInput0 + 48);

        __m512 _aj0 = _mm512_set1_ps(Filter[--idx]);
        __m512 _aj1 = _mm512_set1_ps(Filter[idx - 16]);
        __m512 _aj2 = _mm512_set1_ps(Filter[idx - 32]);
        __m512 _aj3 = _mm512_set1_ps(Filter[idx - 48]);

        _acc0_1 = _mm512_fmadd_ps(_aj0, _b0_1, _acc0_1);
        _acc0_2 = _mm512_fmadd_ps(_aj0, _b0_2, _acc0_2);
        _acc0_3 = _mm512_fmadd_ps(_aj0, _b0_3, _acc0_3);
        _acc0_4 = _mm512_fmadd_ps(_aj0, _b0_4, _acc0_4);

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
            _acc0_1 = _mm512_add_ps(_acc0_1, _mm512_load_ps(pOutput0));
            _acc0_2 = _mm512_add_ps(_acc0_2, _mm512_load_ps(pOutput0 + 16));
            _acc0_3 = _mm512_add_ps(_acc0_3, _mm512_load_ps(pOutput0 + 32));
            _acc0_4 = _mm512_add_ps(_acc0_4, _mm512_load_ps(pOutput0 + 48));

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
        _acc0_1 = _mm512_add_ps(_acc0_1, _mm512_maskz_load_ps(_rightOutputMask1, pOutput0));
        _acc0_2 = _mm512_add_ps(_acc0_2, _mm512_maskz_load_ps(_rightOutputMask2, pOutput0 + 16));
        _acc0_3 = _mm512_add_ps(_acc0_3, _mm512_maskz_load_ps(_rightOutputMask3, pOutput0 + 32));
        _acc0_4 = _mm512_add_ps(_acc0_4, _mm512_maskz_load_ps(_rightOutputMask4, pOutput0 + 48));

        _mm512_mask_store_ps(pOutput0 + 0, _rightOutputMask1, _acc0_1);
        _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _acc0_2);
        _mm512_mask_store_ps(pOutput0 + 32, _rightOutputMask3, _acc0_3);
        _mm512_mask_store_ps(pOutput0 + 48, _rightOutputMask4, _acc0_4);
    }
}

void
MlasConv2DSlidingKernelK17(const MLAS_CONV_PARAMETERS* Parameters,
                           const float* Input,
                           const float* Filter,
                           float* Output)
{
    constexpr size_t Dimensions = 2;
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const int64_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const int64_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const int64_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    const int64_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

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
    MLAS_UNREFERENCED_PARAMETER(PaddingBottom);

    constexpr int64_t widthStep = 32;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const int64_t paddingRightShift = std::max<int64_t>(paddingLeftSeek - InputWidth, 0ll);

    const __mmask64 _leftInputMask =
        (~0ull << paddingLeftShift) & (0xffffffffull >> paddingRightShift);

    const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek + paddingRightShift;

    const int64_t widthIterations = InputWidthBase / widthStep;
    const int64_t widthInputRemainder = InputWidthBase % widthStep;
    const int64_t widthOutputRemainder = OutputWidth % widthStep;

    const __mmask64 _rightInputMask = ~(~0ull << widthInputRemainder);

    const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
    const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);

    const __mmask64 _rightOutputMask = ~(~0ull << widthOutputRemainder);

    const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
    const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _one = _mm512_set1_epi32(1);

    Output += PaddingTop * OutputWidth;

    // Outer loop
    for (int64_t i = 0; i < InputHeight; i++) {
        const float* pInput0 = Input;

        float* pOutput0 = Output;

        __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

        pInput0 += paddingLeftSeek;

        for (int64_t j = 0; j < widthIterations; j++) {
            float* pOutput1 = pOutput0;

            const float* pFilterRow = Filter;

            __m512 _b0_1 = _mm512_load_ps(pInput0);
            __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

            for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                if ((0 <= gk) && (gk < OutputHeight)) {
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
                }

                pFilterRow += KernelWidth;

                pOutput1 -= OutputWidth;
            }

            _b0_0 = _b0_2;

            pInput0 += widthStep;
            pOutput0 += widthStep;
        }

        // Right Edge
        {
            float* pOutput1 = pOutput0;

            const float* pFilterRow = Filter;

            __m512 _zero = _mm512_setzero_ps();

            __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0);
            __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);

            for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                if ((0 <= gk) && (gk < OutputHeight)) {
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
                        _acc0 = _mm512_add_ps(_acc0, _mm512_load_ps(pOutput2));
                        _acc1 = _mm512_add_ps(_acc1, _mm512_load_ps(pOutput2 + 16));

                        _mm512_store_ps(pOutput2, _acc0);
                        _mm512_store_ps(pOutput2 + 16, _acc1);

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

                    _acc0 = _mm512_add_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask1, pOutput2));
                    _acc1 = _mm512_add_ps(_acc1,
                                          _mm512_maskz_load_ps(_rightOutputMask2, pOutput2 + 16));

                    _mm512_mask_store_ps(pOutput2, _rightOutputMask1, _acc0);
                    _mm512_mask_store_ps(pOutput2 + 16, _rightOutputMask2, _acc1);
                }

                pFilterRow += KernelWidth;

                pOutput1 -= OutputWidth;
            }
        }

        Input += InputWidth;

        Output += OutputWidth;
    }
}

void
MlasConv2DSlidingKernelK3x3(const MLAS_CONV_PARAMETERS* Parameters,
                            const float* Input,
                            const float* Filter,
                            float* Output)
{
    constexpr size_t Dimensions = 2;
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const int64_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const int64_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const int64_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    const int64_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

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

    constexpr int64_t heightStep = 4;

    const int64_t paddingBottomSeek = KernelHeight - PaddingBottom - 1;

    const int64_t InputHeightBase = std::max<int64_t>(int64_t(InputHeight) - paddingBottomSeek, 0);

    const int64_t heightIterations = InputHeightBase / heightStep;
    const int64_t heightRemainder = InputHeightBase % heightStep;

    constexpr int64_t widthStep = 16;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const int64_t paddingRightShift = std::max<int64_t>(paddingLeftSeek - InputWidth, 0ll);

    const __mmask16 _leftInputMask =
        static_cast<__mmask16>((0xffff << paddingLeftShift) & (0xffff >> paddingRightShift));

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek + paddingRightShift;

    const int64_t widthIterations = InputWidthBase / widthStep;
    const int64_t widthInputRemainder = InputWidthBase % widthStep;
    const int64_t widthOutputRemainder = OutputWidth % widthStep;

    const __mmask16 _rightInputMask = static_cast<__mmask16>(~(~0 << widthInputRemainder));
    const __mmask16 _rightOutputMask = static_cast<__mmask16>(~(~0 << widthOutputRemainder));

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _roll_left_2 =
        _mm512_set_epi32(29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14);

    const __m512 _a0 = _mm512_set1_ps(Filter[0]);
    const __m512 _a1 = _mm512_set1_ps(Filter[1]);
    const __m512 _a2 = _mm512_set1_ps(Filter[2]);
    const __m512 _a3 = _mm512_set1_ps(Filter[3]);
    const __m512 _a4 = _mm512_set1_ps(Filter[4]);
    const __m512 _a5 = _mm512_set1_ps(Filter[5]);
    const __m512 _a6 = _mm512_set1_ps(Filter[6]);
    const __m512 _a7 = _mm512_set1_ps(Filter[7]);
    const __m512 _a8 = _mm512_set1_ps(Filter[8]);

    Output += PaddingTop * OutputWidth;

    if (heightIterations > 0) {
        // Outer Loop Prologue
        for (int64_t i = 0; i < heightStep; i++) {
            const float* pInput0 = Input;

            float* pOutput0 = Output;

            __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                float* pOutput1 = pOutput0;

                const float* pFilterRow = Filter;

                __m512 _b0_1 = _mm512_load_ps(pInput0);

                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                    if ((0 <= gk) && (gk < OutputHeight)) {
                        __m512 _acc0 = _mm512_load_ps(pOutput1);

                        __m512 _aj2 = _mm512_set1_ps(pFilterRow[2]);
                        _acc0 = _mm512_fmadd_ps(_aj2, _b0_1, _acc0);

                        __m512 _aj1 = _mm512_set1_ps(pFilterRow[1]);
                        _acc0 = _mm512_fmadd_ps(_aj1, _b1, _acc0);

                        __m512 _aj0 = _mm512_set1_ps(pFilterRow[0]);
                        _acc0 = _mm512_fmadd_ps(_aj0, _b2, _acc0);

                        // Store
                        _mm512_store_ps(pOutput1, _acc0);
                    }

                    pFilterRow += KernelWidth;

                    pOutput1 -= OutputWidth;
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

                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                    if ((0 <= gk) && (gk < OutputHeight)) {
                        __m512 _aj2 = _mm512_set1_ps(pFilterRow[2]);
                        __m512 _acc0 = _mm512_mul_ps(_aj2, _b0_1);

                        __m512 _aj1 = _mm512_set1_ps(pFilterRow[1]);
                        _acc0 = _mm512_fmadd_ps(_aj1, _b1, _acc0);

                        __m512 _aj0 = _mm512_set1_ps(pFilterRow[0]);
                        _acc0 = _mm512_fmadd_ps(_aj0, _b2, _acc0);

                        float* pOutput2 = pOutput1;

                        if (widthOutputRemainder < widthInputRemainder) {
                            _acc0 = _mm512_add_ps(_acc0, _mm512_load_ps(pOutput2));

                            _mm512_store_ps(pOutput2, _acc0);

                            pOutput2 += widthStep;

                            __m512 _b1_1 = _mm512_permutex2var_ps(_b0_1, _roll_left_1, _zero);
                            _acc0 = _mm512_mul_ps(_aj1, _b1_1);

                            __m512 _b1_2 = _mm512_permutex2var_ps(_b0_1, _roll_left_2, _zero);
                            _acc0 = _mm512_fmadd_ps(_aj0, _b1_2, _acc0);
                        }

                        _acc0 =
                            _mm512_add_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutput2));

                        _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc0);
                    }

                    pFilterRow += KernelWidth;

                    pOutput1 -= OutputWidth;
                }
            }

            Input += InputWidth;

            Output += OutputWidth;
        }

        // Outer Loop
        for (int64_t i = 1; i < heightIterations; i++) {
            const float* pInput0 = Input;
            const float* pInput1 = pInput0 + InputWidth;
            const float* pInput2 = pInput1 + InputWidth;
            const float* pInput3 = pInput2 + InputWidth;

            float* pOutput0 = Output;
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
                __m512 _acc0 = _mm512_load_ps(pOutput0);
                __m512 _acc1 = _mm512_load_ps(pOutput1);
                __m512 _acc2 = _mm512_load_ps(pOutput2);
                __m512 _acc3 = _mm512_load_ps(pOutput3);

                __m512 _b0_1 = _mm512_load_ps(pInput0);
                __m512 _b1_1 = _mm512_load_ps(pInput1);
                __m512 _b2_1 = _mm512_load_ps(pInput2);
                __m512 _b3_1 = _mm512_load_ps(pInput3);

                _acce = _mm512_fmadd_ps(_a8, _b0_1, _acce);
                _accf = _mm512_fmadd_ps(_a5, _b0_1, _accf);
                _acc0 = _mm512_fmadd_ps(_a2, _b0_1, _acc0);

                _accf = _mm512_fmadd_ps(_a8, _b1_1, _accf);
                _acc0 = _mm512_fmadd_ps(_a5, _b1_1, _acc0);
                _acc1 = _mm512_fmadd_ps(_a2, _b1_1, _acc1);

                _acc0 = _mm512_fmadd_ps(_a8, _b2_1, _acc0);
                _acc1 = _mm512_fmadd_ps(_a5, _b2_1, _acc1);
                _acc2 = _mm512_fmadd_ps(_a2, _b2_1, _acc2);

                _acc1 = _mm512_fmadd_ps(_a8, _b3_1, _acc1);
                _acc2 = _mm512_fmadd_ps(_a5, _b3_1, _acc2);
                _acc3 = _mm512_fmadd_ps(_a2, _b3_1, _acc3);

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

                    _acce = _mm512_add_ps(_acce, _mm512_load_ps(pOutput_2));
                    _accf = _mm512_add_ps(_accf, _mm512_load_ps(pOutput_1));
                    _acc0 = _mm512_add_ps(_acc0, _mm512_load_ps(pOutput0));
                    _acc1 = _mm512_add_ps(_acc1, _mm512_load_ps(pOutput1));
                    _acc2 = _mm512_add_ps(_acc2, _mm512_load_ps(pOutput2));
                    _acc3 = _mm512_add_ps(_acc3, _mm512_load_ps(pOutput3));

                    _mm512_store_ps(pOutput_2, _acce);
                    _mm512_store_ps(pOutput_1, _accf);
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

                _acce = _mm512_add_ps(_acce, _mm512_maskz_load_ps(_rightOutputMask, pOutput_2));
                _accf = _mm512_add_ps(_accf, _mm512_maskz_load_ps(_rightOutputMask, pOutput_1));
                _acc0 = _mm512_add_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutput0));
                _acc1 = _mm512_add_ps(_acc1, _mm512_maskz_load_ps(_rightOutputMask, pOutput1));
                _acc2 = _mm512_add_ps(_acc2, _mm512_maskz_load_ps(_rightOutputMask, pOutput2));
                _acc3 = _mm512_add_ps(_acc3, _mm512_maskz_load_ps(_rightOutputMask, pOutput3));

                _mm512_mask_store_ps(pOutput_2, _rightOutputMask, _acce);
                _mm512_mask_store_ps(pOutput_1, _rightOutputMask, _accf);

                _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);
                _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
                _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc2);
                _mm512_mask_store_ps(pOutput3, _rightOutputMask, _acc3);
            }

            Input += heightStep * InputWidth;

            Output += heightStep * OutputWidth;
        }
    }

    // Outer Loop Epilogue
    for (int64_t i = InputHeightBase - heightRemainder; i < InputHeight; i++) {
        const float* pInput0 = Input;

        float* pOutput0 = Output;

        __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);

        pInput0 += paddingLeftSeek;

        for (int64_t j = 0; j < widthIterations; j++) {
            const float* pFilterRow = Filter;

            float* pOutput1 = pOutput0;

            __m512 _b0_1 = _mm512_load_ps(pInput0);

            __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
            __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

            for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                if ((0 <= gk) && (gk < OutputHeight)) {
                    __m512 _acc0 = _mm512_load_ps(pOutput1);

                    __m512 _aj2 = _mm512_set1_ps(pFilterRow[2]);
                    _acc0 = _mm512_fmadd_ps(_aj2, _b0_1, _acc0);

                    __m512 _aj1 = _mm512_set1_ps(pFilterRow[1]);
                    _acc0 = _mm512_fmadd_ps(_aj1, _b1, _acc0);

                    __m512 _aj0 = _mm512_set1_ps(pFilterRow[0]);
                    _acc0 = _mm512_fmadd_ps(_aj0, _b2, _acc0);

                    // Store
                    _mm512_store_ps(pOutput1, _acc0);
                }

                pFilterRow += KernelWidth;

                pOutput1 -= OutputWidth;
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

            __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
            __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

            for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                if ((0 <= gk) && (gk < OutputHeight)) {
                    __m512 _aj2 = _mm512_set1_ps(pFilterRow[2]);
                    __m512 _acc0 = _mm512_mul_ps(_aj2, _b0_1);

                    __m512 _aj1 = _mm512_set1_ps(pFilterRow[1]);
                    _acc0 = _mm512_fmadd_ps(_aj1, _b1, _acc0);

                    __m512 _aj0 = _mm512_set1_ps(pFilterRow[0]);
                    _acc0 = _mm512_fmadd_ps(_aj0, _b2, _acc0);

                    float* pOutput2 = pOutput1;

                    if (widthOutputRemainder < widthInputRemainder) {
                        _acc0 = _mm512_add_ps(_acc0, _mm512_load_ps(pOutput2));

                        _mm512_store_ps(pOutput2, _acc0);

                        pOutput2 += widthStep;

                        __m512 _b1_1 = _mm512_permutex2var_ps(_b0_1, _roll_left_1, _zero);
                        _acc0 = _mm512_mul_ps(_aj1, _b1_1);

                        __m512 _b1_2 = _mm512_permutex2var_ps(_b0_1, _roll_left_2, _zero);
                        _acc0 = _mm512_fmadd_ps(_aj0, _b1_2, _acc0);
                    }

                    _acc0 = _mm512_add_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutput2));

                    _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc0);
                }

                pFilterRow += KernelWidth;

                pOutput1 -= OutputWidth;
            }
        }

        Input += InputWidth;

        Output += OutputWidth;
    }
}

void
MlasConv2DSlidingKernelK5x5(const MLAS_CONV_PARAMETERS* Parameters,
                            const float* Input,
                            const float* Filter,
                            float* Output)
{
    constexpr size_t Dimensions = 2;
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const int64_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const int64_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const int64_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    const int64_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

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
    MLAS_UNREFERENCED_PARAMETER(PaddingRight);

    constexpr int64_t heightStep = 4;

    const int64_t paddingBottomSeek = KernelHeight - PaddingBottom - 1;

    const int64_t InputHeightBase = std::max<int64_t>(int64_t(InputHeight) - paddingBottomSeek, 0);

    const int64_t heightIterations = InputHeightBase / heightStep;
    const int64_t heightRemainder = InputHeightBase % heightStep;

    constexpr int64_t widthStep = 16;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const int64_t paddingRightShift = std::max<int64_t>(paddingLeftSeek - InputWidth, 0ll);

    const __mmask16 _leftInputMask =
        static_cast<__mmask16>((0xffff << paddingLeftShift) & (0xffff >> paddingRightShift));

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek + paddingRightShift;

    const int64_t widthIterations = InputWidthBase / widthStep;
    const int64_t widthInputRemainder = InputWidthBase % widthStep;
    const int64_t widthOutputRemainder = OutputWidth % widthStep;

    const __mmask16 _rightInputMask = static_cast<__mmask16>(~(~0 << widthInputRemainder));
    const __mmask16 _rightOutputMask = static_cast<__mmask16>(~(~0 << widthOutputRemainder));

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _roll_left_2 =
        _mm512_set_epi32(29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14);
    const __m512i _roll_left_3 =
        _mm512_set_epi32(28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13);
    const __m512i _roll_left_4 =
        _mm512_set_epi32(27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12);

    float* pOutputRow = Output + PaddingTop * OutputWidth;

    const float* pInputRow = Input;

    if (heightIterations > 0) {
        // Outer Loop Prologue
        for (int64_t i = 0; i < heightStep; i++) {
            const float* pInput0 = pInputRow;

            float* pOutput0 = pOutputRow;

            __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                const float* pFilterRow = Filter;

                float* pOutput1 = pOutput0;

                __m512 _b0_1 = _mm512_load_ps(pInput0);

                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
                __m512 _b3 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
                __m512 _b4 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

                for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                    if ((0 <= gk) && (gk < OutputHeight)) {
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
                    }

                    pFilterRow += KernelWidth;

                    pOutput1 -= OutputWidth;
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

                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
                __m512 _b3 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
                __m512 _b4 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

                for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                    if ((0 <= gk) && (gk < OutputHeight)) {
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
                            _acc0 = _mm512_add_ps(_acc0, _mm512_load_ps(pOutput2));

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

                        _acc0 =
                            _mm512_add_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutput2));

                        _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc0);
                    }

                    pFilterRow += KernelWidth;

                    pOutput1 -= OutputWidth;
                }
            }

            pInputRow += InputWidth;

            pOutputRow += OutputWidth;
        }

        // Outer loop
        for (int64_t i = 1; i < heightIterations; i++) {
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
                __m512 _acc0 = _mm512_load_ps(pOutput0);
                __m512 _acc1 = _mm512_load_ps(pOutput1);
                __m512 _acc2 = _mm512_load_ps(pOutput2);
                __m512 _acc3 = _mm512_load_ps(pOutput3);

                __m512 _ak4 = _mm512_set1_ps(Filter[24]);
                __m512 _ak3 = _mm512_set1_ps(Filter[19]);
                __m512 _ak2 = _mm512_set1_ps(Filter[14]);
                __m512 _ak1 = _mm512_set1_ps(Filter[9]);
                __m512 _ak0 = _mm512_set1_ps(Filter[4]);

                _accc = _mm512_fmadd_ps(_ak4, _b0_1, _accc);
                _accd = _mm512_fmadd_ps(_ak3, _b0_1, _accd);
                _acce = _mm512_fmadd_ps(_ak2, _b0_1, _acce);
                _accf = _mm512_fmadd_ps(_ak1, _b0_1, _accf);
                _acc0 = _mm512_fmadd_ps(_ak0, _b0_1, _acc0);

                _accd = _mm512_fmadd_ps(_ak4, _b1_1, _accd);
                _acce = _mm512_fmadd_ps(_ak3, _b1_1, _acce);
                _accf = _mm512_fmadd_ps(_ak2, _b1_1, _accf);
                _acc0 = _mm512_fmadd_ps(_ak1, _b1_1, _acc0);
                _acc1 = _mm512_fmadd_ps(_ak0, _b1_1, _acc1);

                _acce = _mm512_fmadd_ps(_ak4, _b2_1, _acce);
                _accf = _mm512_fmadd_ps(_ak3, _b2_1, _accf);
                _acc0 = _mm512_fmadd_ps(_ak2, _b2_1, _acc0);
                _acc1 = _mm512_fmadd_ps(_ak1, _b2_1, _acc1);
                _acc2 = _mm512_fmadd_ps(_ak0, _b2_1, _acc2);

                _accf = _mm512_fmadd_ps(_ak4, _b3_1, _accf);
                _acc0 = _mm512_fmadd_ps(_ak3, _b3_1, _acc0);
                _acc1 = _mm512_fmadd_ps(_ak2, _b3_1, _acc1);
                _acc2 = _mm512_fmadd_ps(_ak1, _b3_1, _acc2);
                _acc3 = _mm512_fmadd_ps(_ak0, _b3_1, _acc3);

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

                    _accc = _mm512_add_ps(_accc, _mm512_load_ps(pOutput_4));
                    _accd = _mm512_add_ps(_accd, _mm512_load_ps(pOutput_3));
                    _acce = _mm512_add_ps(_acce, _mm512_load_ps(pOutput_2));
                    _accf = _mm512_add_ps(_accf, _mm512_load_ps(pOutput_1));

                    _acc0 = _mm512_add_ps(_acc0, _mm512_load_ps(pOutput0));
                    _acc1 = _mm512_add_ps(_acc1, _mm512_load_ps(pOutput1));
                    _acc2 = _mm512_add_ps(_acc2, _mm512_load_ps(pOutput2));
                    _acc3 = _mm512_add_ps(_acc3, _mm512_load_ps(pOutput3));

                    _mm512_store_ps(pOutput_4, _accc);
                    _mm512_store_ps(pOutput_3, _accd);
                    _mm512_store_ps(pOutput_2, _acce);
                    _mm512_store_ps(pOutput_1, _accf);

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

                _accc = _mm512_add_ps(_accc, _mm512_maskz_load_ps(_rightOutputMask, pOutput_4));
                _accd = _mm512_add_ps(_accd, _mm512_maskz_load_ps(_rightOutputMask, pOutput_3));
                _acce = _mm512_add_ps(_acce, _mm512_maskz_load_ps(_rightOutputMask, pOutput_2));
                _accf = _mm512_add_ps(_accf, _mm512_maskz_load_ps(_rightOutputMask, pOutput_1));

                _acc0 = _mm512_add_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutput0));
                _acc1 = _mm512_add_ps(_acc1, _mm512_maskz_load_ps(_rightOutputMask, pOutput1));
                _acc2 = _mm512_add_ps(_acc2, _mm512_maskz_load_ps(_rightOutputMask, pOutput2));
                _acc3 = _mm512_add_ps(_acc3, _mm512_maskz_load_ps(_rightOutputMask, pOutput3));

                _mm512_mask_store_ps(pOutput_4, _rightOutputMask, _accc);
                _mm512_mask_store_ps(pOutput_3, _rightOutputMask, _accd);
                _mm512_mask_store_ps(pOutput_2, _rightOutputMask, _acce);
                _mm512_mask_store_ps(pOutput_1, _rightOutputMask, _accf);

                _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);
                _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
                _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc2);
                _mm512_mask_store_ps(pOutput3, _rightOutputMask, _acc3);
            }

            pInputRow += heightStep * InputWidth;

            pOutputRow += heightStep * OutputWidth;
        }
    }

    // Outer Loop Epilogue
    for (int64_t i = InputHeightBase - heightRemainder; i < InputHeight; i++) {
        const float* pInput0 = pInputRow;

        float* pOutput0 = pOutputRow;

        __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);

        pInput0 += paddingLeftSeek;

        for (int64_t j = 0; j < widthIterations; j++) {
            const float* pFilterRow = Filter;

            float* pOutput1 = pOutput0;

            __m512 _b0_1 = _mm512_load_ps(pInput0);

            __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
            __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
            __m512 _b3 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
            __m512 _b4 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

            for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                if ((0 <= gk) && (gk < OutputHeight)) {
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
                }
                pFilterRow += KernelWidth;

                pOutput1 -= OutputWidth;
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

            __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
            __m512 _b2 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
            __m512 _b3 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
            __m512 _b4 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

            for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                if ((0 <= gk) && (gk < OutputHeight)) {
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
                        _acc0 = _mm512_add_ps(_acc0, _mm512_load_ps(pOutput2));

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

                    _acc0 = _mm512_add_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutput2));

                    _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc0);
                }

                pFilterRow += KernelWidth;

                pOutput1 -= OutputWidth;
            }
        }

        pInputRow += InputWidth;

        pOutputRow += OutputWidth;
    }
}

void
MlasConv2DSlidingKernelK32(const MLAS_CONV_PARAMETERS* Parameters,
                           const float* Input,
                           const float* Filter,
                           float* Output)
{
    constexpr size_t Dimensions = 2;
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const int64_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const int64_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const int64_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    const int64_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

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
    MLAS_UNREFERENCED_PARAMETER(PaddingBottom);

    constexpr int64_t widthStep = 32;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const int64_t paddingRightShift = std::max<int64_t>(paddingLeftSeek - InputWidth, 0ll);

    const __mmask64 _leftInputMask =
        (~0ull << paddingLeftShift) & (0xffffffffull >> paddingRightShift);

    const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMask);
    const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek + paddingRightShift;

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

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _one = _mm512_set1_epi32(1);

    const float* pInputRow = Input;

    float* pOutputRow = Output + PaddingTop * OutputWidth;
    // Outer loop
    for (int64_t i = 0; i < InputHeight; i++) {
        const float* pInput0 = pInputRow;

        float* pOutput0 = pOutputRow;

        __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift + 0);
        __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

        pInput0 += paddingLeftSeek;

        for (int64_t j = 0; j < widthIterations; j++) {
            const float* pFilterRow = Filter;

            float* pOutput1 = pOutput0;

            __m512 _b0_1 = _mm512_load_ps(pInput0);
            __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

            for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                if ((0 <= gk) && (gk < OutputHeight)) {
                    int64_t idx = KernelWidth;

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
                }

                pFilterRow += KernelWidth;

                pOutput1 -= OutputWidth;
            }

            _b0_f = _b0_1;
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

            for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                if ((0 <= gk) && (gk < OutputHeight)) {
                    int64_t idx = KernelWidth;

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
                        _acc0_1 = _mm512_add_ps(_acc0_1, _mm512_load_ps(pOutput2));
                        _acc0_2 = _mm512_add_ps(_acc0_2, _mm512_load_ps(pOutput2 + 16));

                        _mm512_store_ps(pOutput2, _acc0_1);
                        _mm512_store_ps(pOutput2 + 16, _acc0_2);

                        pOutput2 += widthStep;

                        idx = KernelWidth - 1;

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

                    _acc0_1 =
                        _mm512_add_ps(_acc0_1, _mm512_maskz_load_ps(_rightOutputMask1, pOutput2));
                    _acc0_2 = _mm512_add_ps(_acc0_2,
                                            _mm512_maskz_load_ps(_rightOutputMask2, pOutput2 + 16));

                    _mm512_mask_store_ps(pOutput2, _rightOutputMask1, _acc0_1);
                    _mm512_mask_store_ps(pOutput2 + 16, _rightOutputMask2, _acc0_2);
                }

                pFilterRow += KernelWidth;

                pOutput1 -= OutputWidth;
            }
        }

        pInputRow += InputWidth;

        pOutputRow += OutputWidth;
    }
}

void
MlasConv2DSlidingKernelK48(const MLAS_CONV_PARAMETERS* Parameters,
                           const float* Input,
                           const float* Filter,
                           float* Output)
{
    constexpr size_t Dimensions = 2;
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const int64_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const int64_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const int64_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    const int64_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

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
    MLAS_UNREFERENCED_PARAMETER(PaddingBottom);

    constexpr int64_t widthStep = 48;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const int64_t paddingRightShift = std::max<int64_t>(paddingLeftSeek - InputWidth, 0ll);

    const __mmask64 _leftInputMask =
        (~0ull << paddingLeftShift) & (0xffffffffffffull >> paddingRightShift);

    const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMask);
    const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);
    const __mmask16 _leftInputMask3 = static_cast<__mmask16>(_leftInputMask >> 32);

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek + paddingRightShift;

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

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _one = _mm512_set1_epi32(1);

    const float* pInputRow = Input;

    float* pOutputRow = Output + PaddingTop * OutputWidth;

    // Outer loop
    for (int64_t i = 0; i < InputHeight; i++) {
        const float* pInput0 = pInputRow;

        float* pOutput0 = pOutputRow;

        __m512 _b0_e = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift + 0);
        __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);
        __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask3, pInput0 - paddingLeftShift + 32);

        pInput0 += paddingLeftSeek;

        for (int64_t j = 0; j < widthIterations; j++) {
            const float* pFilterRow = Filter;

            float* pOutput1 = pOutput0;

            __m512 _b0_1 = _mm512_load_ps(pInput0);
            __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);
            __m512 _b0_3 = _mm512_load_ps(pInput0 + 32);

            for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                if ((0 <= gk) && (gk < OutputHeight)) {
                    int64_t idx = KernelWidth;

                    __m512 _acc0_1 = _mm512_load_ps(pOutput1);
                    __m512 _acc0_2 = _mm512_load_ps(pOutput1 + 16);
                    __m512 _acc0_3 = _mm512_load_ps(pOutput1 + 32);

                    __m512 _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
                    __m512 _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);
                    __m512 _aj2 = _mm512_set1_ps(pFilterRow[idx - 32]);

                    _acc0_1 = _mm512_fmadd_ps(_aj0, _b0_1, _acc0_1);
                    _acc0_2 = _mm512_fmadd_ps(_aj0, _b0_2, _acc0_2);
                    _acc0_3 = _mm512_fmadd_ps(_aj0, _b0_3, _acc0_3);

                    __m512 _acc1_0 = _mm512_mul_ps(_aj1, _b0_0);
                    __m512 _acc1_1 = _mm512_mul_ps(_aj1, _b0_1);
                    __m512 _acc1_2 = _mm512_mul_ps(_aj1, _b0_2);

                    _acc0_1 = _mm512_fmadd_ps(_aj2, _b0_f, _acc0_1);
                    _acc0_2 = _mm512_fmadd_ps(_aj2, _b0_0, _acc0_2);
                    _acc0_3 = _mm512_fmadd_ps(_aj2, _b0_1, _acc0_3);

                    __m512i _roll = _roll_left_1;

                    for (int64_t l = 1ll; l < kernelRemainder; l++) {
                        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
                        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);
                        _aj2 = _mm512_set1_ps(pFilterRow[idx - 32]);

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

                    for (int64_t l = kernelRemainder; l < 16ll; l++) {
                        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
                        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

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

                    // Store
                    _mm512_store_ps(pOutput1 + 0, _acc0_1);
                    _mm512_store_ps(pOutput1 + 16, _acc0_2);
                    _mm512_store_ps(pOutput1 + 32, _acc0_3);
                }

                pFilterRow += KernelWidth;

                pOutput1 -= OutputWidth;
            }

            _b0_e = _b0_1;
            _b0_f = _b0_2;
            _b0_0 = _b0_3;

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
            __m512 _b0_3 = _mm512_maskz_load_ps(_rightInputMask3, pInput0 + 32);

            for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                if ((0 <= gk) && (gk < OutputHeight)) {
                    int64_t idx = KernelWidth;

                    __m512 _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
                    __m512 _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);
                    __m512 _aj2 = _mm512_set1_ps(pFilterRow[idx - 32]);

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

                    for (int64_t l = 1ll; l < kernelRemainder; l++) {
                        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
                        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);
                        _aj2 = _mm512_set1_ps(pFilterRow[idx - 32]);

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

                    for (int64_t l = kernelRemainder; l < 16ll; l++) {
                        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
                        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);

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

                    float* pOutput2 = pOutput1;

                    if (widthOutputRemainder < widthInputRemainder) {
                        _acc0_1 = _mm512_add_ps(_acc0_1, _mm512_load_ps(pOutput2));
                        _acc0_2 = _mm512_add_ps(_acc0_2, _mm512_load_ps(pOutput2 + 16));
                        _acc0_3 = _mm512_add_ps(_acc0_3, _mm512_load_ps(pOutput2 + 32));

                        _mm512_store_ps(pOutput2, _acc0_1);
                        _mm512_store_ps(pOutput2 + 16, _acc0_2);
                        _mm512_store_ps(pOutput2 + 32, _acc0_3);

                        pOutput2 += widthStep;

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

                        for (int64_t l = 1ll; l < kernelRemainder; l++) {
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

                        for (int64_t l = kernelRemainder; l < 16ll; l++) {
                            _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
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

                    _acc0_1 =
                        _mm512_add_ps(_acc0_1, _mm512_maskz_load_ps(_rightOutputMask1, pOutput2));
                    _acc0_2 = _mm512_add_ps(_acc0_2,
                                            _mm512_maskz_load_ps(_rightOutputMask2, pOutput2 + 16));
                    _acc0_3 = _mm512_add_ps(_acc0_3,
                                            _mm512_maskz_load_ps(_rightOutputMask3, pOutput2 + 32));

                    _mm512_mask_store_ps(pOutput2, _rightOutputMask1, _acc0_1);
                    _mm512_mask_store_ps(pOutput2 + 16, _rightOutputMask2, _acc0_2);
                    _mm512_mask_store_ps(pOutput2 + 32, _rightOutputMask3, _acc0_3);
                }

                pFilterRow += KernelWidth;

                pOutput1 -= OutputWidth;
            }
        }

        pInputRow += InputWidth;

        pOutputRow += OutputWidth;
    }
}

void
MlasConv2DSlidingKernelK64(const MLAS_CONV_PARAMETERS* Parameters,
                           const float* Input,
                           const float* Filter,
                           float* Output)
{
    constexpr size_t Dimensions = 2;
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const int64_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const int64_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const int64_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    const int64_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

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
    MLAS_UNREFERENCED_PARAMETER(PaddingBottom);

    constexpr int64_t widthStep = 64;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const int64_t paddingRightShift = std::max<int64_t>(paddingLeftSeek - InputWidth, 0ll);

    const __mmask64 _leftInputMask = (~0ull << paddingLeftShift) & (~0ull >> paddingRightShift);

    const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMask);
    const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);
    const __mmask16 _leftInputMask3 = static_cast<__mmask16>(_leftInputMask >> 32);
    const __mmask16 _leftInputMask4 = static_cast<__mmask16>(_leftInputMask >> 48);

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek + paddingRightShift;

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

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _one = _mm512_set1_epi32(1);

    const float* pInputRow = Input;

    float* pOutputRow = Output + PaddingTop * OutputWidth;

    // Outer loop
    for (int64_t i = 0; i < InputHeight; i++) {
        const float* pInput0 = pInputRow;

        float* pOutput0 = pOutputRow;

        __m512 _b0_d = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift + 0);
        __m512 _b0_e = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);
        __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask3, pInput0 - paddingLeftShift + 32);
        __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask4, pInput0 - paddingLeftShift + 48);

        pInput0 += paddingLeftSeek;

        for (int64_t j = 0; j < widthIterations; j++) {
            const float* pFilterRow = Filter;

            float* pOutput1 = pOutput0;

            __m512 _b0_1 = _mm512_load_ps(pInput0);
            __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);
            __m512 _b0_3 = _mm512_load_ps(pInput0 + 32);
            __m512 _b0_4 = _mm512_load_ps(pInput0 + 48);

            for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                if ((0 <= gk) && (gk < OutputHeight)) {
                    int64_t idx = KernelWidth;

                    __m512 _acc0_1 = _mm512_load_ps(pOutput1);
                    __m512 _acc0_2 = _mm512_load_ps(pOutput1 + 16);
                    __m512 _acc0_3 = _mm512_load_ps(pOutput1 + 32);
                    __m512 _acc0_4 = _mm512_load_ps(pOutput1 + 48);

                    __m512 _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
                    __m512 _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);
                    __m512 _aj2 = _mm512_set1_ps(pFilterRow[idx - 32]);
                    __m512 _aj3 = _mm512_set1_ps(pFilterRow[idx - 48]);

                    _acc0_1 = _mm512_fmadd_ps(_aj0, _b0_1, _acc0_1);
                    _acc0_2 = _mm512_fmadd_ps(_aj0, _b0_2, _acc0_2);
                    _acc0_3 = _mm512_fmadd_ps(_aj0, _b0_3, _acc0_3);
                    _acc0_4 = _mm512_fmadd_ps(_aj0, _b0_4, _acc0_4);

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

                    for (int64_t l = 1ll; l < kernelRemainder; l++) {
                        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
                        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);
                        _aj2 = _mm512_set1_ps(pFilterRow[idx - 32]);
                        _aj3 = _mm512_set1_ps(pFilterRow[idx - 48]);

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

                    for (int64_t l = kernelRemainder; l < 16ll; l++) {
                        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
                        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);
                        _aj2 = _mm512_set1_ps(pFilterRow[idx - 32]);

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

                    // Store
                    _mm512_store_ps(pOutput1 + 0, _acc0_1);
                    _mm512_store_ps(pOutput1 + 16, _acc0_2);
                    _mm512_store_ps(pOutput1 + 32, _acc0_3);
                    _mm512_store_ps(pOutput1 + 48, _acc0_4);
                }

                pFilterRow += KernelWidth;

                pOutput1 -= OutputWidth;
            }

            _b0_d = _b0_1;
            _b0_e = _b0_2;
            _b0_f = _b0_3;
            _b0_0 = _b0_4;

            pInput0 += widthStep;
            pOutput0 += widthStep;
        }

        // Right Edge
        {
            const float* pFilterRow = Filter;

            float* pOutput1 = pOutput0;

            __m512 _zero = _mm512_setzero_ps();

            __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0 + 0);
            __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);
            __m512 _b0_3 = _mm512_maskz_load_ps(_rightInputMask3, pInput0 + 32);
            __m512 _b0_4 = _mm512_maskz_load_ps(_rightInputMask4, pInput0 + 48);

            for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
                if ((0 <= gk) && (gk < OutputHeight)) {
                    int64_t idx = KernelWidth;

                    __m512 _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
                    __m512 _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);
                    __m512 _aj2 = _mm512_set1_ps(pFilterRow[idx - 32]);
                    __m512 _aj3 = _mm512_set1_ps(pFilterRow[idx - 48]);

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

                    for (int64_t l = 1ll; l < kernelRemainder; l++) {
                        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
                        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);
                        _aj2 = _mm512_set1_ps(pFilterRow[idx - 32]);
                        _aj3 = _mm512_set1_ps(pFilterRow[idx - 48]);

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

                    for (int64_t l = kernelRemainder; l < 16ll; l++) {
                        _aj0 = _mm512_set1_ps(pFilterRow[--idx]);
                        _aj1 = _mm512_set1_ps(pFilterRow[idx - 16]);
                        _aj2 = _mm512_set1_ps(pFilterRow[idx - 32]);

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

                    float* pOutput2 = pOutput1;

                    if (widthOutputRemainder < widthInputRemainder) {
                        _acc0_1 = _mm512_add_ps(_acc0_1, _mm512_load_ps(pOutput2));
                        _acc0_2 = _mm512_add_ps(_acc0_2, _mm512_load_ps(pOutput2 + 16));
                        _acc0_3 = _mm512_add_ps(_acc0_3, _mm512_load_ps(pOutput2 + 32));
                        _acc0_4 = _mm512_add_ps(_acc0_4, _mm512_load_ps(pOutput2 + 48));

                        _mm512_store_ps(pOutput2, _acc0_1);
                        _mm512_store_ps(pOutput2 + 16, _acc0_2);
                        _mm512_store_ps(pOutput2 + 32, _acc0_3);
                        _mm512_store_ps(pOutput2 + 48, _acc0_4);

                        pOutput2 += widthStep;

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

                        for (int64_t l = 1ll; l < kernelRemainder; l++) {
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

                        for (int64_t l = kernelRemainder; l < 16ll; l++) {
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

                    _acc0_1 =
                        _mm512_add_ps(_acc0_1, _mm512_maskz_load_ps(_rightOutputMask1, pOutput2));
                    _acc0_2 = _mm512_add_ps(_acc0_2,
                                            _mm512_maskz_load_ps(_rightOutputMask2, pOutput2 + 16));
                    _acc0_3 = _mm512_add_ps(_acc0_3,
                                            _mm512_maskz_load_ps(_rightOutputMask3, pOutput2 + 32));
                    _acc0_4 = _mm512_add_ps(_acc0_4,
                                            _mm512_maskz_load_ps(_rightOutputMask4, pOutput2 + 48));

                    _mm512_mask_store_ps(pOutput2, _rightOutputMask1, _acc0_1);
                    _mm512_mask_store_ps(pOutput2 + 16, _rightOutputMask2, _acc0_2);
                    _mm512_mask_store_ps(pOutput2 + 32, _rightOutputMask3, _acc0_3);
                    _mm512_mask_store_ps(pOutput2 + 48, _rightOutputMask4, _acc0_4);
                }

                pFilterRow += KernelWidth;

                pOutput1 -= OutputWidth;
            }
        }

        pInputRow += InputWidth;

        pOutputRow += OutputWidth;
    }
}

void
MlasConv3DSlidingKernel(const MLAS_CONV_PARAMETERS* Parameters,
                        const float* Input,
                        const float* Filter,
                        float* Output)
{
    constexpr size_t Dimensions = 3;
    constexpr size_t DepthShapeIndex = 0;
    constexpr size_t HeightShapeIndex = 1;
    constexpr size_t WidthShapeIndex = 2;

    const int64_t InputDepth = Parameters->InputShape[DepthShapeIndex];
    const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const size_t InputArea = InputHeight * InputWidth;

    const int64_t OutputDepth = Parameters->OutputShape[DepthShapeIndex];
    const size_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];
    const size_t OutputArea = OutputHeight * OutputWidth;

    const int64_t KernelDepth = Parameters->KernelShape[DepthShapeIndex];
    const size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
    const size_t KernelArea = KernelHeight * KernelWidth;

    const int64_t PaddingFront = Parameters->Padding[DepthShapeIndex];
    const size_t PaddingTop = Parameters->Padding[HeightShapeIndex];
    const size_t PaddingLeft = Parameters->Padding[WidthShapeIndex];
    const int64_t PaddingBack = Parameters->Padding[Dimensions + DepthShapeIndex];
    const size_t PaddingBottom = Parameters->Padding[Dimensions + HeightShapeIndex];
    const size_t PaddingRight = Parameters->Padding[Dimensions + WidthShapeIndex];

    assert(OutputDepth == (InputDepth - KernelDepth + PaddingFront + PaddingBack + 1));
    MLAS_UNREFERENCED_PARAMETER(PaddingBack);

    MLAS_CONV_PARAMETERS SubTaskParameters{
        nullptr,
        2,
        Parameters->BatchCount,
        Parameters->GroupCount,
        Parameters->InputChannels,
        {InputHeight, InputWidth, 0},
        {KernelHeight, KernelWidth, 0},
        {Parameters->DilationShape[1], Parameters->DilationShape[2], 0},
        {PaddingTop, PaddingLeft, PaddingBottom, PaddingRight},
        {Parameters->StrideShape[1], Parameters->StrideShape[2], 0},
        Parameters->FilterCount,
        {OutputHeight, OutputWidth, 0},
        Parameters->InputSize,
        Parameters->OutputSize,
        Parameters->KernelSize,
        Parameters->K,
        Parameters->Beta,
        Parameters->Algorithm,
        Parameters->ThreadCount,
        Parameters->u };

    MLAS_CONV_KERNEL_ROUTINE* SubTask = MlasGetSlidingConvolutionKernel(&SubTaskParameters);

    Output += PaddingFront * OutputArea;

    // Outer loop
    for (int64_t i = 0; i < InputDepth; i++) {
        const float* pFilterLayer = Filter;

        float* pOutputLayer = Output;

        for (int64_t k = 0, gk = i + PaddingFront; k < KernelDepth; k++, gk--) {
            if ((0 <= gk) && (gk < OutputDepth)) {
                SubTask(&SubTaskParameters, Input, pFilterLayer, pOutputLayer);
            }

            pFilterLayer += KernelArea;

            pOutputLayer -= OutputArea;
        }

        Input += InputArea;

        Output += OutputArea;
    }
}

MLAS_CONV_KERNEL_ROUTINE*
MlasGetSlidingConvolutionKernel(const MLAS_CONV_PARAMETERS* Parameters)
{
    if (Parameters->KernelSize == 1) {
        return MlasConvPointwiseKernel;
    }

    switch (Parameters->Dimensions) {
        case 1: {
            constexpr size_t WidthShapeIndex = 0;

            const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];

            if (KernelWidth <= 16) {
                return MlasConv1DSlidingKernelK17;
            } else if (KernelWidth <= 32) {
                return MlasConv1DSlidingKernelK32;
            } else if (KernelWidth <= 48) {
                return MlasConv1DSlidingKernelK48;
            } else if (KernelWidth <= 64) {
                return MlasConv1DSlidingKernelK64;
            } else {
                return nullptr;
            }
        } break;

        case 2: {
            constexpr size_t HeightShapeIndex = 0;
            constexpr size_t WidthShapeIndex = 1;

            const int64_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
            const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];

            if ((KernelHeight == 3) && (KernelWidth == 3)) {
                return MlasConv2DSlidingKernelK3x3;
            } else if ((KernelHeight == 5) && (KernelWidth == 5)) {
                return MlasConv2DSlidingKernelK5x5;
            } else if (KernelWidth <= 16) {
                return MlasConv2DSlidingKernelK17;
            } else if (KernelWidth <= 32) {
                return MlasConv2DSlidingKernelK32;
            } else if (KernelWidth <= 48) {
                return MlasConv2DSlidingKernelK48;
            } else if (KernelWidth <= 64) {
                return MlasConv2DSlidingKernelK64;
            } else {
                return nullptr;
            }
        } break;

        case 3:
            return MlasConv3DSlidingKernel;
            break;

        default:
            return nullptr;
            break;
    }
}
