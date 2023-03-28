/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    spool_kernel_avx512.cpp

Abstract:

;   This module implements the kernels for the single precision
;   sliding window pooling operation.
;
;   This implementation uses AVX512F instructions.

--*/

#include <cassert>

#include "mlasi.h"
#include "spool_kernel.h"

void
MlasPool1DSlidingKernelMaxK17S1(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                size_t ChannelCount,
                                const float* Input,
                                float* Output)
{
    constexpr size_t Dimensions = 1;
    constexpr size_t WidthShapeIndex = 0;

    assert(WorkBlock->PoolingKind == MLAS_POOLING_KIND::MlasMaximumPooling);

    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
    const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

    assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
    MLAS_UNREFERENCED_PARAMETER(PaddingRight);

    constexpr size_t widthStep = 16;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const __mmask16 _leftInputMask = static_cast<__mmask16>((~0 << paddingLeftShift));

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

    const int64_t widthIterations = InputWidthBase / widthStep;
    const int64_t widthInputRemainder = InputWidthBase % widthStep;
    const int64_t widthOutputRemainder = OutputWidth % widthStep;

    const __mmask16 _rightInputMask = static_cast<__mmask16>(~(~0 << widthInputRemainder));
    const __mmask16 _rightOutputMask = static_cast<__mmask16>(~(~0 << widthOutputRemainder));

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _one = _mm512_set1_epi32(1);

    float* pOutput0 = Output;

    const __m512 _padding = _mm512_set1_ps(std::numeric_limits<float>::lowest());

    for (size_t c = 0; c < ChannelCount; c++) {
        const float* pInput0 = Input;

        __m512 _b0_0 = _mm512_mask_load_ps(_padding, _leftInputMask, pInput0 - paddingLeftShift);

        pInput0 += paddingLeftSeek;

        for (int i = 0; i < widthIterations; i++) {
            __m512 _b0_1 = _mm512_load_ps(pInput0);

            // Prologue
            __m512i _roll = _roll_left_1;

            __m512 _acc0 = _b0_1;

            __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

            // Inner Loop
            for (int j = 2; j < KernelWidth; j++) {
                _roll = _mm512_sub_epi32(_roll, _one);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
            }

            // Epilogue
            _acc0 = _mm512_max_ps(_acc0, _b0);

            _b0_0 = _b0_1;

            // Store
            _mm512_store_ps(pOutput0, _acc0);

            pInput0 += widthStep;
            pOutput0 += widthStep;
        }

        // Right Edge
        {
            __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput0);

            // Prologue
            __m512i _roll = _roll_left_1;

            __m512 _acc0 = _b0_1;

            __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

            // Inner Loop
            for (int j = 2; j < KernelWidth; j++) {
                _roll = _mm512_sub_epi32(_roll, _one);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
            }

            // Epilogue
            _acc0 = _mm512_max_ps(_acc0, _b0);

            if (widthOutputRemainder < widthInputRemainder) {
                _mm512_store_ps(pOutput0, _acc0);

                pOutput0 += widthStep;

                // Prologue
                _roll = _roll_left_1;

                _acc0 = _padding;

                _b0 = _mm512_permutex2var_ps(_b0_1, _roll, _padding);

                // Inner Loop
                for (int j = 2; j < KernelWidth; j++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_max_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_1, _roll, _padding);
                }

                // Epilogue
                _acc0 = _mm512_max_ps(_acc0, _b0);
            }

            _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);
        }

        Input += InputSize;
    }
}

void
MlasPool1DSlidingKernelMaxK32S1(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                size_t ChannelCount,
                                const float* Input,
                                float* Output)
{
    constexpr size_t Dimensions = 1;

    constexpr size_t WidthShapeIndex = 0;

    assert(WorkBlock->PoolingKind == MLAS_POOLING_KIND::MlasMaximumPooling);

    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
    const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

    assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
    MLAS_UNREFERENCED_PARAMETER(PaddingRight);

    constexpr size_t widthStep = 32;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

    const int64_t widthIterations = InputWidthBase / widthStep;
    const int64_t widthInputRemainder = InputWidthBase % widthStep;
    const int64_t widthOutputRemainder = OutputWidth % widthStep;

    const int64_t kernelRemainder = KernelWidth % 16;

    const __mmask32 _leftInputMask = static_cast<__mmask32>(~0 << paddingLeftShift);

    const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMask);
    const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);

    const __mmask32 _rightInputMask = static_cast<__mmask32>(~(~0 << widthInputRemainder));

    const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
    const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);

    const __mmask32 _rightOutputMask = static_cast<__mmask32>(~(~0 << widthOutputRemainder));

    const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
    const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _one = _mm512_set1_epi32(1);

    float* pOutput0 = Output;

    const __m512 _padding = _mm512_set1_ps(std::numeric_limits<float>::lowest());

    for (size_t c = 0; c < ChannelCount; c++) {
        const float* pInput0 = Input;

        __m512 _b0_f = _mm512_mask_load_ps(_padding, _leftInputMask1, pInput0 - paddingLeftShift);

        pInput0 += (paddingLeftSeek - 16);

        __m512 _b0_0 = _mm512_mask_load_ps(_padding, _leftInputMask2, pInput0);

        pInput0 += 16;

        for (int64_t i = 0; i < widthIterations; i++) {
            __m512 _b0_1 = _mm512_load_ps(pInput0);
            __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

            // Prologue
            __m512i _roll = _roll_left_1;

            __m512 _acc0 = _b0_0;
            __m512 _acc1 = _b0_1;
            __m512 _acc2 = _b0_2;

            __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
            __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
            __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

            // Inner Loops
            for (int64_t j = 1; j < kernelRemainder; j++) {
                _roll = _mm512_sub_epi32(_roll, _one);

                _acc0 = _mm512_max_ps(_acc0, _b0);
                _acc1 = _mm512_max_ps(_acc1, _b1);
                _acc2 = _mm512_max_ps(_acc2, _b2);

                _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
            }

            _acc2 = _mm512_max_ps(_acc2, _acc1);
            _acc1 = _mm512_max_ps(_acc1, _acc0);

            for (int64_t j = kernelRemainder; j < 15; j++) {
                _roll = _mm512_sub_epi32(_roll, _one);

                _acc1 = _mm512_max_ps(_acc1, _b1);
                _acc2 = _mm512_max_ps(_acc2, _b2);

                _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
            }

            // Epilogue
            _acc1 = _mm512_max_ps(_acc1, _b1);
            _acc2 = _mm512_max_ps(_acc2, _b2);

            _b0_f = _b0_1;
            _b0_0 = _b0_2;

            // Store
            _mm512_store_ps(pOutput0, _acc1);
            _mm512_store_ps(pOutput0 + 16, _acc2);

            pInput0 += widthStep;
            pOutput0 += widthStep;
        }

        // Right Edge
        {
            __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask1, pInput0);
            __m512 _b0_2 = _mm512_mask_load_ps(_padding, _rightInputMask2, pInput0 + 16);

            // Prologue
            __m512i _roll = _roll_left_1;

            __m512 _acc0 = _b0_0;
            __m512 _acc1 = _b0_1;
            __m512 _acc2 = _b0_2;

            __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
            __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
            __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

            // Inner Loops
            for (int j = 1; j < kernelRemainder; j++) {
                _roll = _mm512_sub_epi32(_roll, _one);

                _acc0 = _mm512_max_ps(_acc0, _b0);
                _acc1 = _mm512_max_ps(_acc1, _b1);
                _acc2 = _mm512_max_ps(_acc2, _b2);

                _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
            }

            _acc2 = _mm512_max_ps(_acc2, _acc1);
            _acc1 = _mm512_max_ps(_acc1, _acc0);

            for (int64_t j = kernelRemainder; j < 15; j++) {
                _roll = _mm512_sub_epi32(_roll, _one);

                _acc1 = _mm512_max_ps(_acc1, _b1);
                _acc2 = _mm512_max_ps(_acc2, _b2);

                _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
            }

            // Epilogue
            _acc1 = _mm512_max_ps(_acc1, _b1);
            _acc2 = _mm512_max_ps(_acc2, _b2);

            if (widthOutputRemainder < widthInputRemainder) {
                // Store
                _mm512_store_ps(pOutput0, _acc1);
                _mm512_store_ps(pOutput0 + 16, _acc2);

                pOutput0 += widthStep;

                _b0_f = _b0_1;
                _b0_0 = _b0_2;

                _b0_1 = _padding;
                _b0_2 = _padding;

                // Prologue
                _roll = _roll_left_1;

                _acc0 = _b0_0;
                _acc1 = _b0_1;
                _acc2 = _b0_2;

                _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                // Inner Loops
                for (int64_t j = 1; j < kernelRemainder; j++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_max_ps(_acc0, _b0);
                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                _acc2 = _mm512_max_ps(_acc2, _acc1);
                _acc1 = _mm512_max_ps(_acc1, _acc0);

                for (int64_t j = kernelRemainder; j < 15; j++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);

                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                // Epilogue
                _acc1 = _mm512_max_ps(_acc1, _b1);
                _acc2 = _mm512_max_ps(_acc2, _b2);
            }

            // Store
            _mm512_mask_store_ps(pOutput0, _rightOutputMask1, _acc1);
            _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _acc2);
        }

        Input += InputSize;
    }
}

void
MlasPool1DSlidingKernelMax(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                           size_t ChannelCount,
                           const float* Input,
                           float* Output)
{
    constexpr size_t WidthShapeIndex = 0;

    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];

    if (KernelWidth <= 16) {
        MlasPool1DSlidingKernelMaxK17S1(WorkBlock, ChannelCount, Input, Output);
    } else if (KernelWidth <= 32) {
        MlasPool1DSlidingKernelMaxK32S1(WorkBlock, ChannelCount, Input, Output);
    } else
        return;
}

void
MlasPool1DSlidingKernelAvgWithPadK17S1(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                       size_t ChannelCount,
                                       const float* Input,
                                       float* Output)
{
    constexpr size_t Dimensions = 1;
    constexpr size_t WidthShapeIndex = 0;

    assert(WorkBlock->PoolingKind == MLAS_POOLING_KIND::MlasAveragePoolingIncludePad);

    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
    const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

    assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
    MLAS_UNREFERENCED_PARAMETER(PaddingRight);

    constexpr size_t widthStep = 16;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const __mmask16 _leftInputMask = static_cast<__mmask16>((~0 << paddingLeftShift));

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

    const int64_t widthIterations = InputWidthBase / widthStep;
    const int64_t widthInputRemainder = InputWidthBase % widthStep;
    const int64_t widthOutputRemainder = OutputWidth % widthStep;

    const __mmask16 _rightInputMask = static_cast<__mmask16>(~(~0 << widthInputRemainder));
    const __mmask16 _rightOutputMask = static_cast<__mmask16>(~(~0 << widthOutputRemainder));

    const __m512 _weight = _mm512_set1_ps(1.0f / float(KernelWidth));

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _one = _mm512_set1_epi32(1);

    float* pOutput0 = Output;

    for (size_t c = 0; c < ChannelCount; c++) {
        const float* pInput0 = Input;

        __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);

        pInput0 += paddingLeftSeek;

        for (int i = 0; i < widthIterations; i++) {
            __m512 _b0_1 = _mm512_load_ps(pInput0);

            // Prologue
            __m512i _roll = _roll_left_1;

            __m512 _acc0 = _b0_1;

            __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

            // Inner Loop
            for (int64_t j = 2; j < KernelWidth; j++) {
                _roll = _mm512_sub_epi32(_roll, _one);

                _acc0 = _mm512_add_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
            }

            // Epilogue
            _acc0 = _mm512_add_ps(_acc0, _b0);

            _b0_0 = _b0_1;

            _acc0 = _mm512_mul_ps(_acc0, _weight);

            // Store
            _mm512_store_ps(pOutput0, _acc0);

            pInput0 += widthStep;
            pOutput0 += widthStep;
        }

        // Right Edge
        {
            __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);

            // Prologue
            __m512i _roll = _roll_left_1;

            __m512 _acc0 = _b0_1;

            __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

            // Inner Loop
            for (int64_t j = 2; j < KernelWidth; j++) {
                _roll = _mm512_sub_epi32(_roll, _one);

                _acc0 = _mm512_add_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
            }

            // Epilogue
            _acc0 = _mm512_add_ps(_acc0, _b0);

            _acc0 = _mm512_mul_ps(_acc0, _weight);

            if (widthOutputRemainder < widthInputRemainder) {
                // Store
                _mm512_store_ps(pOutput0, _acc0);

                pOutput0 += widthStep;

                _b0_0 = _b0_1;

                _b0_1 = _mm512_setzero_ps();

                // Prologue
                _roll = _roll_left_1;

                _acc0 = _b0_1;

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                // Inner Loop
                for (int64_t j = 2; j < KernelWidth; j++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_add_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                }

                // Epilogue
                _acc0 = _mm512_add_ps(_acc0, _b0);

                _acc0 = _mm512_mul_ps(_acc0, _weight);
            }

            // Store
            _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);
        }

        Input += InputSize;
    }
}

void
MlasPool1DSlidingKernelAvgWithPadK32S1(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                       size_t ChannelCount,
                                       const float* Input,
                                       float* Output)
{
    constexpr size_t Dimensions = 1;

    constexpr size_t WidthShapeIndex = 0;

    assert(WorkBlock->PoolingKind == MLAS_POOLING_KIND::MlasAveragePoolingIncludePad);

    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
    const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

    assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
    MLAS_UNREFERENCED_PARAMETER(PaddingRight);

    constexpr size_t widthStep = 32;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

    const int64_t widthIterations = InputWidthBase / widthStep;
    const int64_t widthInputRemainder = InputWidthBase % widthStep;
    const int64_t widthOutputRemainder = OutputWidth % widthStep;

    const int64_t kernelRemainder = KernelWidth % 16;

    const __mmask32 _leftInputMask = static_cast<__mmask32>(~0 << paddingLeftShift);

    const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMask);
    const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);

    const __mmask32 _rightInputMask = static_cast<__mmask32>(~(~0 << widthInputRemainder));

    const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
    const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);

    const __mmask32 _rightOutputMask = static_cast<__mmask32>(~(~0 << widthOutputRemainder));

    const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
    const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);

    const __m512 _weight = _mm512_set1_ps(1.0f / float(KernelWidth));

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _one = _mm512_set1_epi32(1);

    float* pOutput0 = Output;

    for (size_t c = 0; c < ChannelCount; c++) {
        const float* pInput0 = Input;

        __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift);
        __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

        pInput0 += paddingLeftSeek;

        for (int i = 0; i < widthIterations; i++) {
            __m512 _b0_1 = _mm512_load_ps(pInput0);
            __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

            // Prologue
            __m512i _roll = _roll_left_1;

            __m512 _acc0 = _b0_0;
            __m512 _acc1 = _b0_1;
            __m512 _acc2 = _b0_2;

            __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
            __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
            __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

            // Inner Loops
            for (int j = 1; j < kernelRemainder; j++) {
                _roll = _mm512_sub_epi32(_roll, _one);

                _acc0 = _mm512_add_ps(_acc0, _b0);
                _acc1 = _mm512_add_ps(_acc1, _b1);
                _acc2 = _mm512_add_ps(_acc2, _b2);

                _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
            }

            _acc2 = _mm512_add_ps(_acc2, _acc1);
            _acc1 = _mm512_add_ps(_acc1, _acc0);

            for (int64_t j = kernelRemainder; j < 15; j++) {
                _roll = _mm512_sub_epi32(_roll, _one);

                _acc1 = _mm512_add_ps(_acc1, _b1);
                _acc2 = _mm512_add_ps(_acc2, _b2);

                _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
            }

            // Epilogue
            _acc1 = _mm512_add_ps(_acc1, _b1);
            _acc2 = _mm512_add_ps(_acc2, _b2);

            _b0_f = _b0_1;
            _b0_0 = _b0_2;

            _acc1 = _mm512_mul_ps(_acc1, _weight);
            _acc2 = _mm512_mul_ps(_acc2, _weight);

            // Store
            _mm512_store_ps(pOutput0, _acc1);
            _mm512_store_ps(pOutput0 + 16, _acc2);

            pInput0 += widthStep;
            pOutput0 += widthStep;
        }

        // Right Edge
        {
            __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0);
            __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);

            // Prologue
            __m512i _roll = _roll_left_1;

            __m512 _acc0 = _b0_0;
            __m512 _acc1 = _b0_1;
            __m512 _acc2 = _b0_2;

            __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
            __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
            __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

            // Inner Loops
            for (int64_t j = 1; j < kernelRemainder; j++) {
                _roll = _mm512_sub_epi32(_roll, _one);

                _acc0 = _mm512_add_ps(_acc0, _b0);
                _acc1 = _mm512_add_ps(_acc1, _b1);
                _acc2 = _mm512_add_ps(_acc2, _b2);

                _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
            }

            _acc2 = _mm512_add_ps(_acc2, _acc1);
            _acc1 = _mm512_add_ps(_acc1, _acc0);

            for (int64_t j = kernelRemainder; j < 15; j++) {
                _roll = _mm512_sub_epi32(_roll, _one);

                _acc1 = _mm512_add_ps(_acc1, _b1);
                _acc2 = _mm512_add_ps(_acc2, _b2);

                _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
            }

            // Epilogue
            _acc1 = _mm512_add_ps(_acc1, _b1);
            _acc2 = _mm512_add_ps(_acc2, _b2);

            _acc1 = _mm512_mul_ps(_acc1, _weight);
            _acc2 = _mm512_mul_ps(_acc2, _weight);

            if (widthOutputRemainder < widthInputRemainder) {
                // Store
                _mm512_store_ps(pOutput0, _acc1);
                _mm512_store_ps(pOutput0 + 16, _acc2);

                pOutput0 += widthStep;

                _b0_f = _b0_1;
                _b0_0 = _b0_2;

                _b0_1 = _mm512_setzero_ps();
                _b0_2 = _mm512_setzero_ps();

                _roll = _roll_left_1;

                _acc0 = _b0_0;
                _acc1 = _b0_1;
                _acc2 = _b0_2;

                _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                // Inner Loops
                for (int j = 1; j < kernelRemainder; j++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_add_ps(_acc0, _b0);
                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                _acc2 = _mm512_add_ps(_acc2, _acc1);
                _acc1 = _mm512_add_ps(_acc1, _acc0);

                for (int64_t j = kernelRemainder; j < 15; j++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                // Epilogue
                _acc1 = _mm512_add_ps(_acc1, _b1);
                _acc2 = _mm512_add_ps(_acc2, _b2);

                _acc1 = _mm512_mul_ps(_acc1, _weight);
                _acc2 = _mm512_mul_ps(_acc2, _weight);
            }

            // Store
            _mm512_mask_store_ps(pOutput0, _rightOutputMask1, _acc1);
            _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _acc2);
        }

        Input += InputSize;
    }
}

void
MlasPool1DSlidingKernelAvgWithPad(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                  size_t ChannelCount,
                                  const float* Input,
                                  float* Output)
{
    constexpr size_t WidthShapeIndex = 0;

    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];

    if (KernelWidth <= 16) {
        MlasPool1DSlidingKernelAvgWithPadK17S1(WorkBlock, ChannelCount, Input, Output);
    } else if (KernelWidth <= 32) {
        MlasPool1DSlidingKernelAvgWithPadK32S1(WorkBlock, ChannelCount, Input, Output);
    } else
        return;
}

void
MlasPool1DSlidingKernelAvgNoPad(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                size_t ChannelCount,
                                const float* Input,
                                float* Output)
{
    constexpr size_t Dimensions = 1;
    constexpr size_t WidthShapeIndex = 0;

    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingLeftWidth = WorkBlock->Padding[WidthShapeIndex];
    const int64_t PaddingRightWidth = WorkBlock->Padding[Dimensions + WidthShapeIndex];

    assert(WorkBlock->PoolingKind == MLAS_POOLING_KIND::MlasAveragePoolingExcludePad);

    MLAS_POOL_WORK_BLOCK NewBlock{
        MlasAveragePoolingIncludePad,
        {WorkBlock->InputShape[0], WorkBlock->InputShape[1], WorkBlock->InputShape[2]},
        WorkBlock->InputSize,
        {WorkBlock->OutputShape[0], WorkBlock->OutputShape[1], WorkBlock->OutputShape[2]},
        {WorkBlock->KernelShape[0], WorkBlock->KernelShape[1], WorkBlock->KernelShape[2]},
        {WorkBlock->Padding[0], WorkBlock->Padding[1], WorkBlock->Padding[2], WorkBlock->Padding[3],
         WorkBlock->Padding[4], WorkBlock->Padding[5]},
        {WorkBlock->StrideShape[0], WorkBlock->StrideShape[1], WorkBlock->StrideShape[2]}};

    if (KernelWidth <= 16) {
        MlasPool1DSlidingKernelAvgWithPadK17S1(&NewBlock, ChannelCount, Input, Output);
    } else if (KernelWidth <= 32) {
        MlasPool1DSlidingKernelAvgWithPadK32S1(&NewBlock, ChannelCount, Input, Output);
    } else
        return;

    float shape = float(KernelWidth);

    float* pBegin = Output;

    for (int64_t i = PaddingLeftWidth; i > 0; i--) {
        *pBegin++ *= shape / (shape - float(i));
    }

    float* pEnd = Output + OutputWidth;

    for (int64_t i = PaddingRightWidth; i > 0; i--) {
        *--pEnd *= shape / (shape - float(i));
    }
}

void
MlasPool2DSlidingKernelMaxK17S1(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                size_t ChannelCount,
                                const float* Input,
                                float* Output)
{
    constexpr size_t Dimensions = 2;
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    assert(WorkBlock->PoolingKind == MLAS_POOLING_KIND::MlasMaximumPooling);

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;
    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingTop = WorkBlock->Padding[HeightShapeIndex];
    const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
    const int64_t PaddingBottom = WorkBlock->Padding[Dimensions + HeightShapeIndex];
    const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

    assert(OutputHeight == (InputHeight - KernelHeight + PaddingTop + PaddingBottom + 1));
    assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
    MLAS_UNREFERENCED_PARAMETER(InputHeight);
    MLAS_UNREFERENCED_PARAMETER(PaddingRight);

    constexpr size_t widthStep = 16;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const __mmask16 _leftInputMask = static_cast<__mmask16>((~0 << paddingLeftShift));

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

    const int64_t widthIterations = InputWidthBase / widthStep;
    const int64_t widthInputRemainder = InputWidthBase % widthStep;
    const int64_t widthOutputRemainder = OutputWidth % widthStep;

    const __mmask16 _rightInputMask = static_cast<__mmask16>(~(~0 << widthInputRemainder));
    const __mmask16 _rightOutputMask = static_cast<__mmask16>(~(~0 << widthOutputRemainder));

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _one = _mm512_set1_epi32(1);

    float* pOutputRow = Output;

    const __m512 _padding = _mm512_set1_ps(std::numeric_limits<float>::lowest());

    for (size_t c = 0; c < ChannelCount; c++) {
        const float* pInputRow = Input;

        // Outer Loop Prologue
        for (int64_t i = 0; i < PaddingTop; i++) {
            float* pOutput0 = pOutputRow;

            for (int64_t j = 0; j < widthIterations; j++) {
                _mm512_store_ps(pOutput0, _padding);

                pOutput0 += widthStep;
            }

            if (widthOutputRemainder < widthInputRemainder) {
                _mm512_store_ps(pOutput0, _padding);

                pOutput0 += widthStep;
            }

            _mm512_mask_store_ps(pOutput0, _rightOutputMask, _padding);

            pOutputRow += OutputWidth;
        }

        for (int64_t i = PaddingTop; i < KernelHeight; i++) {
            const float* pInput0 = pInputRow;

            float* pOutput0 = pOutputRow;

            __m512 _b0_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput0 - paddingLeftShift);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_1;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                // Inner Loop
                for (int64_t l = 2; l < KernelWidth; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_max_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                }

                // Epilogue
                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0_0 = _b0_1;

                // Store
                _mm512_store_ps(pOutput0, _acc0);

                float* pOutput1 = pOutput0;

                for (int64_t k = i; k-- > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_load_ps(pOutput1);

                    _acc1 = _mm512_max_ps(_acc0, _acc1);

                    _mm512_store_ps(pOutput1, _acc1);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput0);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_1;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                // Inner Loop
                for (int64_t l = 2; l < KernelWidth; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_max_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                }

                // Epilogue
                _acc0 = _mm512_max_ps(_acc0, _b0);

                if (widthOutputRemainder < widthInputRemainder) {
                    _mm512_store_ps(pOutput0, _acc0);

                    float* pOutput1 = pOutput0;

                    for (int64_t k = i; k-- > 0;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1 = _mm512_load_ps(pOutput1);

                        _acc1 = _mm512_max_ps(_acc0, _acc1);

                        _mm512_store_ps(pOutput1, _acc1);
                    }

                    pOutput0 += widthStep;

                    _b0_0 = _b0_1;

                    _b0_1 = _padding;

                    // Prologue
                    _roll = _roll_left_1;

                    _acc0 = _b0_1;

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                    // Inner Loop
                    for (int64_t l = 2; l < KernelWidth; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc0 = _mm512_max_ps(_acc0, _b0);

                        _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    }

                    // Epilogue
                    _acc0 = _mm512_max_ps(_acc0, _b0);
                }

                _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);

                float* pOutput1 = pOutput0;

                for (int64_t k = i; k-- > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

                    _acc1 = _mm512_max_ps(_acc0, _acc1);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
                }
            }
            pInputRow += InputWidth;

            pOutputRow += OutputWidth;
        }

        // Outer loop
        for (size_t i = KernelHeight; i < OutputHeight; i++) {
            const float* pInput0 = pInputRow;

            float* pOutput0 = pOutputRow;

            __m512 _b0_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput0 - paddingLeftShift);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_1;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                // Inner Loop
                for (int64_t l = 2; l < KernelWidth; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_max_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                }

                // Epilogue
                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0_0 = _b0_1;

                // Store
                _mm512_store_ps(pOutput0, _acc0);

                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; --k > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_load_ps(pOutput1);

                    _acc1 = _mm512_max_ps(_acc0, _acc1);

                    _mm512_store_ps(pOutput1, _acc1);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput0);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_1;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                // Inner Loop
                for (int64_t l = 2; l < KernelWidth; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_max_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                }

                // Epilogue
                _acc0 = _mm512_max_ps(_acc0, _b0);

                if (widthOutputRemainder < widthInputRemainder) {
                    _mm512_store_ps(pOutput0, _acc0);

                    float* pOutput1 = pOutput0;

                    for (int64_t k = KernelHeight; --k > 0;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1 = _mm512_load_ps(pOutput1);

                        _acc1 = _mm512_max_ps(_acc0, _acc1);

                        _mm512_store_ps(pOutput1, _acc1);
                    }

                    pOutput0 += widthStep;

                    _b0_0 = _b0_1;

                    _b0_1 = _padding;

                    // Prologue
                    _roll = _roll_left_1;

                    _acc0 = _b0_1;

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                    // Inner Loop
                    for (int64_t l = 2; l < KernelWidth; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc0 = _mm512_max_ps(_acc0, _b0);

                        _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    }

                    // Epilogue
                    _acc0 = _mm512_max_ps(_acc0, _b0);
                }

                _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);

                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; --k > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

                    _acc1 = _mm512_max_ps(_acc0, _acc1);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
                }
            }

            pInputRow += InputWidth;

            pOutputRow += OutputWidth;
        }

        // Outer Loop Epilogue
        for (int64_t i = 1; i < KernelHeight - PaddingBottom; i++) {
            const float* pInput0 = pInputRow;

            float* pOutput0 = pOutputRow;

            __m512 _b0_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput0 - paddingLeftShift);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_1;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                // Inner Loop
                for (int64_t l = 2; l < KernelWidth; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_max_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                }

                // Epilogue
                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0_0 = _b0_1;

                // Store
                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; k-- > i;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_load_ps(pOutput1);

                    _acc1 = _mm512_max_ps(_acc0, _acc1);

                    _mm512_store_ps(pOutput1, _acc1);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput0);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_1;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                // Inner Loop
                for (int64_t l = 2; l < KernelWidth; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_max_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                }

                // Epilogue
                _acc0 = _mm512_max_ps(_acc0, _b0);

                if (widthOutputRemainder < widthInputRemainder) {
                    float* pOutput1 = pOutput0;

                    for (int64_t k = KernelHeight; k-- > i;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1 = _mm512_load_ps(pOutput1);

                        _acc1 = _mm512_max_ps(_acc0, _acc1);

                        _mm512_store_ps(pOutput1, _acc1);
                    }

                    pOutput0 += widthStep;

                    _b0_0 = _b0_1;

                    _b0_1 = _padding;

                    // Prologue
                    _roll = _roll_left_1;

                    _acc0 = _b0_1;

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                    // Inner Loop
                    for (int64_t l = 2; l < KernelWidth; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc0 = _mm512_max_ps(_acc0, _b0);

                        _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    }

                    // Epilogue
                    _acc0 = _mm512_max_ps(_acc0, _b0);
                }

                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; k-- > i;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

                    _acc1 = _mm512_max_ps(_acc0, _acc1);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
                }
            }

            pInputRow += InputWidth;
        }

        Input += InputSize;
    }
}

void
MlasPool2DSlidingKernelMaxK3x3S1(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                 size_t ChannelCount,
                                 const float* Input,
                                 float* Output)
{
    constexpr size_t Dimensions = 2;
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    assert(WorkBlock->PoolingKind == MLAS_POOLING_KIND::MlasMaximumPooling);

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;
    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingTop = WorkBlock->Padding[HeightShapeIndex];
    const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
    const int64_t PaddingBottom = WorkBlock->Padding[Dimensions + HeightShapeIndex];
    const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

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

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _roll_left_2 =
        _mm512_set_epi32(29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14);

    const __m512 _padding = _mm512_set1_ps(std::numeric_limits<float>::lowest());

    float* pOutputRow = Output;

    for (size_t c = 0; c < ChannelCount; c++) {
        const float* pInputRow = Input;

        // Outer Loop Prologue
        for (int64_t i = 0; i < PaddingTop; i++) {
            float* pOutput0 = pOutputRow;

            for (int64_t j = 0; j < widthIterations; j++) {
                _mm512_store_ps(pOutput0, _padding);

                pOutput0 += widthStep;
            }

            if (widthOutputRemainder < widthInputRemainder) {
                _mm512_store_ps(pOutput0, _padding);

                pOutput0 += widthStep;
            }

            _mm512_mask_store_ps(pOutput0, _rightOutputMask, _padding);

            pOutputRow += OutputWidth;
        }

        for (int64_t i = PaddingTop; i < KernelHeight; i++) {
            const float* pInput0 = pInputRow;

            float* pOutput0 = pOutputRow;

            __m512 _b0_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput0 - paddingLeftShift);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                __m512 _acc0 = _mm512_max_ps(_b0_1, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0_0 = _b0_1;

                // Store
                _mm512_store_ps(pOutput0, _acc0);

                float* pOutput1 = pOutput0;

                for (int64_t k = i; k-- > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_load_ps(pOutput1);

                    _acc1 = _mm512_max_ps(_acc0, _acc1);

                    _mm512_store_ps(pOutput1, _acc1);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput0);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                __m512 _acc0 = _mm512_max_ps(_b0_1, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                if (widthOutputRemainder < widthInputRemainder) {
                    _mm512_store_ps(pOutput0, _acc0);

                    float* pOutput1 = pOutput0;

                    for (int64_t k = i; k-- > 0;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1 = _mm512_load_ps(pOutput1);

                        _acc1 = _mm512_max_ps(_acc0, _acc1);

                        _mm512_store_ps(pOutput1, _acc1);
                    }

                    pOutput0 += widthStep;

                    _b0_0 = _b0_1;

                    _b0_1 = _padding;

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                    _acc0 = _mm512_max_ps(_b0_1, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                    _acc0 = _mm512_max_ps(_acc0, _b0);
                }

                _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);

                float* pOutput1 = pOutput0;

                for (int64_t k = i; k-- > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

                    _acc1 = _mm512_max_ps(_acc0, _acc1);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
                }
            }

            pInputRow += InputWidth;

            pOutputRow += OutputWidth;
        }

        // Main Outer Loop
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

            __m512 _b0_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput0 - paddingLeftShift);
            __m512 _b1_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput1 - paddingLeftShift);
            __m512 _b2_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput2 - paddingLeftShift);
            __m512 _b3_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput3 - paddingLeftShift);

            pInput0 += paddingLeftSeek;
            pInput1 += paddingLeftSeek;
            pInput2 += paddingLeftSeek;
            pInput3 += paddingLeftSeek;

            for (int j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);
                __m512 _b1_1 = _mm512_load_ps(pInput1);
                __m512 _b2_1 = _mm512_load_ps(pInput2);
                __m512 _b3_1 = _mm512_load_ps(pInput3);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
                __m512 _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_1, _b1_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_1, _b2_1);
                __m512 _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_1, _b3_1);

                __m512 _acc0_0 = _mm512_max_ps(_b0_1, _b0);
                __m512 _acc1_0 = _mm512_max_ps(_b1_1, _b1);
                __m512 _acc2_0 = _mm512_max_ps(_b2_1, _b2);
                __m512 _acc3_0 = _mm512_max_ps(_b3_1, _b3);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
                _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
                _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
                _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

                _acc0_0 = _mm512_max_ps(_acc0_0, _b0);
                _acc1_0 = _mm512_max_ps(_acc1_0, _b1);
                _acc2_0 = _mm512_max_ps(_acc2_0, _b2);
                _acc3_0 = _mm512_max_ps(_acc3_0, _b3);

                _b0_0 = _b0_1;
                _b1_0 = _b1_1;
                _b2_0 = _b2_1;
                _b3_0 = _b3_1;

                __m512 _acc2_x = _mm512_load_ps(pOutput_2);
                __m512 _acc1_x = _mm512_load_ps(pOutput_1);

                // Prefix Sums
                _acc2_x = _mm512_max_ps(_acc2_x, _acc0_0);

                _acc0_0 = _mm512_max_ps(_acc0_0, _acc1_0);
                _acc1_0 = _mm512_max_ps(_acc1_0, _acc2_0);

                _acc1_x = _mm512_max_ps(_acc1_x, _acc0_0);

                _acc0_0 = _mm512_max_ps(_acc0_0, _acc2_0);
                _acc1_0 = _mm512_max_ps(_acc1_0, _acc3_0);
                _acc2_0 = _mm512_max_ps(_acc2_0, _acc3_0);

                // Store
                _mm512_store_ps(pOutput_2, _acc2_x);
                _mm512_store_ps(pOutput_1, _acc1_x);

                _mm512_store_ps(pOutput0, _acc0_0);
                _mm512_store_ps(pOutput1, _acc1_0);
                _mm512_store_ps(pOutput2, _acc2_0);
                _mm512_store_ps(pOutput3, _acc3_0);

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
                __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput0);
                __m512 _b1_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput1);
                __m512 _b2_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput2);
                __m512 _b3_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput3);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
                __m512 _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_1, _b1_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_1, _b2_1);
                __m512 _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_1, _b3_1);

                __m512 _acc0_0 = _mm512_max_ps(_b0_1, _b0);
                __m512 _acc1_0 = _mm512_max_ps(_b1_1, _b1);
                __m512 _acc2_0 = _mm512_max_ps(_b2_1, _b2);
                __m512 _acc3_0 = _mm512_max_ps(_b3_1, _b3);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
                _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
                _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
                _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

                _acc0_0 = _mm512_max_ps(_acc0_0, _b0);
                _acc1_0 = _mm512_max_ps(_acc1_0, _b1);
                _acc2_0 = _mm512_max_ps(_acc2_0, _b2);
                _acc3_0 = _mm512_max_ps(_acc3_0, _b3);

                if (widthOutputRemainder < widthInputRemainder) {
                    __m512 _acc2_x = _mm512_load_ps(pOutput_2);
                    __m512 _acc1_x = _mm512_load_ps(pOutput_1);

                    // Prefix Sums
                    _acc2_x = _mm512_max_ps(_acc2_x, _acc0_0);

                    _acc0_0 = _mm512_max_ps(_acc0_0, _acc1_0);
                    _acc1_0 = _mm512_max_ps(_acc1_0, _acc2_0);

                    _acc1_x = _mm512_max_ps(_acc1_x, _acc0_0);

                    _acc0_0 = _mm512_max_ps(_acc0_0, _acc2_0);
                    _acc1_0 = _mm512_max_ps(_acc1_0, _acc3_0);
                    _acc2_0 = _mm512_max_ps(_acc2_0, _acc3_0);

                    // Store
                    _mm512_store_ps(pOutput_2, _acc2_x);
                    _mm512_store_ps(pOutput_1, _acc1_x);

                    _mm512_store_ps(pOutput0, _acc0_0);
                    _mm512_store_ps(pOutput1, _acc1_0);
                    _mm512_store_ps(pOutput2, _acc2_0);
                    _mm512_store_ps(pOutput3, _acc3_0);

                    pOutput_2 += widthStep;
                    pOutput_1 += widthStep;
                    pOutput0 += widthStep;
                    pOutput1 += widthStep;
                    pOutput2 += widthStep;
                    pOutput3 += widthStep;

                    _b0_0 = _b0_1;
                    _b1_0 = _b1_1;
                    _b2_0 = _b2_1;
                    _b3_0 = _b3_1;

                    _b0_1 = _padding;
                    _b1_1 = _padding;
                    _b2_1 = _padding;
                    _b3_1 = _padding;

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
                    _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_1, _b1_1);
                    _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_1, _b2_1);
                    _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_1, _b3_1);

                    _acc0_0 = _mm512_max_ps(_b0_1, _b0);
                    _acc1_0 = _mm512_max_ps(_b1_1, _b1);
                    _acc2_0 = _mm512_max_ps(_b2_1, _b2);
                    _acc3_0 = _mm512_max_ps(_b3_1, _b3);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
                    _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
                    _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
                    _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

                    _acc0_0 = _mm512_max_ps(_acc0_0, _b0);
                    _acc1_0 = _mm512_max_ps(_acc1_0, _b1);
                    _acc2_0 = _mm512_max_ps(_acc2_0, _b2);
                    _acc3_0 = _mm512_max_ps(_acc3_0, _b3);
                }

                __m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_2);
                __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_1);

                // Prefix Sums
                _acc2_x = _mm512_max_ps(_acc2_x, _acc0_0);

                _acc0_0 = _mm512_max_ps(_acc0_0, _acc1_0);
                _acc1_0 = _mm512_max_ps(_acc1_0, _acc2_0);

                _acc1_x = _mm512_max_ps(_acc1_x, _acc0_0);

                _acc0_0 = _mm512_max_ps(_acc0_0, _acc2_0);
                _acc1_0 = _mm512_max_ps(_acc1_0, _acc3_0);
                _acc2_0 = _mm512_max_ps(_acc2_0, _acc3_0);

                // Store
                _mm512_mask_store_ps(pOutput_2, _rightOutputMask, _acc2_x);
                _mm512_mask_store_ps(pOutput_1, _rightOutputMask, _acc1_x);

                _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0_0);
                _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1_0);
                _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc2_0);
                _mm512_mask_store_ps(pOutput3, _rightOutputMask, _acc3_0);
            }

            pInputRow += heightStep * InputWidth;
            pOutputRow += heightStep * OutputWidth;
        }

        // Outer Tail
        for (size_t i = 0; i < heightRemainder; i++) {
            const float* pInput0 = pInputRow;

            float* pOutput0 = pOutputRow;
            float* pOutput_1 = pOutput0 - OutputWidth;
            float* pOutput_2 = pOutput_1 - OutputWidth;

            __m512 _b0_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput0 - paddingLeftShift);

            pInput0 += paddingLeftSeek;

            for (int j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                __m512 _acc0 = _mm512_max_ps(_b0_1, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                __m512 _acc2_x = _mm512_load_ps(pOutput_2);
                __m512 _acc1_x = _mm512_load_ps(pOutput_1);

                _b0_0 = _b0_1;

                // Prefix Sums
                _acc2_x = _mm512_max_ps(_acc2_x, _acc0);
                _acc1_x = _mm512_max_ps(_acc1_x, _acc0);

                // Store
                _mm512_store_ps(pOutput_2, _acc2_x);
                _mm512_store_ps(pOutput_1, _acc1_x);

                _mm512_store_ps(pOutput0, _acc0);

                pInput0 += widthStep;

                pOutput_2 += widthStep;
                pOutput_1 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput0);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                __m512 _acc0 = _mm512_max_ps(_b0_1, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                if (widthOutputRemainder < widthInputRemainder) {
                    __m512 _acc2_x = _mm512_load_ps(pOutput_2);
                    __m512 _acc1_x = _mm512_load_ps(pOutput_1);

                    _b0_0 = _b0_1;

                    // Prefix Sums
                    _acc2_x = _mm512_max_ps(_acc2_x, _acc0);
                    _acc1_x = _mm512_max_ps(_acc1_x, _acc0);

                    // Store
                    _mm512_store_ps(pOutput_2, _acc2_x);
                    _mm512_store_ps(pOutput_1, _acc1_x);

                    pOutput0 += widthStep;

                    _mm512_store_ps(pOutput0, _acc0);

                    _b0_0 = _b0_1;

                    _b0_1 = _padding;

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                    _acc0 = _mm512_max_ps(_b0_1, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                    _acc0 = _mm512_max_ps(_acc0, _b0);
                }

                __m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_2);
                __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_1);

                // Prefix Sums
                _acc2_x = _mm512_max_ps(_acc2_x, _acc0);
                _acc1_x = _mm512_max_ps(_acc1_x, _acc0);

                // Store
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

            __m512 _b0_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput0 - paddingLeftShift);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                __m512 _acc0 = _mm512_max_ps(_b0_1, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0_0 = _b0_1;

                // Store
                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; k-- > i;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_load_ps(pOutput1);

                    _acc1 = _mm512_max_ps(_acc0, _acc1);

                    _mm512_store_ps(pOutput1, _acc1);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput0);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                __m512 _acc0 = _mm512_max_ps(_b0_1, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                if (widthOutputRemainder < widthInputRemainder) {
                    float* pOutput1 = pOutput0;

                    for (int64_t k = KernelHeight; k-- > i;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1 = _mm512_load_ps(pOutput1);

                        _acc1 = _mm512_max_ps(_acc0, _acc1);

                        _mm512_store_ps(pOutput1, _acc1);
                    }

                    pOutput0 += widthStep;

                    _b0_0 = _b0_1;

                    _b0_1 = _padding;

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                    _acc0 = _mm512_max_ps(_b0_1, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                    _acc0 = _mm512_max_ps(_acc0, _b0);
                }

                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; k-- > i;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

                    _acc1 = _mm512_max_ps(_acc0, _acc1);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
                }
            }

            pInputRow += InputWidth;
        }

        Input += InputSize;
    }
}

void
MlasPool2DSlidingKernelMaxK5x5S1(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                 size_t ChannelCount,
                                 const float* Input,
                                 float* Output)
{
    constexpr size_t Dimensions = 2;
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    assert(WorkBlock->PoolingKind == MLAS_POOLING_KIND::MlasMaximumPooling);

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;
    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingTop = WorkBlock->Padding[HeightShapeIndex];
    const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
    const int64_t PaddingBottom = WorkBlock->Padding[Dimensions + HeightShapeIndex];
    const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

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

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _roll_left_2 =
        _mm512_set_epi32(29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14);
    const __m512i _roll_left_3 =
        _mm512_set_epi32(28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13);
    const __m512i _roll_left_4 =
        _mm512_set_epi32(27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12);

    const __m512 _padding = _mm512_set1_ps(std::numeric_limits<float>::lowest());

    float* pOutputRow = Output;

    for (size_t c = 0; c < ChannelCount; c++) {
        const float* pInputRow = Input;

        // Outer Loop Prologue
        for (int64_t i = 0; i < PaddingTop; i++) {
            float* pOutput0 = pOutputRow;

            for (int64_t j = 0; j < widthIterations; j++) {
                _mm512_store_ps(pOutput0, _padding);

                pOutput0 += widthStep;
            }

            if (widthOutputRemainder < widthInputRemainder) {
                _mm512_store_ps(pOutput0, _padding);

                pOutput0 += widthStep;
            }

            _mm512_mask_store_ps(pOutput0, _rightOutputMask, _padding);

            pOutputRow += OutputWidth;
        }

        for (int64_t i = PaddingTop; i < KernelHeight; i++) {
            const float* pInput0 = pInputRow;

            float* pOutput0 = pOutputRow;

            __m512 _b0_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput0 - paddingLeftShift);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                __m512 _acc0 = _mm512_max_ps(_b0_1, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0_0 = _b0_1;

                // Store
                _mm512_store_ps(pOutput0, _acc0);

                float* pOutput1 = pOutput0;

                for (int64_t k = i; k-- > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_load_ps(pOutput1);

                    _acc1 = _mm512_max_ps(_acc0, _acc1);

                    _mm512_store_ps(pOutput1, _acc1);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput0);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                __m512 _acc0 = _mm512_max_ps(_b0_1, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                if (widthOutputRemainder < widthInputRemainder) {
                    _mm512_store_ps(pOutput0, _acc0);

                    float* pOutput1 = pOutput0;

                    for (int64_t k = i; k-- > 0;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1 = _mm512_load_ps(pOutput1);

                        _acc1 = _mm512_max_ps(_acc0, _acc1);

                        _mm512_store_ps(pOutput1, _acc1);
                    }

                    pOutput0 += widthStep;

                    _b0_0 = _b0_1;

                    _b0_1 = _padding;

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                    _acc0 = _mm512_max_ps(_b0_1, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                    _acc0 = _mm512_max_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);

                    _acc0 = _mm512_max_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

                    _acc0 = _mm512_max_ps(_acc0, _b0);
                }

                _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);

                float* pOutput1 = pOutput0;

                for (int64_t k = i; k-- > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

                    _acc1 = _mm512_max_ps(_acc0, _acc1);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
                }
            }

            pInputRow += InputWidth;

            pOutputRow += OutputWidth;
        }

        // Main Outer Loop
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

            __m512 _b0_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput0 - paddingLeftShift);
            __m512 _b1_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput1 - paddingLeftShift);
            __m512 _b2_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput2 - paddingLeftShift);
            __m512 _b3_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput3 - paddingLeftShift);

            pInput0 += paddingLeftSeek;
            pInput1 += paddingLeftSeek;
            pInput2 += paddingLeftSeek;
            pInput3 += paddingLeftSeek;

            for (int j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);
                __m512 _b1_1 = _mm512_load_ps(pInput1);
                __m512 _b2_1 = _mm512_load_ps(pInput2);
                __m512 _b3_1 = _mm512_load_ps(pInput3);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
                __m512 _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_1, _b1_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_1, _b2_1);
                __m512 _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_1, _b3_1);

                __m512 _acc0_0 = _mm512_max_ps(_b0_1, _b0);
                __m512 _acc1_0 = _mm512_max_ps(_b1_1, _b1);
                __m512 _acc2_0 = _mm512_max_ps(_b2_1, _b2);
                __m512 _acc3_0 = _mm512_max_ps(_b3_1, _b3);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
                _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
                _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
                _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

                _acc0_0 = _mm512_max_ps(_acc0_0, _b0);
                _acc1_0 = _mm512_max_ps(_acc1_0, _b1);
                _acc2_0 = _mm512_max_ps(_acc2_0, _b2);
                _acc3_0 = _mm512_max_ps(_acc3_0, _b3);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
                _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_3, _b1_1);
                _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_3, _b2_1);
                _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_3, _b3_1);

                _acc0_0 = _mm512_max_ps(_acc0_0, _b0);
                _acc1_0 = _mm512_max_ps(_acc1_0, _b1);
                _acc2_0 = _mm512_max_ps(_acc2_0, _b2);
                _acc3_0 = _mm512_max_ps(_acc3_0, _b3);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);
                _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_4, _b1_1);
                _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_4, _b2_1);
                _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_4, _b3_1);

                _acc0_0 = _mm512_max_ps(_acc0_0, _b0);
                _acc1_0 = _mm512_max_ps(_acc1_0, _b1);
                _acc2_0 = _mm512_max_ps(_acc2_0, _b2);
                _acc3_0 = _mm512_max_ps(_acc3_0, _b3);

                _b0_0 = _b0_1;
                _b1_0 = _b1_1;
                _b2_0 = _b2_1;
                _b3_0 = _b3_1;

                __m512 _acc4_x = _mm512_load_ps(pOutput_4);
                __m512 _acc3_x = _mm512_load_ps(pOutput_3);
                __m512 _acc2_x = _mm512_load_ps(pOutput_2);
                __m512 _acc1_x = _mm512_load_ps(pOutput_1);

                // Prefix Sums
                _acc4_x = _mm512_max_ps(_acc4_x, _acc0_0);
                _acc2_x = _mm512_max_ps(_acc2_x, _acc2_0);

                _acc0_0 = _mm512_max_ps(_acc0_0, _acc1_0);
                _acc2_0 = _mm512_max_ps(_acc2_0, _acc3_0);

                _acc3_x = _mm512_max_ps(_acc3_x, _acc0_0);
                _acc2_x = _mm512_max_ps(_acc2_x, _acc0_0);

                _acc0_0 = _mm512_max_ps(_acc0_0, _acc2_0);
                _acc1_0 = _mm512_max_ps(_acc1_0, _acc2_0);

                _acc1_x = _mm512_max_ps(_acc1_x, _acc0_0);

                // Store
                _mm512_store_ps(pOutput_4, _acc4_x);
                _mm512_store_ps(pOutput_3, _acc3_x);
                _mm512_store_ps(pOutput_2, _acc2_x);
                _mm512_store_ps(pOutput_1, _acc1_x);

                _mm512_store_ps(pOutput0, _acc0_0);
                _mm512_store_ps(pOutput1, _acc1_0);
                _mm512_store_ps(pOutput2, _acc2_0);
                _mm512_store_ps(pOutput3, _acc3_0);

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
                __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput0);
                __m512 _b1_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput1);
                __m512 _b2_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput2);
                __m512 _b3_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput3);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
                __m512 _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_1, _b1_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_1, _b2_1);
                __m512 _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_1, _b3_1);

                __m512 _acc0_0 = _mm512_max_ps(_b0_1, _b0);
                __m512 _acc1_0 = _mm512_max_ps(_b1_1, _b1);
                __m512 _acc2_0 = _mm512_max_ps(_b2_1, _b2);
                __m512 _acc3_0 = _mm512_max_ps(_b3_1, _b3);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
                _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
                _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
                _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

                _acc0_0 = _mm512_max_ps(_acc0_0, _b0);
                _acc1_0 = _mm512_max_ps(_acc1_0, _b1);
                _acc2_0 = _mm512_max_ps(_acc2_0, _b2);
                _acc3_0 = _mm512_max_ps(_acc3_0, _b3);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
                _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_3, _b1_1);
                _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_3, _b2_1);
                _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_3, _b3_1);

                _acc0_0 = _mm512_max_ps(_acc0_0, _b0);
                _acc1_0 = _mm512_max_ps(_acc1_0, _b1);
                _acc2_0 = _mm512_max_ps(_acc2_0, _b2);
                _acc3_0 = _mm512_max_ps(_acc3_0, _b3);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);
                _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_4, _b1_1);
                _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_4, _b2_1);
                _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_4, _b3_1);

                _acc0_0 = _mm512_max_ps(_acc0_0, _b0);
                _acc1_0 = _mm512_max_ps(_acc1_0, _b1);
                _acc2_0 = _mm512_max_ps(_acc2_0, _b2);
                _acc3_0 = _mm512_max_ps(_acc3_0, _b3);

                if (widthOutputRemainder < widthInputRemainder) {
                    __m512 _acc4_x = _mm512_load_ps(pOutput_4);
                    __m512 _acc3_x = _mm512_load_ps(pOutput_3);
                    __m512 _acc2_x = _mm512_load_ps(pOutput_2);
                    __m512 _acc1_x = _mm512_load_ps(pOutput_1);

                    // Prefix Sums
                    _acc4_x = _mm512_max_ps(_acc4_x, _acc0_0);
                    _acc2_x = _mm512_max_ps(_acc2_x, _acc2_0);

                    _acc0_0 = _mm512_max_ps(_acc0_0, _acc1_0);
                    _acc2_0 = _mm512_max_ps(_acc2_0, _acc3_0);

                    _acc3_x = _mm512_max_ps(_acc3_x, _acc0_0);
                    _acc2_x = _mm512_max_ps(_acc2_x, _acc0_0);

                    _acc0_0 = _mm512_max_ps(_acc0_0, _acc2_0);
                    _acc1_0 = _mm512_max_ps(_acc1_0, _acc2_0);

                    _acc1_x = _mm512_max_ps(_acc1_x, _acc0_0);

                    // Store
                    _mm512_store_ps(pOutput_4, _acc4_x);
                    _mm512_store_ps(pOutput_3, _acc3_x);
                    _mm512_store_ps(pOutput_2, _acc2_x);
                    _mm512_store_ps(pOutput_1, _acc1_x);

                    _mm512_store_ps(pOutput0, _acc0_0);
                    _mm512_store_ps(pOutput1, _acc1_0);
                    _mm512_store_ps(pOutput2, _acc2_0);
                    _mm512_store_ps(pOutput3, _acc3_0);

                    pOutput_4 += widthStep;
                    pOutput_3 += widthStep;
                    pOutput_2 += widthStep;
                    pOutput_1 += widthStep;
                    pOutput0 += widthStep;
                    pOutput1 += widthStep;
                    pOutput2 += widthStep;
                    pOutput3 += widthStep;

                    _b0_0 = _b0_1;
                    _b1_0 = _b1_1;
                    _b2_0 = _b2_1;
                    _b3_0 = _b3_1;

                    _b0_1 = _padding;
                    _b1_1 = _padding;
                    _b2_1 = _padding;
                    _b3_1 = _padding;

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);
                    _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_1, _b1_1);
                    _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_1, _b2_1);
                    _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_1, _b3_1);

                    _acc0_0 = _mm512_max_ps(_b0_1, _b0);
                    _acc1_0 = _mm512_max_ps(_b1_1, _b1);
                    _acc2_0 = _mm512_max_ps(_b2_1, _b2);
                    _acc3_0 = _mm512_max_ps(_b3_1, _b3);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
                    _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
                    _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
                    _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

                    _acc0_0 = _mm512_max_ps(_acc0_0, _b0);
                    _acc1_0 = _mm512_max_ps(_acc1_0, _b1);
                    _acc2_0 = _mm512_max_ps(_acc2_0, _b2);
                    _acc3_0 = _mm512_max_ps(_acc3_0, _b3);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
                    _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_3, _b1_1);
                    _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_3, _b2_1);
                    _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_3, _b3_1);

                    _acc0_0 = _mm512_max_ps(_acc0_0, _b0);
                    _acc1_0 = _mm512_max_ps(_acc1_0, _b1);
                    _acc2_0 = _mm512_max_ps(_acc2_0, _b2);
                    _acc3_0 = _mm512_max_ps(_acc3_0, _b3);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);
                    _b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_4, _b1_1);
                    _b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_4, _b2_1);
                    _b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_4, _b3_1);

                    _acc0_0 = _mm512_max_ps(_acc0_0, _b0);
                    _acc1_0 = _mm512_max_ps(_acc1_0, _b1);
                    _acc2_0 = _mm512_max_ps(_acc2_0, _b2);
                    _acc3_0 = _mm512_max_ps(_acc3_0, _b3);
                }

                __m512 _acc4_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_4);
                __m512 _acc3_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_3);
                __m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_2);
                __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_1);

                // Prefix Sums
                _acc4_x = _mm512_max_ps(_acc4_x, _acc0_0);
                _acc2_x = _mm512_max_ps(_acc2_x, _acc2_0);

                _acc0_0 = _mm512_max_ps(_acc0_0, _acc1_0);
                _acc2_0 = _mm512_max_ps(_acc2_0, _acc3_0);

                _acc3_x = _mm512_max_ps(_acc3_x, _acc0_0);
                _acc2_x = _mm512_max_ps(_acc2_x, _acc0_0);

                _acc0_0 = _mm512_max_ps(_acc0_0, _acc2_0);
                _acc1_0 = _mm512_max_ps(_acc1_0, _acc2_0);

                _acc1_x = _mm512_max_ps(_acc1_x, _acc0_0);

                // Store
                _mm512_mask_store_ps(pOutput_4, _rightOutputMask, _acc4_x);
                _mm512_mask_store_ps(pOutput_3, _rightOutputMask, _acc3_x);
                _mm512_mask_store_ps(pOutput_2, _rightOutputMask, _acc2_x);
                _mm512_mask_store_ps(pOutput_1, _rightOutputMask, _acc1_x);

                _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0_0);
                _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1_0);
                _mm512_mask_store_ps(pOutput2, _rightOutputMask, _acc2_0);
                _mm512_mask_store_ps(pOutput3, _rightOutputMask, _acc3_0);
            }

            pInputRow += heightStep * InputWidth;
            pOutputRow += heightStep * OutputWidth;
        }

        // Outer Tail
        for (size_t i = 0; i < heightRemainder; i++) {
            const float* pInput0 = pInputRow;

            float* pOutput0 = pOutputRow;
            float* pOutput_1 = pOutput0 - OutputWidth;
            float* pOutput_2 = pOutput_1 - OutputWidth;
            float* pOutput_3 = pOutput_2 - OutputWidth;
            float* pOutput_4 = pOutput_3 - OutputWidth;

            __m512 _b0_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput0 - paddingLeftShift);

            pInput0 += paddingLeftSeek;

            for (int j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                __m512 _acc0 = _mm512_max_ps(_b0_1, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                __m512 _acc4_x = _mm512_load_ps(pOutput_4);
                __m512 _acc3_x = _mm512_load_ps(pOutput_3);
                __m512 _acc2_x = _mm512_load_ps(pOutput_2);
                __m512 _acc1_x = _mm512_load_ps(pOutput_1);

                _b0_0 = _b0_1;

                // Prefix Sums
                _acc4_x = _mm512_max_ps(_acc4_x, _acc0);
                _acc3_x = _mm512_max_ps(_acc3_x, _acc0);
                _acc2_x = _mm512_max_ps(_acc2_x, _acc0);
                _acc1_x = _mm512_max_ps(_acc1_x, _acc0);

                // Store
                _mm512_store_ps(pOutput_4, _acc4_x);
                _mm512_store_ps(pOutput_3, _acc3_x);
                _mm512_store_ps(pOutput_2, _acc2_x);
                _mm512_store_ps(pOutput_1, _acc1_x);

                _mm512_store_ps(pOutput0, _acc0);

                pInput0 += widthStep;

                pOutput_4 += widthStep;
                pOutput_3 += widthStep;
                pOutput_2 += widthStep;
                pOutput_1 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput0);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                __m512 _acc0 = _mm512_max_ps(_b0_1, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                if (widthOutputRemainder < widthInputRemainder) {
                    __m512 _acc4_x = _mm512_load_ps(pOutput_4);
                    __m512 _acc3_x = _mm512_load_ps(pOutput_3);
                    __m512 _acc2_x = _mm512_load_ps(pOutput_2);
                    __m512 _acc1_x = _mm512_load_ps(pOutput_1);

                    _b0_0 = _b0_1;

                    _b0_1 = _padding;

                    // Prefix Sums
                    _acc4_x = _mm512_max_ps(_acc4_x, _acc0);
                    _acc3_x = _mm512_max_ps(_acc3_x, _acc0);
                    _acc2_x = _mm512_max_ps(_acc2_x, _acc0);
                    _acc1_x = _mm512_max_ps(_acc1_x, _acc0);

                    // Store
                    _mm512_store_ps(pOutput_4, _acc4_x);
                    _mm512_store_ps(pOutput_3, _acc3_x);
                    _mm512_store_ps(pOutput_2, _acc2_x);
                    _mm512_store_ps(pOutput_1, _acc1_x);

                    _mm512_store_ps(pOutput0, _acc0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                    _acc0 = _mm512_max_ps(_b0_1, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                    _acc0 = _mm512_max_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);

                    _acc0 = _mm512_max_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

                    _acc0 = _mm512_max_ps(_acc0, _b0);
                }

                __m512 _acc4_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_4);
                __m512 _acc3_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_3);
                __m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_2);
                __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask, pOutput_1);

                // Prefix Sums
                _acc4_x = _mm512_max_ps(_acc4_x, _acc0);
                _acc3_x = _mm512_max_ps(_acc3_x, _acc0);
                _acc2_x = _mm512_max_ps(_acc2_x, _acc0);
                _acc1_x = _mm512_max_ps(_acc1_x, _acc0);

                // Store
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

            __m512 _b0_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask, pInput0 - paddingLeftShift);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                __m512 _acc0 = _mm512_max_ps(_b0_1, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0_0 = _b0_1;

                // Store
                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; k-- > i;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_load_ps(pOutput1);

                    _acc1 = _mm512_max_ps(_acc0, _acc1);

                    _mm512_store_ps(pOutput1, _acc1);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput0);

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                __m512 _acc0 = _mm512_max_ps(_b0_1, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

                _acc0 = _mm512_max_ps(_acc0, _b0);

                if (widthOutputRemainder < widthInputRemainder) {
                    float* pOutput1 = pOutput0;

                    for (int64_t k = KernelHeight; k-- > i;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1 = _mm512_load_ps(pOutput1);

                        _acc1 = _mm512_max_ps(_acc0, _acc1);

                        _mm512_store_ps(pOutput1, _acc1);
                    }

                    pOutput0 += widthStep;

                    _b0_0 = _b0_1;

                    _b0_1 = _padding;

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_1, _b0_1);

                    _acc0 = _mm512_max_ps(_b0_1, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);

                    _acc0 = _mm512_max_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);

                    _acc0 = _mm512_max_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);

                    _acc0 = _mm512_max_ps(_acc0, _b0);
                }

                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; k-- > i;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

                    _acc1 = _mm512_max_ps(_acc0, _acc1);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
                }
            }

            pInputRow += InputWidth;
        }

        Input += InputSize;
    }
}

void
MlasPool2DSlidingKernelMaxK32S1(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                size_t ChannelCount,
                                const float* Input,
                                float* Output)
{
    constexpr size_t Dimensions = 2;
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    assert(WorkBlock->PoolingKind == MLAS_POOLING_KIND::MlasMaximumPooling);

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;
    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingTop = WorkBlock->Padding[HeightShapeIndex];
    const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
    const int64_t PaddingBottom = WorkBlock->Padding[Dimensions + HeightShapeIndex];
    const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

    assert(OutputHeight == (InputHeight - KernelHeight + PaddingTop + PaddingBottom + 1));
    assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
    MLAS_UNREFERENCED_PARAMETER(InputHeight);
    MLAS_UNREFERENCED_PARAMETER(PaddingRight);

    constexpr size_t widthStep = 32;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

    const int64_t widthIterations = InputWidthBase / widthStep;
    const int64_t widthInputRemainder = InputWidthBase % widthStep;
    const int64_t widthOutputRemainder = OutputWidth % widthStep;

    const int64_t kernelRemainder = KernelWidth % 16;

    const __mmask32 _leftInputMask = static_cast<__mmask32>(~0 << paddingLeftShift);

    const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMask);
    const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);

    const __mmask32 _rightInputMask = static_cast<__mmask32>(~(~0 << widthInputRemainder));

    const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
    const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);

    const __mmask32 _rightOutputMask = static_cast<__mmask32>(~(~0 << widthOutputRemainder));

    const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
    const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _one = _mm512_set1_epi32(1);

    float* pOutputRow = Output;

    const __m512 _padding = _mm512_set1_ps(std::numeric_limits<float>::lowest());

    for (size_t c = 0; c < ChannelCount; c++) {
        const float* pInputRow = Input;

        // Outer Loop Prologue
        for (int64_t i = 0; i < PaddingTop; i++) {
            float* pOutput0 = pOutputRow;

            for (int64_t j = 0; j < widthIterations; j++) {
                _mm512_store_ps(pOutput0, _padding);

                _mm512_store_ps(pOutput0 + 16, _padding);

                pOutput0 += widthStep;
            }

            if (widthOutputRemainder < widthInputRemainder) {
                _mm512_store_ps(pOutput0, _padding);

                _mm512_store_ps(pOutput0 + 16, _padding);

                pOutput0 += widthStep;
            }

            _mm512_mask_store_ps(pOutput0, _rightOutputMask1, _padding);

            _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _padding);

            pOutputRow += OutputWidth;
        }

        for (int64_t i = PaddingTop; i < KernelHeight; i++) {
            const float* pInput0 = pInputRow;

            float* pOutput0 = pOutputRow;

            __m512 _b0_f =
                _mm512_mask_load_ps(_padding, _leftInputMask1, pInput0 - paddingLeftShift);
            __m512 _b0_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask2, pInput0 - paddingLeftShift + 16);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);
                __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_0;
                __m512 _acc1 = _b0_1;
                __m512 _acc2 = _b0_2;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                // Inner Loops
                for (int64_t l = 1; l < kernelRemainder; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_max_ps(_acc0, _b0);
                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                _acc2 = _mm512_max_ps(_acc2, _acc1);
                _acc1 = _mm512_max_ps(_acc1, _acc0);

                for (int64_t l = kernelRemainder; l < 15; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);

                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                // Epilogue
                _acc1 = _mm512_max_ps(_acc1, _b1);
                _acc2 = _mm512_max_ps(_acc2, _b2);

                _b0_f = _b0_1;
                _b0_0 = _b0_2;

                // Store
                _mm512_store_ps(pOutput0, _acc1);
                _mm512_store_ps(pOutput0 + 16, _acc2);

                float* pOutput1 = pOutput0;

                for (int64_t k = i; k-- > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1_x = _mm512_load_ps(pOutput1);
                    __m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

                    _acc1_x = _mm512_max_ps(_acc1_x, _acc1);
                    _acc2_x = _mm512_max_ps(_acc2_x, _acc2);

                    _mm512_store_ps(pOutput1, _acc1_x);
                    _mm512_store_ps(pOutput1 + 16, _acc2_x);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask1, pInput0);
                __m512 _b0_2 = _mm512_mask_load_ps(_padding, _rightInputMask2, pInput0 + 16);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_0;
                __m512 _acc1 = _b0_1;
                __m512 _acc2 = _b0_2;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                // Inner Loops
                for (int l = 1; l < kernelRemainder; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_max_ps(_acc0, _b0);
                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                _acc2 = _mm512_max_ps(_acc2, _acc1);
                _acc1 = _mm512_max_ps(_acc1, _acc0);

                for (int64_t l = kernelRemainder; l < 15; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);

                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                // Epilogue
                _acc1 = _mm512_max_ps(_acc1, _b1);
                _acc2 = _mm512_max_ps(_acc2, _b2);

                if (widthOutputRemainder < widthInputRemainder) {
                    // Store
                    _mm512_store_ps(pOutput0, _acc1);
                    _mm512_store_ps(pOutput0 + 16, _acc2);

                    float* pOutput1 = pOutput0;

                    for (int64_t k = i; k-- > 0;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1_x = _mm512_load_ps(pOutput1);
                        __m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

                        _acc1_x = _mm512_max_ps(_acc1_x, _acc1);
                        _acc2_x = _mm512_max_ps(_acc2_x, _acc2);

                        _mm512_store_ps(pOutput1, _acc1_x);
                        _mm512_store_ps(pOutput1 + 16, _acc2_x);
                    }

                    pOutput0 += widthStep;

                    _b0_f = _b0_1;
                    _b0_0 = _b0_2;

                    _b0_1 = _padding;
                    _b0_2 = _padding;

                    // Prologue
                    _roll = _roll_left_1;

                    _acc0 = _b0_0;
                    _acc1 = _b0_1;
                    _acc2 = _b0_2;

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                    // Inner Loops
                    for (int l = 1; l < kernelRemainder; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc0 = _mm512_max_ps(_acc0, _b0);
                        _acc1 = _mm512_max_ps(_acc1, _b1);
                        _acc2 = _mm512_max_ps(_acc2, _b2);

                        _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                        _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                        _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                    }

                    _acc2 = _mm512_max_ps(_acc2, _acc1);
                    _acc1 = _mm512_max_ps(_acc1, _acc0);

                    for (int64_t l = kernelRemainder; l < 15; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc1 = _mm512_max_ps(_acc1, _b1);
                        _acc2 = _mm512_max_ps(_acc2, _b2);

                        _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                        _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                    }

                    // Epilogue
                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);
                }

                _mm512_mask_store_ps(pOutput0, _rightOutputMask1, _acc1);
                _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _acc2);

                float* pOutput1 = pOutput0;

                for (int64_t k = i; k-- > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask1, pOutput1);
                    __m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask2, pOutput1 + 16);

                    _acc1_x = _mm512_max_ps(_acc1_x, _acc1);
                    _acc2_x = _mm512_max_ps(_acc2_x, _acc2);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask1, _acc1_x);
                    _mm512_mask_store_ps(pOutput1 + 16, _rightOutputMask2, _acc2_x);
                }
            }

            pInputRow += InputWidth;

            pOutputRow += OutputWidth;
        }

        // Outer loop
        for (size_t i = KernelHeight; i < OutputHeight; i++) {
            const float* pInput0 = pInputRow;

            float* pOutput0 = pOutputRow;

            __m512 _b0_f =
                _mm512_mask_load_ps(_padding, _leftInputMask1, pInput0 - paddingLeftShift);
            __m512 _b0_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask2, pInput0 - paddingLeftShift + 16);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);
                __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_0;
                __m512 _acc1 = _b0_1;
                __m512 _acc2 = _b0_2;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                // Inner Loops
                for (int64_t l = 1; l < kernelRemainder; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_max_ps(_acc0, _b0);
                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                _acc2 = _mm512_max_ps(_acc2, _acc1);
                _acc1 = _mm512_max_ps(_acc1, _acc0);

                for (int64_t l = kernelRemainder; l < 15; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);

                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                // Epilogue
                _acc1 = _mm512_max_ps(_acc1, _b1);
                _acc2 = _mm512_max_ps(_acc2, _b2);

                _b0_f = _b0_1;
                _b0_0 = _b0_2;

                // Store
                _mm512_store_ps(pOutput0, _acc1);
                _mm512_store_ps(pOutput0 + 16, _acc2);

                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; --k > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1_x = _mm512_load_ps(pOutput1);
                    __m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

                    _acc1_x = _mm512_max_ps(_acc1_x, _acc1);
                    _acc2_x = _mm512_max_ps(_acc2_x, _acc2);

                    _mm512_store_ps(pOutput1, _acc1_x);
                    _mm512_store_ps(pOutput1 + 16, _acc2_x);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask1, pInput0);
                __m512 _b0_2 = _mm512_mask_load_ps(_padding, _rightInputMask2, pInput0 + 16);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_0;
                __m512 _acc1 = _b0_1;
                __m512 _acc2 = _b0_2;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                // Inner Loops
                for (int l = 1; l < kernelRemainder; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_max_ps(_acc0, _b0);
                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                _acc2 = _mm512_max_ps(_acc2, _acc1);
                _acc1 = _mm512_max_ps(_acc1, _acc0);

                for (int64_t l = kernelRemainder; l < 15; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);

                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                // Epilogue
                _acc1 = _mm512_max_ps(_acc1, _b1);
                _acc2 = _mm512_max_ps(_acc2, _b2);

                if (widthOutputRemainder < widthInputRemainder) {
                    _mm512_store_ps(pOutput0, _acc0);

                    float* pOutput1 = pOutput0;

                    for (int64_t k = KernelHeight; --k > 0;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1_x = _mm512_load_ps(pOutput1);
                        __m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

                        _acc1_x = _mm512_max_ps(_acc1_x, _acc1);
                        _acc2_x = _mm512_max_ps(_acc2_x, _acc2);

                        _mm512_store_ps(pOutput1, _acc1_x);
                        _mm512_store_ps(pOutput1 + 16, _acc2_x);
                    }

                    pOutput0 += widthStep;

                    _b0_f = _b0_1;
                    _b0_0 = _b0_2;

                    _b0_1 = _padding;
                    _b0_2 = _padding;

                    // Prologue
                    _roll = _roll_left_1;

                    _acc0 = _b0_0;
                    _acc1 = _b0_1;
                    _acc2 = _b0_2;

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                    // Inner Loops
                    for (int l = 1; l < kernelRemainder; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc0 = _mm512_max_ps(_acc0, _b0);
                        _acc1 = _mm512_max_ps(_acc1, _b1);
                        _acc2 = _mm512_max_ps(_acc2, _b2);

                        _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                        _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                        _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                    }

                    _acc2 = _mm512_max_ps(_acc2, _acc1);
                    _acc1 = _mm512_max_ps(_acc1, _acc0);

                    for (int64_t l = kernelRemainder; l < 15; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc1 = _mm512_max_ps(_acc1, _b1);
                        _acc2 = _mm512_max_ps(_acc2, _b2);

                        _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                        _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                    }

                    // Epilogue
                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);
                }

                _mm512_mask_store_ps(pOutput0, _rightOutputMask1, _acc1);
                _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _acc2);

                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; --k > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask1, pOutput1);
                    __m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask2, pOutput1 + 16);

                    _acc1_x = _mm512_max_ps(_acc1_x, _acc1);
                    _acc2_x = _mm512_max_ps(_acc2_x, _acc2);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask1, _acc1_x);
                    _mm512_mask_store_ps(pOutput1 + 16, _rightOutputMask2, _acc2_x);
                }
            }

            pInputRow += InputWidth;

            pOutputRow += OutputWidth;
        }

        // Outer Loop Epilogue
        for (int64_t i = 1; i < KernelHeight - PaddingBottom; i++) {
            const float* pInput0 = pInputRow;

            float* pOutput0 = pOutputRow;

            __m512 _b0_f =
                _mm512_mask_load_ps(_padding, _leftInputMask1, pInput0 - paddingLeftShift);
            __m512 _b0_0 =
                _mm512_mask_load_ps(_padding, _leftInputMask2, pInput0 - paddingLeftShift + 16);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);
                __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_0;
                __m512 _acc1 = _b0_1;
                __m512 _acc2 = _b0_2;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                // Inner Loops
                for (int64_t l = 1; l < kernelRemainder; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_max_ps(_acc0, _b0);
                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                _acc2 = _mm512_max_ps(_acc2, _acc1);
                _acc1 = _mm512_max_ps(_acc1, _acc0);

                for (int64_t l = kernelRemainder; l < 15; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);

                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                // Epilogue
                _acc1 = _mm512_max_ps(_acc1, _b1);
                _acc2 = _mm512_max_ps(_acc2, _b2);

                _b0_f = _b0_1;
                _b0_0 = _b0_2;

                // Store
                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; k-- > i;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1_x = _mm512_load_ps(pOutput1);
                    __m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

                    _acc1_x = _mm512_max_ps(_acc1_x, _acc1);
                    _acc2_x = _mm512_max_ps(_acc2_x, _acc2);

                    _mm512_store_ps(pOutput1, _acc1_x);
                    _mm512_store_ps(pOutput1 + 16, _acc2_x);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask1, pInput0);
                __m512 _b0_2 = _mm512_mask_load_ps(_padding, _rightInputMask2, pInput0 + 16);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_0;
                __m512 _acc1 = _b0_1;
                __m512 _acc2 = _b0_2;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                // Inner Loops
                for (int l = 1; l < kernelRemainder; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_max_ps(_acc0, _b0);
                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                _acc2 = _mm512_max_ps(_acc2, _acc1);
                _acc1 = _mm512_max_ps(_acc1, _acc0);

                for (int64_t l = kernelRemainder; l < 15; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);

                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                // Epilogue
                _acc1 = _mm512_max_ps(_acc1, _b1);
                _acc2 = _mm512_max_ps(_acc2, _b2);

                if (widthOutputRemainder < widthInputRemainder) {
                    float* pOutput1 = pOutput0;

                    for (int64_t k = KernelHeight; k-- > i;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1_x = _mm512_load_ps(pOutput1);
                        __m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

                        _acc1_x = _mm512_max_ps(_acc1_x, _acc1);
                        _acc2_x = _mm512_max_ps(_acc2_x, _acc2);

                        _mm512_store_ps(pOutput1, _acc1_x);
                        _mm512_store_ps(pOutput1 + 16, _acc2_x);
                    }

                    pOutput0 += widthStep;

                    _b0_f = _b0_1;
                    _b0_0 = _b0_2;

                    _b0_1 = _padding;
                    _b0_2 = _padding;

                    // Prologue
                    _roll = _roll_left_1;

                    _acc0 = _b0_0;
                    _acc1 = _b0_1;
                    _acc2 = _b0_2;

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                    // Inner Loops
                    for (int l = 1; l < kernelRemainder; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc0 = _mm512_max_ps(_acc0, _b0);
                        _acc1 = _mm512_max_ps(_acc1, _b1);
                        _acc2 = _mm512_max_ps(_acc2, _b2);

                        _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                        _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                        _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                    }

                    _acc2 = _mm512_max_ps(_acc2, _acc1);
                    _acc1 = _mm512_max_ps(_acc1, _acc0);

                    for (int64_t l = kernelRemainder; l < 15; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc1 = _mm512_max_ps(_acc1, _b1);
                        _acc2 = _mm512_max_ps(_acc2, _b2);

                        _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                        _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                    }

                    // Epilogue
                    _acc1 = _mm512_max_ps(_acc1, _b1);
                    _acc2 = _mm512_max_ps(_acc2, _b2);
                }

                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; k-- > i;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask1, pOutput1);
                    __m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask2, pOutput1 + 16);

                    _acc1_x = _mm512_max_ps(_acc1_x, _acc1);
                    _acc2_x = _mm512_max_ps(_acc2_x, _acc2);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask1, _acc1_x);
                    _mm512_mask_store_ps(pOutput1 + 16, _rightOutputMask2, _acc2_x);
                }
            }

            pInputRow += InputWidth;
        }

        Input += InputSize;
    }
}

void
MlasPool2DSlidingKernelMax(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                  size_t ChannelCount,
                                  const float* Input,
                                  float* Output)
{
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];

    if ((KernelHeight == 3) && (KernelWidth == 3)) {
        MlasPool2DSlidingKernelMaxK3x3S1(WorkBlock, ChannelCount, Input, Output);
    } else if ((KernelHeight == 5) && (KernelWidth == 5)) {
        MlasPool2DSlidingKernelMaxK5x5S1(WorkBlock, ChannelCount, Input, Output);
    } else if (KernelWidth <= 16) {
        MlasPool2DSlidingKernelMaxK17S1(WorkBlock, ChannelCount, Input, Output);
    } else if (KernelWidth <= 32) {
        MlasPool2DSlidingKernelMaxK32S1(WorkBlock, ChannelCount, Input, Output);
    } else
        return;
}

void
MlasPool2DSlidingKernelAvgWithPadK17S1(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                       size_t ChannelCount,
                                       const float* Input,
                                       float* Output)
{
    constexpr size_t Dimensions = 2;
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    assert(WorkBlock->PoolingKind == MLAS_POOLING_KIND::MlasAveragePoolingIncludePad);

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;
    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingTop = WorkBlock->Padding[HeightShapeIndex];
    const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
    const int64_t PaddingBottom = WorkBlock->Padding[Dimensions + HeightShapeIndex];
    const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

    assert(OutputHeight == (InputHeight - KernelHeight + PaddingTop + PaddingBottom + 1));
    assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
    MLAS_UNREFERENCED_PARAMETER(InputHeight);
    MLAS_UNREFERENCED_PARAMETER(PaddingRight);

    constexpr size_t widthStep = 16;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const __mmask16 _leftInputMask = (~0 << paddingLeftShift);

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

    const int64_t widthIterations = InputWidthBase / widthStep;
    const int64_t widthInputRemainder = InputWidthBase % widthStep;
    const int64_t widthOutputRemainder = OutputWidth % widthStep;

    const __mmask16 _rightInputMask = static_cast<__mmask16>(~(~0 << widthInputRemainder));
    const __mmask16 _rightOutputMask = static_cast<__mmask16>(~(~0 << widthOutputRemainder));

    const __m512 _weight = _mm512_set1_ps(1.0f / (float(KernelHeight) * float(KernelWidth)));

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _one = _mm512_set1_epi32(1);

    float* pOutputRow = Output;

    for (size_t c = 0; c < ChannelCount; c++) {
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
                __m512 _b0_1 = _mm512_load_ps(pInput0);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_1;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                // Inner Loop
                for (int64_t l = 2; l < KernelWidth; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_add_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                }

                // Epilogue
                _acc0 = _mm512_add_ps(_acc0, _b0);

                _b0_0 = _b0_1;

                _acc0 = _mm512_mul_ps(_acc0, _weight);

                // Store
                _mm512_store_ps(pOutput0, _acc0);

                float* pOutput1 = pOutput0;

                for (int64_t k = i; k-- > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_load_ps(pOutput1);

                    _acc1 = _mm512_add_ps(_acc0, _acc1);

                    _mm512_store_ps(pOutput1, _acc1);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_1;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                // Inner Loop
                for (int64_t l = 2; l < KernelWidth; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_add_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                }

                // Epilogue
                _acc0 = _mm512_add_ps(_acc0, _b0);

                _acc0 = _mm512_mul_ps(_acc0, _weight);

                if (widthOutputRemainder < widthInputRemainder) {
                    _mm512_store_ps(pOutput0, _acc0);

                    float* pOutput1 = pOutput0;

                    for (int64_t k = i; k-- > 0;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1 = _mm512_load_ps(pOutput1);

                        _acc1 = _mm512_add_ps(_acc0, _acc1);

                        _mm512_store_ps(pOutput1, _acc1);
                    }

                    pOutput0 += widthStep;

                    _b0_0 = _b0_1;

                    _b0_1 = _mm512_setzero_ps();

                    // Prologue
                    _roll = _roll_left_1;

                    _acc0 = _b0_1;

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                    // Inner Loop
                    for (int64_t l = 2; l < KernelWidth; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc0 = _mm512_add_ps(_acc0, _b0);

                        _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    }

                    // Epilogue
                    _acc0 = _mm512_add_ps(_acc0, _b0);

                    _acc0 = _mm512_mul_ps(_acc0, _weight);
                }

                _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);

                float* pOutput1 = pOutput0;

                for (int64_t k = i; k-- > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

                    _acc1 = _mm512_add_ps(_acc0, _acc1);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
                }
            }
            pInputRow += InputWidth;

            pOutputRow += OutputWidth;
        }

        // Outer loop
        for (size_t i = KernelHeight; i < OutputHeight; i++) {
            const float* pInput0 = pInputRow;

            float* pOutput0 = pOutputRow;

            __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_1;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                // Inner Loop
                for (int64_t l = 2; l < KernelWidth; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_add_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                }

                // Epilogue
                _acc0 = _mm512_add_ps(_acc0, _b0);

                _b0_0 = _b0_1;

                _acc0 = _mm512_mul_ps(_acc0, _weight);

                // Store
                _mm512_store_ps(pOutput0, _acc0);

                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; --k > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_load_ps(pOutput1);

                    _acc1 = _mm512_add_ps(_acc0, _acc1);

                    _mm512_store_ps(pOutput1, _acc1);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_1;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                // Inner Loop
                for (int64_t l = 2; l < KernelWidth; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_add_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                }

                // Epilogue
                _acc0 = _mm512_add_ps(_acc0, _b0);

                _acc0 = _mm512_mul_ps(_acc0, _weight);

                if (widthOutputRemainder < widthInputRemainder) {
                    _mm512_store_ps(pOutput0, _acc0);

                    float* pOutput1 = pOutput0;

                    for (int64_t k = KernelHeight; --k > 0;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1 = _mm512_load_ps(pOutput1);

                        _acc1 = _mm512_add_ps(_acc0, _acc1);

                        _mm512_store_ps(pOutput1, _acc1);
                    }

                    pOutput0 += widthStep;

                    _b0_0 = _b0_1;

                    _b0_1 = _mm512_setzero_ps();

                    // Prologue
                    _roll = _roll_left_1;

                    _acc0 = _b0_1;

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                    // Inner Loop
                    for (int64_t l = 2; l < KernelWidth; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc0 = _mm512_add_ps(_acc0, _b0);

                        _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    }

                    // Epilogue
                    _acc0 = _mm512_add_ps(_acc0, _b0);

                    _acc0 = _mm512_mul_ps(_acc0, _weight);
                }

                _mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);

                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; --k > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

                    _acc1 = _mm512_add_ps(_acc0, _acc1);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
                }
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
                __m512 _b0_1 = _mm512_load_ps(pInput0);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_1;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                // Inner Loop
                for (int64_t l = 2; l < KernelWidth; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_add_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                }

                // Epilogue
                _acc0 = _mm512_add_ps(_acc0, _b0);

                _b0_0 = _b0_1;

                _acc0 = _mm512_mul_ps(_acc0, _weight);

                // Store
                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; k-- > i;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_load_ps(pOutput1);

                    _acc1 = _mm512_add_ps(_acc0, _acc1);

                    _mm512_store_ps(pOutput1, _acc1);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_1;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                // Inner Loop
                for (int64_t l = 2; l < KernelWidth; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_add_ps(_acc0, _b0);

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                }

                // Epilogue
                _acc0 = _mm512_add_ps(_acc0, _b0);

                _acc0 = _mm512_mul_ps(_acc0, _weight);

                if (widthOutputRemainder < widthInputRemainder) {
                    float* pOutput1 = pOutput0;

                    for (int64_t k = KernelHeight; k-- > i;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1 = _mm512_load_ps(pOutput1);

                        _acc1 = _mm512_add_ps(_acc0, _acc1);

                        _mm512_store_ps(pOutput1, _acc1);
                    }

                    pOutput0 += widthStep;

                    _b0_0 = _b0_1;

                    _b0_1 = _mm512_setzero_ps();

                    // Prologue
                    _roll = _roll_left_1;

                    _acc0 = _b0_1;

                    _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

                    // Inner Loop
                    for (int64_t l = 2; l < KernelWidth; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc0 = _mm512_add_ps(_acc0, _b0);

                        _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    }

                    // Epilogue
                    _acc0 = _mm512_add_ps(_acc0, _b0);

                    _acc0 = _mm512_mul_ps(_acc0, _weight);
                }

                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; k-- > i;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

                    _acc1 = _mm512_add_ps(_acc0, _acc1);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
                }
            }

            pInputRow += InputWidth;
        }

        Input += InputSize;
    }
}

void
MlasPool2DSlidingKernelAvgWithPadK32S1(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                       size_t ChannelCount,
                                       const float* Input,
                                       float* Output)
{
    constexpr size_t Dimensions = 2;
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    assert(WorkBlock->PoolingKind == MLAS_POOLING_KIND::MlasAveragePoolingIncludePad);

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;
    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingTop = WorkBlock->Padding[HeightShapeIndex];
    const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
    const int64_t PaddingBottom = WorkBlock->Padding[Dimensions + HeightShapeIndex];
    const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

    assert(OutputHeight == (InputHeight - KernelHeight + PaddingTop + PaddingBottom + 1));
    assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
    MLAS_UNREFERENCED_PARAMETER(InputHeight);
    MLAS_UNREFERENCED_PARAMETER(PaddingRight);

    constexpr size_t widthStep = 32;

    const int64_t paddingLeftSeek = KernelWidth - PaddingLeft - 1;

    const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

    const int64_t InputWidthBase = InputWidth - paddingLeftSeek;

    const int64_t widthIterations = InputWidthBase / widthStep;
    const int64_t widthInputRemainder = InputWidthBase % widthStep;
    const int64_t widthOutputRemainder = OutputWidth % widthStep;

    const int64_t kernelRemainder = KernelWidth % 16;

    const __mmask32 _leftInputMask = static_cast<__mmask32>(~0 << paddingLeftShift);

    const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMask);
    const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMask >> 16);

    const __mmask32 _rightInputMask = static_cast<__mmask32>(~(~0 << widthInputRemainder));

    const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
    const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);

    const __mmask32 _rightOutputMask = static_cast<__mmask32>(~(~0 << widthOutputRemainder));

    const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
    const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);

    const __m512 _weight = _mm512_set1_ps(1.0f / (float(KernelHeight) * float(KernelWidth)));

    const __m512i _roll_left_1 =
        _mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
    const __m512i _one = _mm512_set1_epi32(1);

    float* pOutputRow = Output;

    for (size_t c = 0; c < ChannelCount; c++) {
        const float* pInputRow = Input;

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

            __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift);
            __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);
                __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_0;
                __m512 _acc1 = _b0_1;
                __m512 _acc2 = _b0_2;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                // Inner Loops
                for (int64_t l = 1; l < kernelRemainder; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_add_ps(_acc0, _b0);
                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                _acc2 = _mm512_add_ps(_acc2, _acc1);
                _acc1 = _mm512_add_ps(_acc1, _acc0);

                for (int64_t l = kernelRemainder; l < 15; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                // Epilogue
                _acc1 = _mm512_add_ps(_acc1, _b1);
                _acc2 = _mm512_add_ps(_acc2, _b2);

                _acc1 = _mm512_mul_ps(_acc1, _weight);
                _acc2 = _mm512_mul_ps(_acc2, _weight);

                _b0_f = _b0_1;
                _b0_0 = _b0_2;

                // Store
                _mm512_store_ps(pOutput0, _acc1);
                _mm512_store_ps(pOutput0 + 16, _acc2);

                float* pOutput1 = pOutput0;

                for (int64_t k = i; k-- > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1_x = _mm512_load_ps(pOutput1);
                    __m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

                    _acc1_x = _mm512_add_ps(_acc1_x, _acc1);
                    _acc2_x = _mm512_add_ps(_acc2_x, _acc2);

                    _mm512_store_ps(pOutput1, _acc1_x);
                    _mm512_store_ps(pOutput1 + 16, _acc2_x);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0);
                __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_0;
                __m512 _acc1 = _b0_1;
                __m512 _acc2 = _b0_2;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                // Inner Loops
                for (int l = 1; l < kernelRemainder; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_add_ps(_acc0, _b0);
                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                _acc2 = _mm512_add_ps(_acc2, _acc1);
                _acc1 = _mm512_add_ps(_acc1, _acc0);

                for (int64_t l = kernelRemainder; l < 15; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                // Epilogue
                _acc1 = _mm512_add_ps(_acc1, _b1);
                _acc2 = _mm512_add_ps(_acc2, _b2);

                _acc1 = _mm512_mul_ps(_acc1, _weight);
                _acc2 = _mm512_mul_ps(_acc2, _weight);

                if (widthOutputRemainder < widthInputRemainder) {
                    // Store
                    _mm512_store_ps(pOutput0, _acc1);
                    _mm512_store_ps(pOutput0 + 16, _acc2);

                    float* pOutput1 = pOutput0;

                    for (int64_t k = i; k-- > 0;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1_x = _mm512_load_ps(pOutput1);
                        __m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

                        _acc1_x = _mm512_add_ps(_acc1_x, _acc1);
                        _acc2_x = _mm512_add_ps(_acc2_x, _acc2);

                        _mm512_store_ps(pOutput1, _acc1_x);
                        _mm512_store_ps(pOutput1 + 16, _acc2_x);
                    }

                    pOutput0 += widthStep;

                    _b0_f = _b0_1;
                    _b0_0 = _b0_2;

                    _b0_1 = _b0_2 = _mm512_setzero_ps();

                    // Prologue
                    _roll = _roll_left_1;

                    _acc0 = _b0_0;
                    _acc1 = _b0_1;
                    _acc2 = _b0_2;

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                    // Inner Loops
                    for (int l = 1; l < kernelRemainder; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc0 = _mm512_add_ps(_acc0, _b0);
                        _acc1 = _mm512_add_ps(_acc1, _b1);
                        _acc2 = _mm512_add_ps(_acc2, _b2);

                        _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                        _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                        _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                    }

                    _acc2 = _mm512_add_ps(_acc2, _acc1);
                    _acc1 = _mm512_add_ps(_acc1, _acc0);

                    for (int64_t l = kernelRemainder; l < 15; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc1 = _mm512_add_ps(_acc1, _b1);
                        _acc2 = _mm512_add_ps(_acc2, _b2);

                        _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                        _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                    }

                    // Epilogue
                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _acc1 = _mm512_mul_ps(_acc1, _weight);
                    _acc2 = _mm512_mul_ps(_acc2, _weight);
                }

                _mm512_mask_store_ps(pOutput0, _rightOutputMask1, _acc1);
                _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _acc2);

                float* pOutput1 = pOutput0;

                for (int64_t k = i; k-- > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask1, pOutput1);
                    __m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask2, pOutput1 + 16);

                    _acc1_x = _mm512_add_ps(_acc1_x, _acc1);
                    _acc2_x = _mm512_add_ps(_acc2_x, _acc2);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask1, _acc1_x);
                    _mm512_mask_store_ps(pOutput1 + 16, _rightOutputMask2, _acc2_x);
                }
            }

            pInputRow += InputWidth;

            pOutputRow += OutputWidth;
        }

        // Outer loop
        for (size_t i = KernelHeight; i < OutputHeight; i++) {
            const float* pInput0 = pInputRow;

            float* pOutput0 = pOutputRow;

            __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift);
            __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);
                __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_0;
                __m512 _acc1 = _b0_1;
                __m512 _acc2 = _b0_2;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                // Inner Loops
                for (int64_t l = 1; l < kernelRemainder; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_add_ps(_acc0, _b0);
                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                _acc2 = _mm512_add_ps(_acc2, _acc1);
                _acc1 = _mm512_add_ps(_acc1, _acc0);

                for (int64_t l = kernelRemainder; l < 15; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                // Epilogue
                _acc1 = _mm512_add_ps(_acc1, _b1);
                _acc2 = _mm512_add_ps(_acc2, _b2);

                _acc1 = _mm512_mul_ps(_acc1, _weight);
                _acc2 = _mm512_mul_ps(_acc2, _weight);

                _b0_f = _b0_1;
                _b0_0 = _b0_2;

                // Store
                _mm512_store_ps(pOutput0, _acc1);
                _mm512_store_ps(pOutput0 + 16, _acc2);

                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; --k > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1_x = _mm512_load_ps(pOutput1);
                    __m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

                    _acc1_x = _mm512_add_ps(_acc1_x, _acc1);
                    _acc2_x = _mm512_add_ps(_acc2_x, _acc2);

                    _mm512_store_ps(pOutput1, _acc1_x);
                    _mm512_store_ps(pOutput1 + 16, _acc2_x);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0);
                __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_0;
                __m512 _acc1 = _b0_1;
                __m512 _acc2 = _b0_2;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                // Inner Loops
                for (int l = 1; l < kernelRemainder; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_add_ps(_acc0, _b0);
                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                _acc2 = _mm512_add_ps(_acc2, _acc1);
                _acc1 = _mm512_add_ps(_acc1, _acc0);

                for (int64_t l = kernelRemainder; l < 15; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                // Epilogue
                _acc1 = _mm512_add_ps(_acc1, _b1);
                _acc2 = _mm512_add_ps(_acc2, _b2);

                _acc1 = _mm512_mul_ps(_acc1, _weight);
                _acc2 = _mm512_mul_ps(_acc2, _weight);

                if (widthOutputRemainder < widthInputRemainder) {
                    _mm512_store_ps(pOutput0, _acc0);

                    float* pOutput1 = pOutput0;

                    for (int64_t k = KernelHeight; --k > 0;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1_x = _mm512_load_ps(pOutput1);
                        __m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

                        _acc1_x = _mm512_add_ps(_acc1_x, _acc1);
                        _acc2_x = _mm512_add_ps(_acc2_x, _acc2);

                        _mm512_store_ps(pOutput1, _acc1_x);
                        _mm512_store_ps(pOutput1 + 16, _acc2_x);
                    }

                    pOutput0 += widthStep;

                    _b0_f = _b0_1;
                    _b0_0 = _b0_2;

                    _b0_1 = _b0_2 = _mm512_setzero_ps();

                    // Prologue
                    _roll = _roll_left_1;

                    _acc0 = _b0_0;
                    _acc1 = _b0_1;
                    _acc2 = _b0_2;

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                    // Inner Loops
                    for (int l = 1; l < kernelRemainder; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc0 = _mm512_add_ps(_acc0, _b0);
                        _acc1 = _mm512_add_ps(_acc1, _b1);
                        _acc2 = _mm512_add_ps(_acc2, _b2);

                        _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                        _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                        _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                    }

                    _acc2 = _mm512_add_ps(_acc2, _acc1);
                    _acc1 = _mm512_add_ps(_acc1, _acc0);

                    for (int64_t l = kernelRemainder; l < 15; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc1 = _mm512_add_ps(_acc1, _b1);
                        _acc2 = _mm512_add_ps(_acc2, _b2);

                        _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                        _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                    }

                    // Epilogue
                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _acc1 = _mm512_mul_ps(_acc1, _weight);
                    _acc2 = _mm512_mul_ps(_acc2, _weight);
                }

                _mm512_mask_store_ps(pOutput0, _rightOutputMask1, _acc1);
                _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask2, _acc2);

                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; --k > 0;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask1, pOutput1);
                    __m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask2, pOutput1 + 16);

                    _acc1_x = _mm512_add_ps(_acc1_x, _acc1);
                    _acc2_x = _mm512_add_ps(_acc2_x, _acc2);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask1, _acc1_x);
                    _mm512_mask_store_ps(pOutput1 + 16, _rightOutputMask2, _acc2_x);
                }
            }

            pInputRow += InputWidth;

            pOutputRow += OutputWidth;
        }

        // Outer Loop Epilogue
        for (int64_t i = 1; i < KernelHeight - PaddingBottom; i++) {
            const float* pInput0 = pInputRow;

            float* pOutput0 = pOutputRow;

            __m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift);
            __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

            pInput0 += paddingLeftSeek;

            for (int64_t j = 0; j < widthIterations; j++) {
                __m512 _b0_1 = _mm512_load_ps(pInput0);
                __m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_0;
                __m512 _acc1 = _b0_1;
                __m512 _acc2 = _b0_2;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                // Inner Loops
                for (int64_t l = 1; l < kernelRemainder; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_add_ps(_acc0, _b0);
                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                _acc2 = _mm512_add_ps(_acc2, _acc1);
                _acc1 = _mm512_add_ps(_acc1, _acc0);

                for (int64_t l = kernelRemainder; l < 15; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                // Epilogue
                _acc1 = _mm512_add_ps(_acc1, _b1);
                _acc2 = _mm512_add_ps(_acc2, _b2);

                _acc1 = _mm512_mul_ps(_acc1, _weight);
                _acc2 = _mm512_mul_ps(_acc2, _weight);

                _b0_f = _b0_1;
                _b0_0 = _b0_2;

                // Store
                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; k-- > i;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1_x = _mm512_load_ps(pOutput1);
                    __m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

                    _acc1_x = _mm512_add_ps(_acc1_x, _acc1);
                    _acc2_x = _mm512_add_ps(_acc2_x, _acc2);

                    _mm512_store_ps(pOutput1, _acc1_x);
                    _mm512_store_ps(pOutput1 + 16, _acc2_x);
                }

                pInput0 += widthStep;
                pOutput0 += widthStep;
            }

            // Right Edge
            {
                __m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0);
                __m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);

                // Prologue
                __m512i _roll = _roll_left_1;

                __m512 _acc0 = _b0_0;
                __m512 _acc1 = _b0_1;
                __m512 _acc2 = _b0_2;

                __m512 _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                __m512 _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                __m512 _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                // Inner Loops
                for (int l = 1; l < kernelRemainder; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc0 = _mm512_add_ps(_acc0, _b0);
                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                _acc2 = _mm512_add_ps(_acc2, _acc1);
                _acc1 = _mm512_add_ps(_acc1, _acc0);

                for (int64_t l = kernelRemainder; l < 15; l++) {
                    _roll = _mm512_sub_epi32(_roll, _one);

                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                }

                // Epilogue
                _acc1 = _mm512_add_ps(_acc1, _b1);
                _acc2 = _mm512_add_ps(_acc2, _b2);

                _acc1 = _mm512_mul_ps(_acc1, _weight);
                _acc2 = _mm512_mul_ps(_acc2, _weight);

                if (widthOutputRemainder < widthInputRemainder) {
                    float* pOutput1 = pOutput0;

                    for (int64_t k = KernelHeight; k-- > i;) {
                        pOutput1 -= OutputWidth;

                        __m512 _acc1_x = _mm512_load_ps(pOutput1);
                        __m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

                        _acc1_x = _mm512_add_ps(_acc1_x, _acc1);
                        _acc2_x = _mm512_add_ps(_acc2_x, _acc2);

                        _mm512_store_ps(pOutput1, _acc1_x);
                        _mm512_store_ps(pOutput1 + 16, _acc2_x);
                    }

                    pOutput0 += widthStep;

                    _b0_f = _b0_1;
                    _b0_0 = _b0_2;

                    _b0_1 = _b0_2 = _mm512_setzero_ps();

                    // Prologue
                    _roll = _roll_left_1;

                    _acc0 = _b0_0;
                    _acc1 = _b0_1;
                    _acc2 = _b0_2;

                    _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                    _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                    _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);

                    // Inner Loops
                    for (int l = 1; l < kernelRemainder; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc0 = _mm512_add_ps(_acc0, _b0);
                        _acc1 = _mm512_add_ps(_acc1, _b1);
                        _acc2 = _mm512_add_ps(_acc2, _b2);

                        _b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
                        _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                        _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                    }

                    _acc2 = _mm512_add_ps(_acc2, _acc1);
                    _acc1 = _mm512_add_ps(_acc1, _acc0);

                    for (int64_t l = kernelRemainder; l < 15; l++) {
                        _roll = _mm512_sub_epi32(_roll, _one);

                        _acc1 = _mm512_add_ps(_acc1, _b1);
                        _acc2 = _mm512_add_ps(_acc2, _b2);

                        _b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
                        _b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
                    }

                    // Epilogue
                    _acc1 = _mm512_add_ps(_acc1, _b1);
                    _acc2 = _mm512_add_ps(_acc2, _b2);

                    _acc1 = _mm512_mul_ps(_acc1, _weight);
                    _acc2 = _mm512_mul_ps(_acc2, _weight);
                }

                float* pOutput1 = pOutput0;

                for (int64_t k = KernelHeight; k-- > i;) {
                    pOutput1 -= OutputWidth;

                    __m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask1, pOutput1);
                    __m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask2, pOutput1 + 16);

                    _acc1_x = _mm512_add_ps(_acc1_x, _acc1);
                    _acc2_x = _mm512_add_ps(_acc2_x, _acc2);

                    _mm512_mask_store_ps(pOutput1, _rightOutputMask1, _acc1_x);
                    _mm512_mask_store_ps(pOutput1 + 16, _rightOutputMask2, _acc2_x);
                }
            }

            pInputRow += InputWidth;
        }

        Input += InputSize;
    }
}

void
MlasPool2DSlidingKernelAvgWithPad(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                  size_t ChannelCount,
                                  const float* Input,
                                  float* Output)
{
    constexpr size_t WidthShapeIndex = 1;

    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];

    if (KernelWidth <= 16) {
        MlasPool2DSlidingKernelAvgWithPadK17S1(WorkBlock, ChannelCount, Input, Output);
    } else if (KernelWidth <= 32) {
        MlasPool2DSlidingKernelAvgWithPadK32S1(WorkBlock, ChannelCount, Input, Output);
    } else
        return;
}

void
MlasPool2DSlidingKernelAvgNoPad(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                size_t ChannelCount,
                                const float* Input,
                                float* Output)
{
    constexpr size_t Dimensions = 2;
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingTop = WorkBlock->Padding[HeightShapeIndex];
    const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
    const int64_t PaddingBottom = WorkBlock->Padding[Dimensions + HeightShapeIndex];
    const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

    assert(WorkBlock->PoolingKind == MLAS_POOLING_KIND::MlasAveragePoolingExcludePad);

    MLAS_POOL_WORK_BLOCK NewBlock{
        MlasAveragePoolingIncludePad,
        {WorkBlock->InputShape[0], WorkBlock->InputShape[1], WorkBlock->InputShape[2]},
        WorkBlock->InputSize,
        {WorkBlock->OutputShape[0], WorkBlock->OutputShape[1], WorkBlock->OutputShape[2]},
        {WorkBlock->KernelShape[0], WorkBlock->KernelShape[1], WorkBlock->KernelShape[2]},
        {WorkBlock->Padding[0], WorkBlock->Padding[1], WorkBlock->Padding[2], WorkBlock->Padding[3],
         WorkBlock->Padding[4], WorkBlock->Padding[5]},
        {WorkBlock->StrideShape[0], WorkBlock->StrideShape[1], WorkBlock->StrideShape[2]}};

    if (KernelWidth <= 16) {
        MlasPool2DSlidingKernelAvgWithPadK17S1(&NewBlock, ChannelCount, Input, Output);
    } else if (KernelWidth <= 32) {
        MlasPool2DSlidingKernelAvgWithPadK32S1(&NewBlock, ChannelCount, Input, Output);
    } else
        return;

    float shape = float(KernelHeight) * float(KernelWidth);

    float* pOuputRow = Output;

    for (int64_t i = PaddingTop; i > 0; i--) {
        float* pBegin = pOuputRow;

        for (int64_t j = PaddingLeft; j > 0; j--) {
            *pBegin++ *= shape / ((KernelHeight - i) * (KernelWidth - j));
        }

        float* pEnd = pOuputRow + OutputWidth;

        for (int64_t j = PaddingRight; j > 0; j--) {
            *--pEnd *= shape / ((KernelHeight - i) * (KernelWidth - j));
        }

        while (pBegin < pEnd) {
            *pBegin++ *= shape / ((KernelHeight - i) * KernelWidth);
        }

        pOuputRow += OutputWidth;
    }

    for (int64_t i = PaddingTop; i < int64_t(OutputHeight - PaddingBottom); i++) {
        float* pBegin = pOuputRow;

        for (int64_t j = PaddingLeft; j > 0; j--) {
            *pBegin++ *= shape / (shape - float(j) * KernelHeight);
        }

        float* pEnd = pOuputRow + OutputWidth;

        for (int64_t j = PaddingRight; j > 0; j--) {
            *--pEnd *= shape / (shape - float(j) * KernelHeight);
        }

        pOuputRow += OutputWidth;
    }

    for (int64_t i = 1; i <= PaddingBottom; i++) {
        float* pBegin = pOuputRow;

        for (int64_t j = PaddingLeft; j > 0; j--) {
            *pBegin++ *= shape / ((KernelHeight - i) * (KernelWidth - j));
        }

        float* pEnd = pOuputRow + OutputWidth;

        for (int64_t j = PaddingRight; j > 0; j--) {
            *--pEnd *= shape / ((KernelHeight - i) * (KernelWidth - j));
        }

        while (pBegin < pEnd) {
            *pBegin++ *= shape / ((KernelHeight - i) * KernelWidth);
        }

        pOuputRow += OutputWidth;
    }
}