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

	const int64_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
	const int64_t InputSize = WorkBlock->InputSize;
	const int64_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
	const int64_t OutputSize = OutputWidth;

	const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
	const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
	const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

	assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
	MLAS_UNREFERENCED_PARAMETER(PaddingRight);

	constexpr size_t widthStep = 16;

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

		Output += OutputSize;
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

	const int64_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
	const int64_t InputSize = WorkBlock->InputSize;
	const int64_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
	const int64_t OutputSize = OutputWidth;

	const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
	const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
	const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

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

		Output += OutputSize;
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
	}
	else if (KernelWidth <= 32) {
		MlasPool1DSlidingKernelMaxK32S1(WorkBlock, ChannelCount, Input, Output);
	}
	else
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

	const int64_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
	const int64_t InputSize = WorkBlock->InputSize;
	const int64_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
	const int64_t OutputSize = OutputWidth;

	const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
	const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
	const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

	assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
	MLAS_UNREFERENCED_PARAMETER(PaddingRight);

	constexpr size_t widthStep = 16;

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

		Output += OutputSize;
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

	const int64_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
	const int64_t InputSize = WorkBlock->InputSize;
	const int64_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
	const int64_t OutputSize = OutputWidth;

	const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
	const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
	const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

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

		Output += OutputSize;
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
	}
	else if (KernelWidth <= 32) {
		MlasPool1DSlidingKernelAvgWithPadK32S1(WorkBlock, ChannelCount, Input, Output);
	}
	else
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
		{WorkBlock->StrideShape[0], WorkBlock->StrideShape[1], WorkBlock->StrideShape[2]} };

	if (KernelWidth <= 16) {
		MlasPool1DSlidingKernelAvgWithPadK17S1(&NewBlock, ChannelCount, Input, Output);
	}
	else if (KernelWidth <= 32) {
		MlasPool1DSlidingKernelAvgWithPadK32S1(&NewBlock, ChannelCount, Input, Output);
	}
	else
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

	const int64_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
	const int64_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
	const int64_t InputSize = WorkBlock->InputSize;
	const int64_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
	const int64_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
	const int64_t OutputSize = OutputHeight * OutputWidth;

	const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
	const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
	const int64_t PaddingTop = WorkBlock->Padding[HeightShapeIndex];
	const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
	const int64_t PaddingBottom = WorkBlock->Padding[Dimensions + HeightShapeIndex];
	const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

	assert(OutputHeight == (InputHeight - KernelHeight + PaddingTop + PaddingBottom + 1));
	assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
	MLAS_UNREFERENCED_PARAMETER(PaddingBottom);
	MLAS_UNREFERENCED_PARAMETER(PaddingRight);

	constexpr size_t widthStep = 16;

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
	const __m512i _one = _mm512_set1_epi32(1);

	const __m512 _padding = _mm512_set1_ps(std::numeric_limits<float>::lowest());

	for (size_t c = 0; c < ChannelCount; c++) {
		float* pOutputRow = Output + PaddingTop * OutputWidth;

		const float* pInputRow = Input;

		// Outer loop
		for (int64_t i = 0; i < InputHeight; i++) {
			const float* pInput0 = pInputRow;

			float* pOutput0 = pOutputRow;

			__m512 _b0_0 =
				_mm512_mask_load_ps(_padding, _leftInputMask, pInput0 - paddingLeftShift);

			pInput0 += paddingLeftSeek;

			for (int64_t j = 0; j < widthIterations; j++) {
				__m512 _b0_1 = _mm512_load_ps(pInput0);

				// Inner Loop
				__m512i _roll = _roll_left_1;

				__m512 _acc0 = _b0_1;

				for (int64_t l = 1; l < KernelWidth; l++) {
					__m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

					_acc0 = _mm512_max_ps(_acc0, _b0);

					_roll = _mm512_sub_epi32(_roll, _one);
				}

				// Store
				float* pOutput1 = pOutput0;

				for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
					if ((0 <= gk) && (gk < OutputHeight)) {
						__m512 _acc1 = _mm512_load_ps(pOutput1);

						_acc1 = _mm512_max_ps(_acc0, _acc1);

						_mm512_store_ps(pOutput1, _acc1);
					}

					pOutput1 -= OutputWidth;
				}

				_b0_0 = _b0_1;

				pInput0 += widthStep;
				pOutput0 += widthStep;
			}

			// Right Edge
			{
				__m512 _b0_1 = _mm512_mask_load_ps(_padding, _rightInputMask, pInput0);

				// Inner Loop
				__m512i _roll = _roll_left_1;

				__m512 _acc0 = _b0_1;

				for (int64_t l = 1; l < KernelWidth; l++) {
					__m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

					_acc0 = _mm512_max_ps(_acc0, _b0);

					_roll = _mm512_sub_epi32(_roll, _one);
				}

				if (widthOutputRemainder < widthInputRemainder) {
					float* pOutput1 = pOutput0;

					for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
						if ((0 <= gk) && (gk < OutputHeight)) {
							__m512 _acc1 = _mm512_load_ps(pOutput1);

							_acc1 = _mm512_max_ps(_acc0, _acc1);

							_mm512_store_ps(pOutput1, _acc1);
						}

						pOutput1 -= OutputWidth;
					}

					pOutput0 += widthStep;

					_b0_0 = _b0_1;

					_b0_1 = _padding;

					// Inner Loop
					_roll = _roll_left_1;

					_acc0 = _b0_1;

					for (int64_t l = 1; l < KernelWidth; l++) {
						__m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

						_acc0 = _mm512_max_ps(_acc0, _b0);

						_roll = _mm512_sub_epi32(_roll, _one);
					}
				}

				float* pOutput1 = pOutput0;

				for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
					if ((0 <= gk) && (gk < OutputHeight)) {
						__m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

						_acc1 = _mm512_max_ps(_acc0, _acc1);

						_mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
					}

					pOutput1 -= OutputWidth;
				}
			}

			pInputRow += InputWidth;

			pOutputRow += OutputWidth;
		}

		Input += InputSize;

		Output += OutputSize;
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

	const int64_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
	const int64_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
	const int64_t InputSize = WorkBlock->InputSize;
	const int64_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
	const int64_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
	const int64_t OutputSize = OutputHeight * OutputWidth;

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

	const __m512 _padding = _mm512_set1_ps(std::numeric_limits<float>::lowest());

	for (size_t c = 0; c < ChannelCount; c++) {
		float* pOutputRow = Output + PaddingTop * OutputWidth;

		const float* pInputRow = Input;

		if (heightIterations > 0) {
			// Outer Loop Prologue
			for (int64_t i = 0; i < heightStep; i++) {
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

					for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
						if ((0 <= gk) && (gk < OutputHeight)) {
							__m512 _acc1 = _mm512_load_ps(pOutput1);

							_acc1 = _mm512_max_ps(_acc0, _acc1);

							_mm512_store_ps(pOutput1, _acc1);
						}

						pOutput1 -= OutputWidth;
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

						for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
							if ((0 <= gk) && (gk < OutputHeight)) {
								__m512 _acc1 = _mm512_load_ps(pOutput1);

								_acc1 = _mm512_max_ps(_acc0, _acc1);

								_mm512_store_ps(pOutput1, _acc1);
							}

							pOutput1 -= OutputWidth;
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

					for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
						if ((0 <= gk) && (gk < OutputHeight)) {
							__m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

							_acc1 = _mm512_max_ps(_acc0, _acc1);

							_mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
						}

						pOutput1 -= OutputWidth;
					}
				}

				pInputRow += InputWidth;

				pOutputRow += OutputWidth;
			}

			// Main Outer Loop
			for (int64_t i = 1; i < heightIterations; i++) {
				const float* pInput0 = pInputRow;
				const float* pInput1 = pInput0 + InputWidth;
				const float* pInput2 = pInput1 + InputWidth;
				const float* pInput3 = pInput2 + InputWidth;

				float* pOutput0 = pOutputRow;
				float* pOutput1 = pOutput0 + OutputWidth;
				float* pOutput2 = pOutput1 + OutputWidth;
				float* pOutput3 = pOutput2 + OutputWidth;
				float* pOutputf = pOutput0 - OutputWidth;
				float* pOutpute = pOutputf - OutputWidth;

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

					__m512 _acc0 = _mm512_max_ps(_b0_1, _b0);
					__m512 _acc1 = _mm512_max_ps(_b1_1, _b1);
					__m512 _acc2 = _mm512_max_ps(_b2_1, _b2);
					__m512 _acc3 = _mm512_max_ps(_b3_1, _b3);

					_b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
					_b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
					_b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
					_b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

					_acc0 = _mm512_max_ps(_acc0, _b0);
					_acc1 = _mm512_max_ps(_acc1, _b1);
					_acc2 = _mm512_max_ps(_acc2, _b2);
					_acc3 = _mm512_max_ps(_acc3, _b3);

					_b0_0 = _b0_1;
					_b1_0 = _b1_1;
					_b2_0 = _b2_1;
					_b3_0 = _b3_1;

					// Prefix Sums
					__m512 _acce = _mm512_max_ps(_acc0, _mm512_load_ps(pOutpute));

					_acc0 = _mm512_max_ps(_acc0, _acc1);
					_acc1 = _mm512_max_ps(_acc1, _acc2);

					__m512 _accf = _mm512_max_ps(_acc0, _mm512_load_ps(pOutputf));

					_acc0 = _mm512_max_ps(_acc0, _acc2);
					_acc1 = _mm512_max_ps(_acc1, _acc3);
					_acc2 = _mm512_max_ps(_acc2, _acc3);

					_acc0 = _mm512_max_ps(_acc0, _mm512_load_ps(pOutput0));
					_acc1 = _mm512_max_ps(_acc1, _mm512_load_ps(pOutput1));
					_acc2 = _mm512_max_ps(_acc2, _mm512_load_ps(pOutput2));
					_acc3 = _mm512_max_ps(_acc3, _mm512_load_ps(pOutput3));

					// Store
					_mm512_store_ps(pOutpute, _acce);
					_mm512_store_ps(pOutputf, _accf);

					_mm512_store_ps(pOutput0, _acc0);
					_mm512_store_ps(pOutput1, _acc1);
					_mm512_store_ps(pOutput2, _acc2);
					_mm512_store_ps(pOutput3, _acc3);

					pInput0 += widthStep;
					pInput1 += widthStep;
					pInput2 += widthStep;
					pInput3 += widthStep;

					pOutpute += widthStep;
					pOutputf += widthStep;
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

					__m512 _acc0 = _mm512_max_ps(_b0_1, _b0);
					__m512 _acc1 = _mm512_max_ps(_b1_1, _b1);
					__m512 _acc2 = _mm512_max_ps(_b2_1, _b2);
					__m512 _acc3 = _mm512_max_ps(_b3_1, _b3);

					_b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
					_b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
					_b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
					_b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

					_acc0 = _mm512_max_ps(_acc0, _b0);
					_acc1 = _mm512_max_ps(_acc1, _b1);
					_acc2 = _mm512_max_ps(_acc2, _b2);
					_acc3 = _mm512_max_ps(_acc3, _b3);

					if (widthOutputRemainder < widthInputRemainder) {
						__m512 _acce = _mm512_max_ps(_acc0, _mm512_load_ps(pOutpute));

						_acc0 = _mm512_max_ps(_acc0, _acc1);
						_acc1 = _mm512_max_ps(_acc1, _acc2);

						__m512 _accf = _mm512_max_ps(_acc0, _mm512_load_ps(pOutputf));

						_acc0 = _mm512_max_ps(_acc0, _acc2);
						_acc1 = _mm512_max_ps(_acc1, _acc3);
						_acc2 = _mm512_max_ps(_acc2, _acc3);

						_acc0 = _mm512_max_ps(_acc0, _mm512_load_ps(pOutput0));
						_acc1 = _mm512_max_ps(_acc1, _mm512_load_ps(pOutput1));
						_acc2 = _mm512_max_ps(_acc2, _mm512_load_ps(pOutput2));
						_acc3 = _mm512_max_ps(_acc3, _mm512_load_ps(pOutput3));

						_mm512_store_ps(pOutpute, _acce);
						_mm512_store_ps(pOutputf, _accf);
						_mm512_store_ps(pOutput0, _acc0);
						_mm512_store_ps(pOutput1, _acc1);
						_mm512_store_ps(pOutput2, _acc2);
						_mm512_store_ps(pOutput3, _acc3);

						pOutpute += widthStep;
						pOutputf += widthStep;
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

						_acc0 = _mm512_max_ps(_b0_1, _b0);
						_acc1 = _mm512_max_ps(_b1_1, _b1);
						_acc2 = _mm512_max_ps(_b2_1, _b2);
						_acc3 = _mm512_max_ps(_b3_1, _b3);

						_b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
						_b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
						_b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
						_b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

						_acc0 = _mm512_max_ps(_acc0, _b0);
						_acc1 = _mm512_max_ps(_acc1, _b1);
						_acc2 = _mm512_max_ps(_acc2, _b2);
						_acc3 = _mm512_max_ps(_acc3, _b3);
					}

					// Prefix Sums
					__m512 _acce = _mm512_max_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutpute));

					_acc0 = _mm512_max_ps(_acc0, _acc1);
					_acc1 = _mm512_max_ps(_acc1, _acc2);

					__m512 _accf = _mm512_max_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutputf));

					_acc0 = _mm512_max_ps(_acc0, _acc2);
					_acc1 = _mm512_max_ps(_acc1, _acc3);
					_acc2 = _mm512_max_ps(_acc2, _acc3);

					_acc0 = _mm512_max_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutput0));
					_acc1 = _mm512_max_ps(_acc1, _mm512_maskz_load_ps(_rightOutputMask, pOutput1));
					_acc2 = _mm512_max_ps(_acc2, _mm512_maskz_load_ps(_rightOutputMask, pOutput2));
					_acc3 = _mm512_max_ps(_acc3, _mm512_maskz_load_ps(_rightOutputMask, pOutput3));

					// Store
					_mm512_mask_store_ps(pOutpute, _rightOutputMask, _acce);
					_mm512_mask_store_ps(pOutputf, _rightOutputMask, _accf);

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

				for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
					if ((0 <= gk) && (gk < OutputHeight)) {
						__m512 _acc1 = _mm512_load_ps(pOutput1);

						_acc1 = _mm512_max_ps(_acc0, _acc1);

						_mm512_store_ps(pOutput1, _acc1);
					}

					pOutput1 -= OutputWidth;
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

					for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
						if ((0 <= gk) && (gk < OutputHeight)) {
							__m512 _acc1 = _mm512_load_ps(pOutput1);

							_acc1 = _mm512_max_ps(_acc0, _acc1);

							_mm512_store_ps(pOutput1, _acc1);
						}

						pOutput1 -= OutputWidth;
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

				for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
					if ((0 <= gk) && (gk < OutputHeight)) {
						__m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

						_acc1 = _mm512_max_ps(_acc0, _acc1);

						_mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
					}

					pOutput1 -= OutputWidth;
				}
			}

			pInputRow += InputWidth;

			pOutputRow += OutputWidth;
		}

		Input += InputSize;

		Output += OutputSize;
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

	const int64_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
	const int64_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
	const int64_t InputSize = WorkBlock->InputSize;
	const int64_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
	const int64_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
	const int64_t OutputSize = OutputHeight * OutputWidth;

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

	const __m512 _padding = _mm512_set1_ps(std::numeric_limits<float>::lowest());

	for (size_t c = 0; c < ChannelCount; c++) {
		float* pOutputRow = Output + PaddingTop * OutputWidth;

		const float* pInputRow = Input;

		if (heightIterations > 0) {
			// Outer Loop Prologue
			for (int64_t i = 0; i < heightStep; i++) {
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

					for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
						if ((0 <= gk) && (gk < OutputHeight)) {
							__m512 _acc1 = _mm512_load_ps(pOutput1);

							_acc1 = _mm512_max_ps(_acc0, _acc1);

							_mm512_store_ps(pOutput1, _acc1);
						}

						pOutput1 -= OutputWidth;
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

						for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
							if ((0 <= gk) && (gk < OutputHeight)) {
								__m512 _acc1 = _mm512_load_ps(pOutput1);

								_acc1 = _mm512_max_ps(_acc0, _acc1);

								_mm512_store_ps(pOutput1, _acc1);
							}

							pOutput1 -= OutputWidth;
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

					for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
						if ((0 <= gk) && (gk < OutputHeight)) {
							__m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

							_acc1 = _mm512_max_ps(_acc0, _acc1);

							_mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
						}

						pOutput1 -= OutputWidth;
					}
				}

				pInputRow += InputWidth;

				pOutputRow += OutputWidth;
			}

			// Main Outer Loop
			for (int64_t i = 1; i < heightIterations; i++) {
				const float* pInput0 = pInputRow;
				const float* pInput1 = pInput0 + InputWidth;
				const float* pInput2 = pInput1 + InputWidth;
				const float* pInput3 = pInput2 + InputWidth;

				float* pOutput0 = pOutputRow;
				float* pOutput1 = pOutput0 + OutputWidth;
				float* pOutput2 = pOutput1 + OutputWidth;
				float* pOutput3 = pOutput2 + OutputWidth;
				float* pOutputf = pOutput0 - OutputWidth;
				float* pOutpute = pOutputf - OutputWidth;
				float* pOutputd = pOutpute - OutputWidth;
				float* pOutputc = pOutputd - OutputWidth;

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

					__m512 _acc0 = _mm512_max_ps(_b0_1, _b0);
					__m512 _acc1 = _mm512_max_ps(_b1_1, _b1);
					__m512 _acc2 = _mm512_max_ps(_b2_1, _b2);
					__m512 _acc3 = _mm512_max_ps(_b3_1, _b3);

					_b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
					_b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
					_b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
					_b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

					_acc0 = _mm512_max_ps(_acc0, _b0);
					_acc1 = _mm512_max_ps(_acc1, _b1);
					_acc2 = _mm512_max_ps(_acc2, _b2);
					_acc3 = _mm512_max_ps(_acc3, _b3);

					_b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
					_b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_3, _b1_1);
					_b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_3, _b2_1);
					_b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_3, _b3_1);

					_acc0 = _mm512_max_ps(_acc0, _b0);
					_acc1 = _mm512_max_ps(_acc1, _b1);
					_acc2 = _mm512_max_ps(_acc2, _b2);
					_acc3 = _mm512_max_ps(_acc3, _b3);

					_b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);
					_b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_4, _b1_1);
					_b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_4, _b2_1);
					_b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_4, _b3_1);

					_acc0 = _mm512_max_ps(_acc0, _b0);
					_acc1 = _mm512_max_ps(_acc1, _b1);
					_acc2 = _mm512_max_ps(_acc2, _b2);
					_acc3 = _mm512_max_ps(_acc3, _b3);

					_b0_0 = _b0_1;
					_b1_0 = _b1_1;
					_b2_0 = _b2_1;
					_b3_0 = _b3_1;

					// Prefix Sums
					__m512 _accc = _mm512_max_ps(_acc0, _mm512_load_ps(pOutputc));
					__m512 _acce = _mm512_max_ps(_acc2, _mm512_load_ps(pOutpute));

					_acc0 = _mm512_max_ps(_acc0, _acc1);
					_acc2 = _mm512_max_ps(_acc2, _acc3);

					__m512 _accd = _mm512_max_ps(_acc0, _mm512_load_ps(pOutputd));
					_acce = _mm512_max_ps(_acce, _acc0);

					_acc0 = _mm512_max_ps(_acc0, _acc2);
					_acc1 = _mm512_max_ps(_acc1, _acc2);

					__m512 _accf = _mm512_max_ps(_acc0, _mm512_load_ps(pOutputf));

					_acc0 = _mm512_max_ps(_acc0, _mm512_load_ps(pOutput0));
					_acc1 = _mm512_max_ps(_acc1, _mm512_load_ps(pOutput1));
					_acc2 = _mm512_max_ps(_acc2, _mm512_load_ps(pOutput2));
					_acc3 = _mm512_max_ps(_acc3, _mm512_load_ps(pOutput3));

					// Store
					_mm512_store_ps(pOutputc, _accc);
					_mm512_store_ps(pOutputd, _accd);
					_mm512_store_ps(pOutpute, _acce);
					_mm512_store_ps(pOutputf, _accf);

					_mm512_store_ps(pOutput0, _acc0);
					_mm512_store_ps(pOutput1, _acc1);
					_mm512_store_ps(pOutput2, _acc2);
					_mm512_store_ps(pOutput3, _acc3);

					pInput0 += widthStep;
					pInput1 += widthStep;
					pInput2 += widthStep;
					pInput3 += widthStep;

					pOutputc += widthStep;
					pOutputd += widthStep;
					pOutpute += widthStep;
					pOutputf += widthStep;
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

					__m512 _acc0 = _mm512_max_ps(_b0_1, _b0);
					__m512 _acc1 = _mm512_max_ps(_b1_1, _b1);
					__m512 _acc2 = _mm512_max_ps(_b2_1, _b2);
					__m512 _acc3 = _mm512_max_ps(_b3_1, _b3);

					_b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
					_b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
					_b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
					_b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

					_acc0 = _mm512_max_ps(_acc0, _b0);
					_acc1 = _mm512_max_ps(_acc1, _b1);
					_acc2 = _mm512_max_ps(_acc2, _b2);
					_acc3 = _mm512_max_ps(_acc3, _b3);

					_b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
					_b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_3, _b1_1);
					_b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_3, _b2_1);
					_b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_3, _b3_1);

					_acc0 = _mm512_max_ps(_acc0, _b0);
					_acc1 = _mm512_max_ps(_acc1, _b1);
					_acc2 = _mm512_max_ps(_acc2, _b2);
					_acc3 = _mm512_max_ps(_acc3, _b3);

					_b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);
					_b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_4, _b1_1);
					_b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_4, _b2_1);
					_b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_4, _b3_1);

					_acc0 = _mm512_max_ps(_acc0, _b0);
					_acc1 = _mm512_max_ps(_acc1, _b1);
					_acc2 = _mm512_max_ps(_acc2, _b2);
					_acc3 = _mm512_max_ps(_acc3, _b3);

					if (widthOutputRemainder < widthInputRemainder) {
						// Prefix Sums
						__m512 _accc = _mm512_max_ps(_acc0, _mm512_load_ps(pOutputc));
						__m512 _acce = _mm512_max_ps(_acc2, _mm512_load_ps(pOutpute));

						_acc0 = _mm512_max_ps(_acc0, _acc1);
						_acc2 = _mm512_max_ps(_acc2, _acc3);

						__m512 _accd = _mm512_max_ps(_acc0, _mm512_load_ps(pOutputd));
						_acce = _mm512_max_ps(_acce, _acc0);

						_acc0 = _mm512_max_ps(_acc0, _acc2);
						_acc1 = _mm512_max_ps(_acc1, _acc2);

						__m512 _accf = _mm512_max_ps(_acc0, _mm512_load_ps(pOutputf));

						_acc0 = _mm512_max_ps(_acc0, _mm512_load_ps(pOutput0));
						_acc1 = _mm512_max_ps(_acc1, _mm512_load_ps(pOutput1));
						_acc2 = _mm512_max_ps(_acc2, _mm512_load_ps(pOutput2));
						_acc3 = _mm512_max_ps(_acc3, _mm512_load_ps(pOutput3));

						// Store
						_mm512_store_ps(pOutputc, _accc);
						_mm512_store_ps(pOutputd, _accd);
						_mm512_store_ps(pOutpute, _acce);
						_mm512_store_ps(pOutputf, _accf);

						_mm512_store_ps(pOutput0, _acc0);
						_mm512_store_ps(pOutput1, _acc1);
						_mm512_store_ps(pOutput2, _acc2);
						_mm512_store_ps(pOutput3, _acc3);

						pOutputc += widthStep;
						pOutputd += widthStep;
						pOutpute += widthStep;
						pOutputf += widthStep;
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

						_acc0 = _mm512_max_ps(_b0_1, _b0);
						_acc1 = _mm512_max_ps(_b1_1, _b1);
						_acc2 = _mm512_max_ps(_b2_1, _b2);
						_acc3 = _mm512_max_ps(_b3_1, _b3);

						_b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_2, _b0_1);
						_b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_2, _b1_1);
						_b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_2, _b2_1);
						_b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_2, _b3_1);

						_acc0 = _mm512_max_ps(_acc0, _b0);
						_acc1 = _mm512_max_ps(_acc1, _b1);
						_acc2 = _mm512_max_ps(_acc2, _b2);
						_acc3 = _mm512_max_ps(_acc3, _b3);

						_b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_3, _b0_1);
						_b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_3, _b1_1);
						_b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_3, _b2_1);
						_b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_3, _b3_1);

						_acc0 = _mm512_max_ps(_acc0, _b0);
						_acc1 = _mm512_max_ps(_acc1, _b1);
						_acc2 = _mm512_max_ps(_acc2, _b2);
						_acc3 = _mm512_max_ps(_acc3, _b3);

						_b0 = _mm512_permutex2var_ps(_b0_0, _roll_left_4, _b0_1);
						_b1 = _mm512_permutex2var_ps(_b1_0, _roll_left_4, _b1_1);
						_b2 = _mm512_permutex2var_ps(_b2_0, _roll_left_4, _b2_1);
						_b3 = _mm512_permutex2var_ps(_b3_0, _roll_left_4, _b3_1);

						_acc0 = _mm512_max_ps(_acc0, _b0);
						_acc1 = _mm512_max_ps(_acc1, _b1);
						_acc2 = _mm512_max_ps(_acc2, _b2);
						_acc3 = _mm512_max_ps(_acc3, _b3);
					}

					// Prefix Sums
					__m512 _accc = _mm512_max_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutputc));
					__m512 _acce = _mm512_max_ps(_acc2, _mm512_maskz_load_ps(_rightOutputMask, pOutpute));

					_acc0 = _mm512_max_ps(_acc0, _acc1);
					_acc2 = _mm512_max_ps(_acc2, _acc3);

					__m512 _accd = _mm512_max_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutputd));
					_acce = _mm512_max_ps(_acce, _acc0);

					_acc0 = _mm512_max_ps(_acc0, _acc2);
					_acc1 = _mm512_max_ps(_acc1, _acc2);

					__m512 _accf = _mm512_max_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutputf));

					_acc0 = _mm512_max_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutput0));
					_acc1 = _mm512_max_ps(_acc1, _mm512_maskz_load_ps(_rightOutputMask, pOutput1));
					_acc2 = _mm512_max_ps(_acc2, _mm512_maskz_load_ps(_rightOutputMask, pOutput2));
					_acc3 = _mm512_max_ps(_acc3, _mm512_maskz_load_ps(_rightOutputMask, pOutput3));

					// Store
					_mm512_mask_store_ps(pOutputc, _rightOutputMask, _accc);
					_mm512_mask_store_ps(pOutputd, _rightOutputMask, _accd);
					_mm512_mask_store_ps(pOutpute, _rightOutputMask, _acce);
					_mm512_mask_store_ps(pOutputf, _rightOutputMask, _accf);

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

				for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
					if ((0 <= gk) && (gk < OutputHeight)) {
						__m512 _acc1 = _mm512_load_ps(pOutput1);

						_acc1 = _mm512_max_ps(_acc0, _acc1);

						_mm512_store_ps(pOutput1, _acc1);
					}

					pOutput1 -= OutputWidth;
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

					for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
						if ((0 <= gk) && (gk < OutputHeight)) {
							__m512 _acc1 = _mm512_load_ps(pOutput1);

							_acc1 = _mm512_max_ps(_acc0, _acc1);

							_mm512_store_ps(pOutput1, _acc1);
						}

						pOutput1 -= OutputWidth;
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

				for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
					if ((0 <= gk) && (gk < OutputHeight)) {
						__m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

						_acc1 = _mm512_max_ps(_acc0, _acc1);

						_mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
					}

					pOutput1 -= OutputWidth;
				}
			}

			pInputRow += InputWidth;

			pOutputRow += OutputWidth;
		}

		Input += InputSize;

		Output += OutputSize;
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

	const int64_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
	const int64_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
	const int64_t InputSize = WorkBlock->InputSize;
	const int64_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
	const int64_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
	const int64_t OutputSize = OutputHeight * OutputWidth;

	const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
	const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
	const int64_t PaddingTop = WorkBlock->Padding[HeightShapeIndex];
	const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
	const int64_t PaddingBottom = WorkBlock->Padding[Dimensions + HeightShapeIndex];
	const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

	assert(OutputHeight == (InputHeight - KernelHeight + PaddingTop + PaddingBottom + 1));
	assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
    MLAS_UNREFERENCED_PARAMETER(PaddingBottom);
	MLAS_UNREFERENCED_PARAMETER(PaddingRight);

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

	const __mmask32 _rightInputMask = static_cast<__mmask32>(~(~0 << widthInputRemainder));

	const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
	const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);

	const __mmask32 _rightOutputMask = static_cast<__mmask32>(~(~0 << widthOutputRemainder));

	const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
	const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);

	const __m512i _roll_left_1 =
		_mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
	const __m512i _one = _mm512_set1_epi32(1);

	const __m512 _padding = _mm512_set1_ps(std::numeric_limits<float>::lowest());

	for (size_t c = 0; c < ChannelCount; c++) {
		float* pOutputRow = Output + PaddingTop * OutputWidth;

		const float* pInputRow = Input;

		// Outer loop
		for (int64_t i = 0; i < InputHeight; i++) {
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

				for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
					if ((0 <= gk) && (gk < OutputHeight)) {
						__m512 _acc1_x = _mm512_load_ps(pOutput1);
						__m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

						_acc1_x = _mm512_max_ps(_acc1_x, _acc1);
						_acc2_x = _mm512_max_ps(_acc2_x, _acc2);

						_mm512_store_ps(pOutput1, _acc1_x);
						_mm512_store_ps(pOutput1 + 16, _acc2_x);
					}

					pOutput1 -= OutputWidth;
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

					for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
						if ((0 <= gk) && (gk < OutputHeight)) {
							__m512 _acc1_x = _mm512_load_ps(pOutput1);
							__m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

							_acc1_x = _mm512_max_ps(_acc1_x, _acc1);
							_acc2_x = _mm512_max_ps(_acc2_x, _acc2);

							_mm512_store_ps(pOutput1, _acc1_x);
							_mm512_store_ps(pOutput1 + 16, _acc2_x);
						}

						pOutput1 -= OutputWidth;
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

				for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
					if ((0 <= gk) && (gk < OutputHeight)) {
						__m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask1, pOutput1);
						__m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask2, pOutput1 + 16);

						_acc1_x = _mm512_max_ps(_acc1_x, _acc1);
						_acc2_x = _mm512_max_ps(_acc2_x, _acc2);

						_mm512_mask_store_ps(pOutput1, _rightOutputMask1, _acc1_x);
						_mm512_mask_store_ps(pOutput1 + 16, _rightOutputMask2, _acc2_x);
					}

					pOutput1 -= OutputWidth;
				}
			}

			pInputRow += InputWidth;

			pOutputRow += OutputWidth;
		}

		Input += InputSize;

		Output += OutputSize;
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
	}
	else if ((KernelHeight == 5) && (KernelWidth == 5)) {
		MlasPool2DSlidingKernelMaxK5x5S1(WorkBlock, ChannelCount, Input, Output);
	}
	else if (KernelWidth <= 16) {
		MlasPool2DSlidingKernelMaxK17S1(WorkBlock, ChannelCount, Input, Output);
	}
	else if (KernelWidth <= 32) {
		MlasPool2DSlidingKernelMaxK32S1(WorkBlock, ChannelCount, Input, Output);
	}
	else
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

	const int64_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
	const int64_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
	const int64_t InputSize = WorkBlock->InputSize;
	const int64_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
	const int64_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
	const int64_t OutputSize = OutputHeight * OutputWidth;

	const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
	const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
	const int64_t PaddingTop = WorkBlock->Padding[HeightShapeIndex];
	const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
	const int64_t PaddingBottom = WorkBlock->Padding[Dimensions + HeightShapeIndex];
	const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

	assert(OutputHeight == (InputHeight - KernelHeight + PaddingTop + PaddingBottom + 1));
	assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
    MLAS_UNREFERENCED_PARAMETER(PaddingBottom);
    MLAS_UNREFERENCED_PARAMETER(PaddingRight);

	constexpr size_t widthStep = 16;

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

	const __m512 _weight = _mm512_set1_ps(1.0f / (float(KernelHeight) * float(KernelWidth)));

	const __m512i _roll_left_1 =
		_mm512_set_epi32(30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15);
	const __m512i _one = _mm512_set1_epi32(1);

	for (size_t c = 0; c < ChannelCount; c++) {
		float* pOutputRow = Output + PaddingTop * OutputWidth;

		const float* pInputRow = Input;

		// Outer loop
		for (int64_t i = 0; i < InputHeight; i++) {
			const float* pInput0 = pInputRow;

			float* pOutput0 = pOutputRow;

			__m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);

			pInput0 += paddingLeftSeek;

			for (int64_t j = 0; j < widthIterations; j++) {
				__m512 _b0_1 = _mm512_load_ps(pInput0);

				// Inner Loop
				__m512i _roll = _roll_left_1;

				__m512 _acc0 = _b0_1;

				for (int64_t l = 1; l < KernelWidth; l++) {
					__m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

					_acc0 = _mm512_add_ps(_acc0, _b0);

					_roll = _mm512_sub_epi32(_roll, _one);
				}

				_acc0 = _mm512_mul_ps(_acc0, _weight);

				_b0_0 = _b0_1;

				// Store
				float* pOutput1 = pOutput0;

				for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
					if ((0 <= gk) && (gk < OutputHeight)) {
						__m512 _acc1 = _mm512_load_ps(pOutput1);

						_acc1 = _mm512_add_ps(_acc0, _acc1);

						_mm512_store_ps(pOutput1, _acc1);
					}

					pOutput1 -= OutputWidth;
				}

				pInput0 += widthStep;
				pOutput0 += widthStep;
			}

			// Right Edge
			{
				__m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);

				// Inner Loop
				__m512i _roll = _roll_left_1;

				__m512 _acc0 = _b0_1;

				for (int64_t l = 1; l < KernelWidth; l++) {
					__m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

					_acc0 = _mm512_add_ps(_acc0, _b0);

					_roll = _mm512_sub_epi32(_roll, _one);
				}

				_acc0 = _mm512_mul_ps(_acc0, _weight);

				if (widthOutputRemainder < widthInputRemainder) {
					float* pOutput1 = pOutput0;

					for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
						if ((0 <= gk) && (gk < OutputHeight)) {
							__m512 _acc1 = _mm512_load_ps(pOutput1);

							_acc1 = _mm512_add_ps(_acc0, _acc1);

							_mm512_store_ps(pOutput1, _acc1);
						}

						pOutput1 -= OutputWidth;
					}

					pOutput0 += widthStep;

					_b0_0 = _b0_1;

					_b0_1 = _mm512_setzero_ps();

					// Inner Loop
					_roll = _roll_left_1;

					_acc0 = _b0_1;

					for (int64_t l = 1; l < KernelWidth; l++) {
						__m512 _b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);

						_acc0 = _mm512_add_ps(_acc0, _b0);

						_roll = _mm512_sub_epi32(_roll, _one);
					}

					_acc0 = _mm512_mul_ps(_acc0, _weight);
				}

				float* pOutput1 = pOutput0;

				for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
					if ((0 <= gk) && (gk < OutputHeight)) {
						__m512 _acc1 = _mm512_maskz_load_ps(_rightOutputMask, pOutput1);

						_acc1 = _mm512_add_ps(_acc0, _acc1);

						_mm512_mask_store_ps(pOutput1, _rightOutputMask, _acc1);
					}

					pOutput1 -= OutputWidth;
				}
			}

			pInputRow += InputWidth;

			pOutputRow += OutputWidth;
		}

		Input += InputSize;

		Output += OutputSize;
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

	const int64_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
	const int64_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
	const int64_t InputSize = WorkBlock->InputSize;
	const int64_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
	const int64_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
	const int64_t OutputSize = OutputHeight * OutputWidth;

	const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
	const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
	const int64_t PaddingTop = WorkBlock->Padding[HeightShapeIndex];
	const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
	const int64_t PaddingBottom = WorkBlock->Padding[Dimensions + HeightShapeIndex];
	const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

	assert(OutputHeight == (InputHeight - KernelHeight + PaddingTop + PaddingBottom + 1));
	assert(OutputWidth == (InputWidth - KernelWidth + PaddingLeft + PaddingRight + 1));
    MLAS_UNREFERENCED_PARAMETER(PaddingBottom);
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

	for (size_t c = 0; c < ChannelCount; c++) {
		float* pOutputRow = Output + PaddingTop * OutputWidth;

		const float* pInputRow = Input;

		// Outer loop
		for (int64_t i = 0; i < InputHeight; i++) {
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

				for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
					if ((0 <= gk) && (gk < OutputHeight)) {
						__m512 _acc1_x = _mm512_load_ps(pOutput1);
						__m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

						_acc1_x = _mm512_add_ps(_acc1_x, _acc1);
						_acc2_x = _mm512_add_ps(_acc2_x, _acc2);

						_mm512_store_ps(pOutput1, _acc1_x);
						_mm512_store_ps(pOutput1 + 16, _acc2_x);
					}

					pOutput1 -= OutputWidth;
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

					for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
						if ((0 <= gk) && (gk < OutputHeight)) {
							__m512 _acc1_x = _mm512_load_ps(pOutput1);
							__m512 _acc2_x = _mm512_load_ps(pOutput1 + 16);

							_acc1_x = _mm512_add_ps(_acc1_x, _acc1);
							_acc2_x = _mm512_add_ps(_acc2_x, _acc2);

							_mm512_store_ps(pOutput1, _acc1_x);
							_mm512_store_ps(pOutput1 + 16, _acc2_x);
						}

						pOutput1 -= OutputWidth;
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

				for (int64_t k = 0, gk = i + PaddingTop; k < KernelHeight; k++, gk--) {
					if ((0 <= gk) && (gk < OutputHeight)) {
						__m512 _acc1_x = _mm512_maskz_load_ps(_rightOutputMask1, pOutput1);
						__m512 _acc2_x = _mm512_maskz_load_ps(_rightOutputMask2, pOutput1 + 16);

						_acc1_x = _mm512_add_ps(_acc1_x, _acc1);
						_acc2_x = _mm512_add_ps(_acc2_x, _acc2);

						_mm512_mask_store_ps(pOutput1, _rightOutputMask1, _acc1_x);
						_mm512_mask_store_ps(pOutput1 + 16, _rightOutputMask2, _acc2_x);
					}

					pOutput1 -= OutputWidth;
				}
			}

			pInputRow += InputWidth;

			pOutputRow += OutputWidth;
		}

		Input += InputSize;

		Output += OutputSize;
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
	}
	else if (KernelWidth <= 32) {
		MlasPool2DSlidingKernelAvgWithPadK32S1(WorkBlock, ChannelCount, Input, Output);
	}
	else
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
		{WorkBlock->StrideShape[0], WorkBlock->StrideShape[1], WorkBlock->StrideShape[2]} };

	if (KernelWidth <= 16) {
		MlasPool2DSlidingKernelAvgWithPadK17S1(&NewBlock, ChannelCount, Input, Output);
	}
	else if (KernelWidth <= 32) {
		MlasPool2DSlidingKernelAvgWithPadK32S1(&NewBlock, ChannelCount, Input, Output);
	}
	else
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

void
MlasPool3DSlidingKernelMax(const MLAS_POOL_WORK_BLOCK* WorkBlock,
	size_t ChannelCount,
	const float* Input,
	float* Output)
{
	constexpr size_t Dimensions = 3;
	constexpr size_t DepthShapeIndex = 0;
	constexpr size_t HeightShapeIndex = 1;
	constexpr size_t WidthShapeIndex = 2;

	const size_t InputDepth = WorkBlock->InputShape[DepthShapeIndex];
	const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
	const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
	const size_t InputArea = InputHeight * InputWidth;

	const size_t OutputDepth = WorkBlock->OutputShape[DepthShapeIndex];
	const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
	const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
	const size_t OutputArea = OutputHeight * OutputWidth;

	const int64_t KernelDepth = WorkBlock->KernelShape[DepthShapeIndex];
	const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
	const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];

	const int64_t PaddingFront = WorkBlock->Padding[DepthShapeIndex];
	const int64_t PaddingTop = WorkBlock->Padding[HeightShapeIndex];
	const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
	const int64_t PaddingBottom = WorkBlock->Padding[Dimensions + HeightShapeIndex];
	const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

	MLAS_POOL_WORK_BLOCK SubTaskParameters{
		WorkBlock->PoolingKind,
		{InputHeight, InputWidth, 0},
		InputArea,
		{OutputHeight, OutputWidth, 0},
		{KernelHeight, KernelWidth, 0},
		{PaddingTop, PaddingLeft, PaddingBottom, PaddingRight} };

	Output += PaddingFront * OutputArea;

	// Outer loop
	for (size_t i = 0; i < InputDepth; i++) {
		float* pOutputLayer = Output;

		for (int64_t k = 0, gk = i + PaddingFront; k < KernelDepth; k++, gk--) {
			if ((0 <= gk) && (gk < int64_t(OutputDepth))) {
				MlasPool2DSlidingKernelMax(&SubTaskParameters, ChannelCount, Input, pOutputLayer);
			}

			pOutputLayer -= OutputArea;
		}

		Input += InputArea;

		Output += OutputArea;
	}
}

void
MlasPool3DSlidingKernelAvgWithPad(const MLAS_POOL_WORK_BLOCK* WorkBlock,
	size_t ChannelCount,
	const float* Input,
	float* Output)
{
	constexpr size_t Dimensions = 3;
	constexpr size_t DepthShapeIndex = 0;
	constexpr size_t HeightShapeIndex = 1;
	constexpr size_t WidthShapeIndex = 2;

	const size_t InputDepth = WorkBlock->InputShape[DepthShapeIndex];
	const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
	const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
	const size_t InputArea = InputHeight * InputWidth;

	const size_t OutputDepth = WorkBlock->OutputShape[DepthShapeIndex];
	const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
	const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
	const size_t OutputArea = OutputHeight * OutputWidth;
	const size_t OutputSize = OutputDepth * OutputArea;

	const int64_t KernelDepth = WorkBlock->KernelShape[DepthShapeIndex];
	const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
	const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
	const int64_t KernelArea = KernelHeight * KernelWidth;
	const int64_t KernelSize = KernelDepth * KernelArea;

	const int64_t PaddingFront = WorkBlock->Padding[DepthShapeIndex];
	const int64_t PaddingTop = WorkBlock->Padding[HeightShapeIndex];
	const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
	const int64_t PaddingBottom = WorkBlock->Padding[Dimensions + HeightShapeIndex];
	const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

	MLAS_POOL_WORK_BLOCK SubTaskParameters{
		WorkBlock->PoolingKind,
		{InputHeight, InputWidth, 0},
		InputArea,
		{OutputHeight, OutputWidth, 0},
		{KernelHeight, KernelWidth, 0},
		{PaddingTop, PaddingLeft, PaddingBottom, PaddingRight} };

	float* pOutputLayer = Output + PaddingFront * OutputArea;

	// Outer loop
	for (size_t i = 0; i < InputDepth; i++) {
		float* pOutput0 = pOutputLayer;

		for (int64_t k = 0, gk = i + PaddingFront; k < KernelDepth; k++, gk--) {
			if ((0 <= gk) && (gk < int64_t(OutputDepth))) {
				MlasPool2DSlidingKernelAvgWithPad(&SubTaskParameters, ChannelCount, Input, pOutput0);
			}

			pOutput0 -= OutputArea;
		}

		Input += InputArea;

		pOutputLayer += OutputArea;
	}

	// Adjustment
	const float factor = float(KernelArea) / float(KernelSize);

	for (size_t i = 0; i < OutputSize; i++) {
		Output[i] *= factor;
	}
}

void
MlasPool3DSlidingKernelAvgNoPad(const MLAS_POOL_WORK_BLOCK* WorkBlock,
	size_t ChannelCount,
	const float* Input,
	float* Output)
{
	constexpr size_t Dimensions = 3;
	constexpr size_t DepthShapeIndex = 0;
	constexpr size_t HeightShapeIndex = 1;
	constexpr size_t WidthShapeIndex = 2;

	const size_t InputDepth = WorkBlock->InputShape[DepthShapeIndex];
	const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
	const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
	const size_t InputArea = InputHeight * InputWidth;

	const size_t OutputDepth = WorkBlock->OutputShape[DepthShapeIndex];
	const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
	const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
	const size_t OutputArea = OutputHeight * OutputWidth;

	const int64_t KernelDepth = WorkBlock->KernelShape[DepthShapeIndex];
	const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
	const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];

	const int64_t PaddingFront = WorkBlock->Padding[DepthShapeIndex];
	const int64_t PaddingTop = WorkBlock->Padding[HeightShapeIndex];
	const int64_t PaddingLeft = WorkBlock->Padding[WidthShapeIndex];
	const int64_t PaddingBack = WorkBlock->Padding[Dimensions + DepthShapeIndex];
	const int64_t PaddingBottom = WorkBlock->Padding[Dimensions + HeightShapeIndex];
	const int64_t PaddingRight = WorkBlock->Padding[Dimensions + WidthShapeIndex];

	assert(WorkBlock->PoolingKind == MLAS_POOLING_KIND::MlasAveragePoolingExcludePad);

	MLAS_POOL_WORK_BLOCK SubTaskParameters{
		MLAS_POOLING_KIND::MlasAveragePoolingIncludePad,
		{InputHeight, InputWidth, 0},
		InputArea,
		{OutputHeight, OutputWidth, 0},
		{KernelHeight, KernelWidth, 0},
		{PaddingTop, PaddingLeft, PaddingBottom, PaddingRight} };

	float* pOutputLayer = Output + PaddingFront * OutputArea;

	// Outer loop
	for (size_t i = 0; i < InputDepth; i++) {
		float* pOutput0 = pOutputLayer;

		for (int64_t k = 0, gk = i + PaddingFront; k < KernelDepth; k++, gk--) {
			if ((0 <= gk) && (gk < int64_t(OutputDepth))) {
				MlasPool2DSlidingKernelAvgWithPad(&SubTaskParameters, ChannelCount, Input, pOutput0);
			}

			pOutput0 -= OutputArea;
		}

		Input += InputArea;

		pOutputLayer += OutputArea;
	}

	// Adjustment
	float shape = float(KernelHeight) * float(KernelWidth);

	float* pOuputRow = Output;

	for (int64_t i = PaddingFront; i > 0; i--) {
		for (int64_t j = PaddingTop; j > 0; j--) {
			float* pBegin = pOuputRow;

			for (int64_t k = PaddingLeft; k > 0; k--) {
				*pBegin++ *= shape / ((KernelDepth - i) * (KernelHeight - j) * (KernelWidth - k));
			}

			float* pEnd = pOuputRow + OutputWidth;

			for (int64_t k = PaddingRight; k > 0; k--) {
				*--pEnd *= shape / ((KernelDepth - i) * (KernelHeight - j) * (KernelWidth - k));
			}

			while (pBegin < pEnd) {
				*pBegin++ *= shape / ((KernelDepth - i) * (KernelHeight - j) * KernelWidth);
			}

			pOuputRow += OutputWidth;
		}

		for (int64_t j = PaddingTop; j < int64_t(OutputHeight - PaddingBottom); j++) {
			float* pBegin = pOuputRow;

			for (int64_t k = PaddingLeft; k > 0; k--) {
				*pBegin++ *= shape / ((KernelDepth - i) * KernelHeight * (KernelWidth - k));
			}

			float* pEnd = pOuputRow + OutputWidth;

			for (int64_t k = PaddingRight; k > 0; k--) {
				*--pEnd *= shape / ((KernelDepth - i) * KernelHeight * (KernelWidth - k));
			}

			while (pBegin < pEnd) {
				*pBegin++ *= shape / ((KernelDepth - i) * KernelHeight * KernelWidth);
			}

			pOuputRow += OutputWidth;
		}

		for (int64_t j = 1; j <= PaddingBottom; j++) {
			float* pBegin = pOuputRow;

			for (int64_t k = PaddingLeft; k > 0; k--) {
				*pBegin++ *= shape / ((KernelDepth - i) * (KernelHeight - j) * (KernelWidth - k));
			}

			float* pEnd = pOuputRow + OutputWidth;

			for (int64_t k = PaddingRight; k > 0; k--) {
				*--pEnd *= shape / ((KernelDepth - i) * (KernelHeight - j) * (KernelWidth - k));
			}

			while (pBegin < pEnd) {
				*pBegin++ *= shape / ((KernelDepth - i) * (KernelHeight - j) * KernelWidth);
			}

			pOuputRow += OutputWidth;
		}
	}

	for (int64_t i = PaddingFront; i < int64_t(OutputDepth - PaddingBack); i++) {
		for (int64_t j = PaddingTop; j > 0; j--) {
			float* pBegin = pOuputRow;

			for (int64_t k = PaddingLeft; k > 0; k--) {
				*pBegin++ *= shape / (KernelDepth * (KernelHeight - j) * (KernelWidth - k));
			}

			float* pEnd = pOuputRow + OutputWidth;

			for (int64_t k = PaddingRight; k > 0; k--) {
				*--pEnd *= shape / (KernelDepth * (KernelHeight - j) * (KernelWidth - k));
			}

			while (pBegin < pEnd) {
				*pBegin++ *= shape / (KernelDepth * (KernelHeight - j) * KernelWidth);
			}

			pOuputRow += OutputWidth;
		}

		for (int64_t j = PaddingTop; j < int64_t(OutputHeight - PaddingBottom); j++) {
			float* pBegin = pOuputRow;

			for (int64_t k = PaddingLeft; k > 0; k--) {
				*pBegin++ *= shape / (KernelDepth * KernelHeight * (KernelWidth - k));
			}

			float* pEnd = pOuputRow + OutputWidth;

			for (int64_t k = PaddingRight; k > 0; k--) {
				*--pEnd *= shape / (KernelDepth * KernelHeight * (KernelWidth - k));
			}

			while (pBegin < pEnd) {
				*pBegin++ *= shape / (KernelDepth * KernelHeight * KernelWidth);
			}

			pOuputRow += OutputWidth;
		}

		for (int64_t j = 1; j <= PaddingBottom; j++) {
			float* pBegin = pOuputRow;

			for (int64_t k = PaddingLeft; k > 0; k--) {
				*pBegin++ *= shape / (KernelDepth * (KernelHeight - j) * (KernelWidth - k));
			}

			float* pEnd = pOuputRow + OutputWidth;

			for (int64_t k = PaddingRight; k > 0; k--) {
				*--pEnd *= shape / (KernelDepth * (KernelHeight - j) * (KernelWidth - k));
			}

			while (pBegin < pEnd) {
				*pBegin++ *= shape / (KernelDepth * (KernelHeight - j) * KernelWidth);
			}

			pOuputRow += OutputWidth;
		}
	}
	
	for (int64_t i = 1; i <= PaddingBack; i++) {
		for (int64_t j = PaddingTop; j > 0; j--) {
			float* pBegin = pOuputRow;

			for (int64_t k = PaddingLeft; k > 0; k--) {
				*pBegin++ *= shape / ((KernelDepth - i) * (KernelHeight - j) * (KernelWidth - k));
			}

			float* pEnd = pOuputRow + OutputWidth;

			for (int64_t k = PaddingRight; k > 0; k--) {
				*--pEnd *= shape / ((KernelDepth - i) * (KernelHeight - j) * (KernelWidth - k));
			}

			while (pBegin < pEnd) {
				*pBegin++ *= shape / ((KernelDepth - i) * (KernelHeight - j) * KernelWidth);
			}

			pOuputRow += OutputWidth;
		}

		for (int64_t j = PaddingTop; j < int64_t(OutputHeight - PaddingBottom); j++) {
			float* pBegin = pOuputRow;

			for (int64_t k = PaddingLeft; k > 0; k--) {
				*pBegin++ *= shape / ((KernelDepth - i) * KernelHeight * (KernelWidth - k));
			}

			float* pEnd = pOuputRow + OutputWidth;

			for (int64_t k = PaddingRight; k > 0; k--) {
				*--pEnd *= shape / ((KernelDepth - i) * KernelHeight * (KernelWidth - k));
			}

			while (pBegin < pEnd) {
				*pBegin++ *= shape / ((KernelDepth - i) * KernelHeight * KernelWidth);
			}

			pOuputRow += OutputWidth;
		}

		for (int64_t j = 1; j <= PaddingBottom; j++) {
			float* pBegin = pOuputRow;

			for (int64_t k = PaddingLeft; k > 0; k--) {
				*pBegin++ *= shape / ((KernelDepth - i) * (KernelHeight - j) * (KernelWidth - k));
			}

			float* pEnd = pOuputRow + OutputWidth;

			for (int64_t k = PaddingRight; k > 0; k--) {
				*--pEnd *= shape / ((KernelDepth - i) * (KernelHeight - j) * (KernelWidth - k));
			}

			while (pBegin < pEnd) {
				*pBegin++ *= shape / ((KernelDepth - i) * (KernelHeight - j) * KernelWidth);
			}

			pOuputRow += OutputWidth;
		}
	}
}
