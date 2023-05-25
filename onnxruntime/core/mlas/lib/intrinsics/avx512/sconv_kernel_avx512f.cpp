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
    MLAS_UNREFERENCED_PARAMETER(OutputSize);

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

	const int64_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];
	const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
	const int64_t PaddingLeft = Parameters->Padding[WidthShapeIndex];
	const int64_t PaddingRight = Parameters->Padding[Dimensions + WidthShapeIndex];

	const int64_t DilatedKernelWidth = (KernelWidth - 1) * DilationWidth + 1;

	assert(OutputWidth == (InputWidth - DilatedKernelWidth + PaddingLeft + PaddingRight + 1));
	MLAS_UNREFERENCED_PARAMETER(PaddingRight);

	constexpr int64_t widthStep = 16;

	const int64_t paddingLeftSeek = DilatedKernelWidth - PaddingLeft - 1;

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

	const __m512i _roll_init =
		_mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16);

	const __m512 _zero = _mm512_setzero_ps();

	const __m512i _dilation = _mm512_set1_epi32(static_cast<int>(DilationWidth));

	float* pOutput0 = Output;

	const float* pInput0 = Input;

	__m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask, pInput0 - paddingLeftShift);

	pInput0 += paddingLeftSeek;

	for (int64_t i = 0; i < widthIterations; i++) {
		__m512 _b0_1 = _mm512_load_ps(pInput0);

		__m512 _b0 = _b0_1;

		__m512i _roll = _roll_init;

		__m512 _acc0 = _mm512_load_ps(pOutput0);

		for (int64_t k = KernelWidth; k-- > 0;) {
			__m512 _ak = _mm512_set1_ps(Filter[k]);

			_acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);

			_roll = _mm512_sub_epi32(_roll, _dilation);

			_b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
		}

		_b0_0 = _b0_1;

		// Store
		_mm512_store_ps(pOutput0, _acc0);

		pInput0 += widthStep;
		pOutput0 += widthStep;
	}

	// Right Edge
	{
		__m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);

		__m512 _b0 = _b0_1;

		__m512i _roll = _roll_init;

		__m512 _acc0 = _zero;

		for (int64_t k = KernelWidth; k-- > 0;) {
			__m512 _ak = _mm512_set1_ps(Filter[k]);

			_acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);

			_roll = _mm512_sub_epi32(_roll, _dilation);

			_b0 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
		}

		if (widthOutputRemainder < widthInputRemainder) {
			_acc0 = _mm512_add_ps(_acc0, _mm512_load_ps(pOutput0));

			_mm512_store_ps(pOutput0, _acc0);

			pOutput0 += widthStep;

			_b0 = _zero;

			_roll = _roll_init;

			_acc0 = _zero;

			for (int64_t k = KernelWidth; k-- > 0;) {
				__m512 _ak = _mm512_set1_ps(Filter[k]);

				_acc0 = _mm512_fmadd_ps(_ak, _b0, _acc0);

				_roll = _mm512_sub_epi32(_roll, _dilation);

				_b0 = _mm512_permutex2var_ps(_b0_1, _roll, _zero);
			}
		}

		// Store
		_acc0 = _mm512_add_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutput0));

		_mm512_mask_store_ps(pOutput0, _rightOutputMask, _acc0);
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

	const int64_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];
	const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
	const int64_t PaddingLeft = Parameters->Padding[WidthShapeIndex];
	const int64_t PaddingRight = Parameters->Padding[Dimensions + WidthShapeIndex];

	const int64_t DilatedKernelWidth = (KernelWidth - 1) * DilationWidth + 1;

	assert(OutputWidth == (InputWidth - DilatedKernelWidth + PaddingLeft + PaddingRight + 1));
	MLAS_UNREFERENCED_PARAMETER(PaddingRight);

	const int64_t kernelPeriod = 16 / DilationWidth;

	const int64_t kernelRemainder = DilatedKernelWidth - 16;

	constexpr size_t widthStep = 32;

	const int64_t paddingLeftSeek = DilatedKernelWidth - PaddingLeft - 1;

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

	const __mmask64 _rightInputMask = ~(~0ull << widthInputRemainder);

	const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
	const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);

	const __mmask64 _rightOutputMask = ~(~0ull << widthOutputRemainder);

	const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
	const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);

	const __m512i _roll_init =
		_mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16);

	const __m512i _dilation = _mm512_set1_epi32(static_cast<int>(DilationWidth));

	float* pOutput0 = Output;

	const float* pInput0 = Input;

	__m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift + 0);
	__m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);

	pInput0 += paddingLeftSeek;

	for (int64_t i = 0; i < widthIterations; i++) {
		int64_t idx0 = KernelWidth;
		int64_t idx1 = KernelWidth - kernelPeriod;

		__m512 _b0_1 = _mm512_load_ps(pInput0);
		__m512 _b0_2 = _mm512_load_ps(pInput0 + 16);

		__m512 _acc0_1 = _mm512_load_ps(pOutput0);
		__m512 _acc0_2 = _mm512_load_ps(pOutput0 + 16);

		__m512 _acc1_0 = _mm512_setzero_ps();
		__m512 _acc1_1 = _mm512_setzero_ps();

		__m512 _b0 = _b0_0;
		__m512 _b1 = _b0_1;
		__m512 _b2 = _b0_2;

		__m512i _roll = _roll_init;

		int64_t k = 0;
		for (; k < kernelRemainder; k += DilationWidth) {
			__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
			__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);

			_acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
			_acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

			_acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
			_acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

			_roll = _mm512_sub_epi32(_roll, _dilation);

			_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
			_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
			_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
		}

		for (; k < 16ll; k += DilationWidth) {
			__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);

			_acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
			_acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

			_roll = _mm512_sub_epi32(_roll, _dilation);

			_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
			_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
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
		int64_t idx0 = KernelWidth;
		int64_t idx1 = KernelWidth - kernelPeriod;

		__m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0 + 0);
		__m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);

		__m512 _acc0_1 = _mm512_setzero_ps();
		__m512 _acc0_2 = _mm512_setzero_ps();

		__m512 _acc1_0 = _mm512_setzero_ps();
		__m512 _acc1_1 = _mm512_setzero_ps();

		__m512 _b0 = _b0_0;
		__m512 _b1 = _b0_1;
		__m512 _b2 = _b0_2;

		__m512i _roll = _roll_init;

		int64_t k = 0;
		for (; k < kernelRemainder; k += DilationWidth) {
			__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
			__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);

			_acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
			_acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

			_acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
			_acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

			_roll = _mm512_sub_epi32(_roll, _dilation);

			_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
			_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
			_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
		}

		for (; k < 16ll; k += DilationWidth) {
			__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);

			_roll = _mm512_sub_epi32(_roll, _dilation);

			_acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
			_acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

			_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
			_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
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

			idx0 = KernelWidth;
			idx1 = KernelWidth - kernelPeriod;

			_acc0_1 = _acc0_2 = _mm512_setzero_ps();

			_acc1_0 = _acc1_1 = _mm512_setzero_ps();

			_b0 = _b0_0;
			_b1 = _b0_1;
			_b2 = _b0_2;

			_roll = _roll_init;

			for (k = 0; k < kernelRemainder; k += DilationWidth) {
				__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
				__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);

				_acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
				_acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);

				_acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
				_acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

				_roll = _mm512_sub_epi32(_roll, _dilation);

				_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
				_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
				_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
			}

			for (; k < 16ll; k += DilationWidth) {
				__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);

				_acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
				_acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);

				_roll = _mm512_sub_epi32(_roll, _dilation);

				_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
				_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
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

	const int64_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];
	const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
	const int64_t PaddingLeft = Parameters->Padding[WidthShapeIndex];
	const int64_t PaddingRight = Parameters->Padding[Dimensions + WidthShapeIndex];

	const int64_t DilatedKernelWidth = (KernelWidth - 1) * DilationWidth + 1;

	assert(OutputWidth == (InputWidth - DilatedKernelWidth + PaddingLeft + PaddingRight + 1));
	MLAS_UNREFERENCED_PARAMETER(PaddingRight);

	const int64_t kernelPeriod = 16 / DilationWidth;

	const int64_t kernelRemainder = DilatedKernelWidth - 32;

	constexpr size_t widthStep = 48;

	const int64_t paddingLeftSeek = DilatedKernelWidth - PaddingLeft - 1;

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

	const __mmask64 _rightInputMask = ~(~0ull << widthInputRemainder);

	const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMask);
	const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMask >> 16);
	const __mmask16 _rightInputMask3 = static_cast<__mmask16>(_rightInputMask >> 32);

	const __mmask64 _rightOutputMask = ~(~0ull << widthOutputRemainder);

	const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMask);
	const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMask >> 16);
	const __mmask16 _rightOutputMask3 = static_cast<__mmask16>(_rightOutputMask >> 32);

	const __m512i _roll_init =
		_mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16);

	const __m512i _dilation = _mm512_set1_epi32(static_cast<int>(DilationWidth));

	float* pOutput0 = Output;

	const float* pInput0 = Input;

	__m512 _b0_e = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift + 0);
	__m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);
	__m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask3, pInput0 - paddingLeftShift + 32);

	pInput0 += paddingLeftSeek;

	for (int64_t i = 0; i < widthIterations; i++) {
		int64_t idx0 = KernelWidth;
		int64_t idx1 = KernelWidth - kernelPeriod;
		int64_t idx2 = KernelWidth - 2 * kernelPeriod;

		__m512 _b0_1 = _mm512_load_ps(pInput0);
		__m512 _b0_2 = _mm512_load_ps(pInput0 + 16);
		__m512 _b0_3 = _mm512_load_ps(pInput0 + 32);

		__m512 _acc0_1 = _mm512_load_ps(pOutput0);
		__m512 _acc0_2 = _mm512_load_ps(pOutput0 + 16);
		__m512 _acc0_3 = _mm512_load_ps(pOutput0 + 32);

		__m512 _acc1_0 = _mm512_setzero_ps();
		__m512 _acc1_1 = _mm512_setzero_ps();
		__m512 _acc1_2 = _mm512_setzero_ps();

		__m512 _bf = _b0_f;
		__m512 _b0 = _b0_0;
		__m512 _b1 = _b0_1;
		__m512 _b2 = _b0_2;
		__m512 _b3 = _b0_3;

		__m512i _roll = _roll_init;

		int64_t k = 0;
		for (; k < kernelRemainder; k += DilationWidth) {
			__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
			__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);
			__m512 _aj2 = _mm512_set1_ps(Filter[--idx2]);

			_acc0_1 = _mm512_fmadd_ps(_aj2, _bf, _acc0_1);
			_acc0_2 = _mm512_fmadd_ps(_aj2, _b0, _acc0_2);
			_acc0_3 = _mm512_fmadd_ps(_aj2, _b1, _acc0_3);

			_acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
			_acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
			_acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);

			_acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
			_acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
			_acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);

			_roll = _mm512_sub_epi32(_roll, _dilation);

			_bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);
			_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
			_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
			_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
			_b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
		}

		for (; k < 16ll; k += DilationWidth) {
			__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
			__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);

			_acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
			_acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
			_acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);

			_acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
			_acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
			_acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);

			_roll = _mm512_sub_epi32(_roll, _dilation);

			_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
			_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
			_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
			_b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
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
		int64_t idx0 = KernelWidth;
		int64_t idx1 = KernelWidth - kernelPeriod;
		int64_t idx2 = KernelWidth - 2 * kernelPeriod;

		__m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0 + 0);
		__m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);
		__m512 _b0_3 = _mm512_maskz_load_ps(_rightInputMask3, pInput0 + 32);

		__m512 _acc0_1 = _mm512_setzero_ps();
		__m512 _acc0_2 = _mm512_setzero_ps();
		__m512 _acc0_3 = _mm512_setzero_ps();

		__m512 _acc1_0 = _mm512_setzero_ps();
		__m512 _acc1_1 = _mm512_setzero_ps();
		__m512 _acc1_2 = _mm512_setzero_ps();

		__m512 _bf = _b0_f;
		__m512 _b0 = _b0_0;
		__m512 _b1 = _b0_1;
		__m512 _b2 = _b0_2;
		__m512 _b3 = _b0_3;

		__m512i _roll = _roll_init;

		int64_t k = 0;
		for (; k < kernelRemainder; k += DilationWidth) {
			__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
			__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);
			__m512 _aj2 = _mm512_set1_ps(Filter[--idx2]);

			_acc0_1 = _mm512_fmadd_ps(_aj2, _bf, _acc0_1);
			_acc0_2 = _mm512_fmadd_ps(_aj2, _b0, _acc0_2);
			_acc0_3 = _mm512_fmadd_ps(_aj2, _b1, _acc0_3);

			_acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
			_acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
			_acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);

			_acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
			_acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
			_acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);

			_roll = _mm512_sub_epi32(_roll, _dilation);

			_bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);
			_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
			_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
			_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
			_b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
		}

		for (; k < 16ll; k += DilationWidth) {
			__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
			__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);

			_acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
			_acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
			_acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);

			_acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
			_acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
			_acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);

			_roll = _mm512_sub_epi32(_roll, _dilation);

			_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
			_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
			_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
			_b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
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

			idx0 = KernelWidth;
			idx1 = KernelWidth - kernelPeriod;
			idx2 = KernelWidth - 2 * kernelPeriod;

			_b0_1 = _b0_2 = _b0_3 = _mm512_setzero_ps();

			_acc0_1 = _acc0_2 = _acc0_3 = _mm512_setzero_ps();

			_acc1_0 = _acc1_1 = _acc1_2 = _mm512_setzero_ps();

			_bf = _b0_f;
			_b0 = _b0_0;
			_b1 = _b0_1;
			_b2 = _b0_2;
			_b3 = _b0_3;

			_roll = _roll_init;

			for (k = 0; k < kernelRemainder; k += DilationWidth) {
				__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
				__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);
				__m512 _aj2 = _mm512_set1_ps(Filter[--idx2]);

				_acc0_1 = _mm512_fmadd_ps(_aj2, _bf, _acc0_1);
				_acc0_2 = _mm512_fmadd_ps(_aj2, _b0, _acc0_2);
				_acc0_3 = _mm512_fmadd_ps(_aj2, _b1, _acc0_3);

				_acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
				_acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
				_acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);

				_acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
				_acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
				_acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);

				_roll = _mm512_sub_epi32(_roll, _dilation);

				_bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);
				_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
				_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
				_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
				_b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
			}

			for (; k < 16ll; k += DilationWidth) {
				__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
				__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);

				_acc1_0 = _mm512_fmadd_ps(_aj1, _b0, _acc1_0);
				_acc1_1 = _mm512_fmadd_ps(_aj1, _b1, _acc1_1);
				_acc1_2 = _mm512_fmadd_ps(_aj1, _b2, _acc1_2);

				_acc0_1 = _mm512_fmadd_ps(_aj0, _b1, _acc0_1);
				_acc0_2 = _mm512_fmadd_ps(_aj0, _b2, _acc0_2);
				_acc0_3 = _mm512_fmadd_ps(_aj0, _b3, _acc0_3);

				_roll = _mm512_sub_epi32(_roll, _dilation);

				_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
				_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
				_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
				_b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
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

	const int64_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];
	const int64_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
	const int64_t PaddingLeft = Parameters->Padding[WidthShapeIndex];
	const int64_t PaddingRight = Parameters->Padding[Dimensions + WidthShapeIndex];

	const int64_t DilatedKernelWidth = (KernelWidth - 1) * DilationWidth + 1;

	assert(OutputWidth == (InputWidth - DilatedKernelWidth + PaddingLeft + PaddingRight + 1));
	MLAS_UNREFERENCED_PARAMETER(PaddingRight);

	const int64_t kernelPeriod = 16 / DilationWidth;

	const int64_t kernelRemainder = DilatedKernelWidth - 48;

	constexpr size_t widthStep = 64;

	const int64_t paddingLeftSeek = DilatedKernelWidth - PaddingLeft - 1;

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

	const __m512i _roll_init =
		_mm512_set_epi32(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16);

	const __m512i _dilation = _mm512_set1_epi32(static_cast<int>(DilationWidth));

	float* pOutput0 = Output;

	const float* pInput0 = Input;

	__m512 _b0_d = _mm512_maskz_load_ps(_leftInputMask1, pInput0 - paddingLeftShift + 0);
	__m512 _b0_e = _mm512_maskz_load_ps(_leftInputMask2, pInput0 - paddingLeftShift + 16);
	__m512 _b0_f = _mm512_maskz_load_ps(_leftInputMask3, pInput0 - paddingLeftShift + 32);
	__m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask4, pInput0 - paddingLeftShift + 48);

	pInput0 += paddingLeftSeek;

	for (int64_t i = 0; i < widthIterations; i++) {
		int64_t idx0 = KernelWidth;
		int64_t idx1 = KernelWidth - kernelPeriod;
		int64_t idx2 = KernelWidth - 2 * kernelPeriod;
		int64_t idx3 = KernelWidth - 3 * kernelPeriod;

		__m512 _b0_1 = _mm512_load_ps(pInput0);
		__m512 _b0_2 = _mm512_load_ps(pInput0 + 16);
		__m512 _b0_3 = _mm512_load_ps(pInput0 + 32);
		__m512 _b0_4 = _mm512_load_ps(pInput0 + 48);

		__m512 _acc0_1 = _mm512_load_ps(pOutput0);
		__m512 _acc0_2 = _mm512_load_ps(pOutput0 + 16);
		__m512 _acc0_3 = _mm512_load_ps(pOutput0 + 32);
		__m512 _acc0_4 = _mm512_load_ps(pOutput0 + 48);

		__m512 _acc1_0 = _mm512_setzero_ps();
		__m512 _acc1_1 = _mm512_setzero_ps();
		__m512 _acc1_2 = _mm512_setzero_ps();
		__m512 _acc1_3 = _mm512_setzero_ps();

		__m512 _be = _b0_e;
		__m512 _bf = _b0_f;
		__m512 _b0 = _b0_0;
		__m512 _b1 = _b0_1;
		__m512 _b2 = _b0_2;
		__m512 _b3 = _b0_3;
		__m512 _b4 = _b0_4;

		__m512i _roll = _roll_init;

		int64_t k = 0;
		for (; k < kernelRemainder; k += DilationWidth) {
			__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
			__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);
			__m512 _aj2 = _mm512_set1_ps(Filter[--idx2]);
			__m512 _aj3 = _mm512_set1_ps(Filter[--idx3]);

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

			_roll = _mm512_sub_epi32(_roll, _dilation);

			_be = _mm512_permutex2var_ps(_b0_d, _roll, _b0_e);
			_bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);
			_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
			_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
			_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
			_b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
			_b4 = _mm512_permutex2var_ps(_b0_3, _roll, _b0_4);
		}

		for (; k < 16ll; k += DilationWidth) {
			__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
			__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);
			__m512 _aj2 = _mm512_set1_ps(Filter[--idx2]);

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

			_roll = _mm512_sub_epi32(_roll, _dilation);

			_bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);
			_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
			_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
			_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
			_b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
			_b4 = _mm512_permutex2var_ps(_b0_3, _roll, _b0_4);
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
		int64_t idx0 = KernelWidth;
		int64_t idx1 = KernelWidth - kernelPeriod;
		int64_t idx2 = KernelWidth - 2 * kernelPeriod;
		int64_t idx3 = KernelWidth - 3 * kernelPeriod;

		__m512 _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0 + 0);
		__m512 _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 16);
		__m512 _b0_3 = _mm512_maskz_load_ps(_rightInputMask3, pInput0 + 32);
		__m512 _b0_4 = _mm512_maskz_load_ps(_rightInputMask4, pInput0 + 48);

		__m512 _acc0_1 = _mm512_setzero_ps();
		__m512 _acc0_2 = _mm512_setzero_ps();
		__m512 _acc0_3 = _mm512_setzero_ps();
		__m512 _acc0_4 = _mm512_setzero_ps();

		__m512 _acc1_0 = _mm512_setzero_ps();
		__m512 _acc1_1 = _mm512_setzero_ps();
		__m512 _acc1_2 = _mm512_setzero_ps();
		__m512 _acc1_3 = _mm512_setzero_ps();

		__m512 _be = _b0_e;
		__m512 _bf = _b0_f;
		__m512 _b0 = _b0_0;
		__m512 _b1 = _b0_1;
		__m512 _b2 = _b0_2;
		__m512 _b3 = _b0_3;
		__m512 _b4 = _b0_4;

		__m512i _roll = _roll_init;

		int64_t k = 0;
		for (; k < kernelRemainder; k += DilationWidth) {
			__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
			__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);
			__m512 _aj2 = _mm512_set1_ps(Filter[--idx2]);
			__m512 _aj3 = _mm512_set1_ps(Filter[--idx3]);

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

			_roll = _mm512_sub_epi32(_roll, _dilation);

			_be = _mm512_permutex2var_ps(_b0_d, _roll, _b0_e);
			_bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);
			_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
			_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
			_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
			_b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
			_b4 = _mm512_permutex2var_ps(_b0_3, _roll, _b0_4);
		}

		for (; k < 16ll; k += DilationWidth) {
			__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
			__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);
			__m512 _aj2 = _mm512_set1_ps(Filter[--idx2]);

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

			_roll = _mm512_sub_epi32(_roll, _dilation);

			_bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);
			_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
			_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
			_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
			_b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
			_b4 = _mm512_permutex2var_ps(_b0_3, _roll, _b0_4);
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

			idx0 = KernelWidth;
			idx1 = KernelWidth - kernelPeriod;
			idx2 = KernelWidth - 2 * kernelPeriod;
			idx2 = KernelWidth - 3 * kernelPeriod;

			_b0_1 = _b0_2 = _b0_3 = _b0_4 = _mm512_setzero_ps();

			_acc0_1 = _acc0_2 = _acc0_3 = _acc0_4 = _mm512_setzero_ps();

			_acc1_0 = _acc1_1 = _acc1_2 = _acc1_3 = _mm512_setzero_ps();

			_be = _b0_e;
			_bf = _b0_f;
			_b0 = _b0_0;
			_b1 = _b0_1;
			_b2 = _b0_2;
			_b3 = _b0_3;
			_b4 = _b0_4;

			_roll = _roll_init;

			for (k = 0; k < kernelRemainder; k += DilationWidth) {
				__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
				__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);
				__m512 _aj2 = _mm512_set1_ps(Filter[--idx2]);
				__m512 _aj3 = _mm512_set1_ps(Filter[--idx3]);

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

				_roll = _mm512_sub_epi32(_roll, _dilation);

				_be = _mm512_permutex2var_ps(_b0_d, _roll, _b0_e);
				_bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);
				_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
				_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
				_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
				_b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
				_b4 = _mm512_permutex2var_ps(_b0_3, _roll, _b0_3);
			}

			for (; k < 16ll; k += DilationWidth) {
				__m512 _aj0 = _mm512_set1_ps(Filter[--idx0]);
				__m512 _aj1 = _mm512_set1_ps(Filter[--idx1]);
				__m512 _aj2 = _mm512_set1_ps(Filter[--idx2]);

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

				_roll = _mm512_sub_epi32(_roll, _dilation);

				_bf = _mm512_permutex2var_ps(_b0_e, _roll, _b0_f);
				_b0 = _mm512_permutex2var_ps(_b0_f, _roll, _b0_0);
				_b1 = _mm512_permutex2var_ps(_b0_0, _roll, _b0_1);
				_b2 = _mm512_permutex2var_ps(_b0_1, _roll, _b0_2);
				_b3 = _mm512_permutex2var_ps(_b0_2, _roll, _b0_3);
				_b4 = _mm512_permutex2var_ps(_b0_3, _roll, _b0_3);
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
MlasConv1DSlidingKernelD8K51(const MLAS_CONV_PARAMETERS* Parameters,
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

        const int64_t DilatedWidth =
            KernelWidth + (KernelWidth - 1) * (Parameters->DilationShape[WidthShapeIndex] - 1);

        assert(KernelWidth == 51);
        assert(OutputWidth == (InputWidth - DilatedWidth + PaddingLeft + PaddingRight + 1));
        MLAS_UNREFERENCED_PARAMETER(PaddingRight);

        constexpr size_t widthStep = 400;

        const int64_t paddingLeftSeek = DilatedWidth - PaddingLeft - 1;

        const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

        const int64_t paddingRightShift = std::max<int64_t>(paddingLeftSeek - InputWidth, 0ll);

        const __mmask64 _leftInputMaska = paddingLeftShift < 64ll ? ~0ll << paddingLeftShift : 0ll;

        const __mmask16 _leftInputMask0 = static_cast<__mmask16>(_leftInputMaska);
        const __mmask16 _leftInputMask1 = static_cast<__mmask16>(_leftInputMaska >> 16);
        const __mmask16 _leftInputMask2 = static_cast<__mmask16>(_leftInputMaska >> 32);
        const __mmask16 _leftInputMask3 = static_cast<__mmask16>(_leftInputMaska >> 48);

        const __mmask64 _leftInputMaskb = ~0ll >> paddingRightShift;

        const __mmask16 _leftInputMask4 = static_cast<__mmask16>(_leftInputMaskb);
        const __mmask16 _leftInputMask5 = static_cast<__mmask16>(_leftInputMaskb >> 16);
        const __mmask16 _leftInputMask6 = static_cast<__mmask16>(_leftInputMaskb >> 32);
        const __mmask16 _leftInputMask7 = static_cast<__mmask16>(_leftInputMaskb >> 48);

        const int64_t InputWidthBase = InputWidth - paddingLeftSeek + paddingRightShift;

        const int64_t widthIterations = InputWidthBase / widthStep;
        const int64_t widthInputRemainder = InputWidthBase % widthStep;
        const int64_t widthOutputRemainder = OutputWidth % widthStep;

        const __mmask64 _rightInputMaska =
            widthInputRemainder < 64ll ? ~(~0ll << widthInputRemainder) : ~0ll;

        const __mmask16 _rightInputMask0 = static_cast<__mmask16>(_rightInputMaska);
        const __mmask16 _rightInputMask1 = static_cast<__mmask16>(_rightInputMaska >> 16);
        const __mmask16 _rightInputMask2 = static_cast<__mmask16>(_rightInputMaska >> 32);
        const __mmask16 _rightInputMask3 = static_cast<__mmask16>(_rightInputMaska >> 48);

        const __mmask64 _rightInputMaskb =
            ~(0xffffffffffffffffll << std::max<int64_t>(widthInputRemainder - 64ll, 0ll));

        const __mmask16 _rightInputMask4 = static_cast<__mmask16>(_rightInputMaskb);
        const __mmask16 _rightInputMask5 = static_cast<__mmask16>(_rightInputMaskb >> 16);
        const __mmask16 _rightInputMask6 = static_cast<__mmask16>(_rightInputMaskb >> 32);
        const __mmask16 _rightInputMask7 = static_cast<__mmask16>(_rightInputMaskb >> 48);

        const __mmask64 _rightOutputMaska =
            ~(0xffffffffffffffffll << std::min<int64_t>(widthOutputRemainder, 64ll));

        const __mmask16 _rightOutputMask0 = static_cast<__mmask16>(_rightOutputMaska);
        const __mmask16 _rightOutputMask1 = static_cast<__mmask16>(_rightOutputMaska >> 16);
        const __mmask16 _rightOutputMask2 = static_cast<__mmask16>(_rightOutputMaska >> 32);
        const __mmask16 _rightOutputMask3 = static_cast<__mmask16>(_rightOutputMaska >> 48);

        const __mmask64 _rightOutputMaskb =
            ~(0xffffffffffffffffll << std::max<int64_t>(widthOutputRemainder - 64ll, 0ll));

        const __mmask16 _rightOutputMask4 = static_cast<__mmask16>(_rightOutputMaskb);
        const __mmask16 _rightOutputMask5 = static_cast<__mmask16>(_rightOutputMaskb >> 16);
        const __mmask16 _rightOutputMask6 = static_cast<__mmask16>(_rightOutputMaskb >> 32);
        const __mmask16 _rightOutputMask7 = static_cast<__mmask16>(_rightOutputMaskb >> 48);

        const __m512 _a0 = _mm512_set1_ps(Filter[0]);
        const __m512 _a1 = _mm512_set1_ps(Filter[1]);
        const __m512 _a2 = _mm512_set1_ps(Filter[2]);
        const __m512 _a3 = _mm512_set1_ps(Filter[3]);
        const __m512 _a4 = _mm512_set1_ps(Filter[4]);
        const __m512 _a5 = _mm512_set1_ps(Filter[5]);
        const __m512 _a6 = _mm512_set1_ps(Filter[6]);
        const __m512 _a7 = _mm512_set1_ps(Filter[7]);
        const __m512 _a8 = _mm512_set1_ps(Filter[8]);
        const __m512 _a9 = _mm512_set1_ps(Filter[9]);
        const __m512 _a10 = _mm512_set1_ps(Filter[10]);
        const __m512 _a11 = _mm512_set1_ps(Filter[11]);
        const __m512 _a12 = _mm512_set1_ps(Filter[12]);
        const __m512 _a13 = _mm512_set1_ps(Filter[13]);
        const __m512 _a14 = _mm512_set1_ps(Filter[14]);
        const __m512 _a15 = _mm512_set1_ps(Filter[15]);
        const __m512 _a16 = _mm512_set1_ps(Filter[16]);
        const __m512 _a17 = _mm512_set1_ps(Filter[17]);
        const __m512 _a18 = _mm512_set1_ps(Filter[18]);
        const __m512 _a19 = _mm512_set1_ps(Filter[19]);
        const __m512 _a20 = _mm512_set1_ps(Filter[20]);
        const __m512 _a21 = _mm512_set1_ps(Filter[21]);
        const __m512 _a22 = _mm512_set1_ps(Filter[22]);
        const __m512 _a23 = _mm512_set1_ps(Filter[23]);
        const __m512 _a24 = _mm512_set1_ps(Filter[24]);
        const __m512 _a25 = _mm512_set1_ps(Filter[25]);
        const __m512 _a26 = _mm512_set1_ps(Filter[26]);
        const __m512 _a27 = _mm512_set1_ps(Filter[27]);
        const __m512 _a28 = _mm512_set1_ps(Filter[28]);
        const __m512 _a29 = _mm512_set1_ps(Filter[29]);
        const __m512 _a30 = _mm512_set1_ps(Filter[30]);
        const __m512 _a31 = _mm512_set1_ps(Filter[31]);
        const __m512 _a32 = _mm512_set1_ps(Filter[32]);
        const __m512 _a33 = _mm512_set1_ps(Filter[33]);
        const __m512 _a34 = _mm512_set1_ps(Filter[34]);
        const __m512 _a35 = _mm512_set1_ps(Filter[35]);
        const __m512 _a36 = _mm512_set1_ps(Filter[36]);
        const __m512 _a37 = _mm512_set1_ps(Filter[37]);
        const __m512 _a38 = _mm512_set1_ps(Filter[38]);
        const __m512 _a39 = _mm512_set1_ps(Filter[39]);
        const __m512 _a40 = _mm512_set1_ps(Filter[40]);
        const __m512 _a41 = _mm512_set1_ps(Filter[41]);
        const __m512 _a42 = _mm512_set1_ps(Filter[42]);
        const __m512 _a43 = _mm512_set1_ps(Filter[43]);
        const __m512 _a44 = _mm512_set1_ps(Filter[44]);
        const __m512 _a45 = _mm512_set1_ps(Filter[45]);
        const __m512 _a46 = _mm512_set1_ps(Filter[46]);
        const __m512 _a47 = _mm512_set1_ps(Filter[47]);
        const __m512 _a48 = _mm512_set1_ps(Filter[48]);
        const __m512 _a49 = _mm512_set1_ps(Filter[49]);
        const __m512 _a50 = _mm512_set1_ps(Filter[50]);

        float* pOutput0 = Output;

        const float* pInput0 = Input - paddingLeftShift;

        __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask0, pInput0 + 0);
        __m512 _b0_1 = _mm512_maskz_load_ps(_leftInputMask1, pInput0 + 16);
        __m512 _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

        __m512 _b0_2 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 + 32);
        __m512 _b1_2 = _mm512_shuffle_f32x4(_b0_1, _b0_2, 0x4e);

        __m512 _b0_3 = _mm512_maskz_load_ps(_leftInputMask3, pInput0 + 48);
        __m512 _b1_3 = _mm512_shuffle_f32x4(_b0_2, _b0_3, 0x4e);

        __m512 _b0_4 = _mm512_maskz_load_ps(_leftInputMask4, pInput0 + 64);
        __m512 _b1_4 = _mm512_shuffle_f32x4(_b0_3, _b0_4, 0x4e);

        __m512 _b0_5 = _mm512_maskz_load_ps(_leftInputMask5, pInput0 + 80);
        __m512 _b1_5 = _mm512_shuffle_f32x4(_b0_4, _b0_5, 0x4e);

        __m512 _b0_6 = _mm512_maskz_load_ps(_leftInputMask6, pInput0 + 96);
        __m512 _b1_6 = _mm512_shuffle_f32x4(_b0_5, _b0_6, 0x4e);

        __m512 _b0_7 = _mm512_maskz_load_ps(_leftInputMask7, pInput0 + 112);
        __m512 _b1_7 = _mm512_shuffle_f32x4(_b0_6, _b0_7, 0x4e);

        __m512 _b0_8 = _mm512_maskz_load_ps(_leftInputMask6, pInput0 + 128);
        __m512 _b1_8 = _mm512_shuffle_f32x4(_b0_7, _b0_8, 0x4e);

        __m512 _b0_9 = _mm512_maskz_load_ps(_leftInputMask7, pInput0 + 144);
        __m512 _b1_9 = _mm512_shuffle_f32x4(_b0_8, _b0_9, 0x4e);

        __m512 _b0_10 = _mm512_maskz_load_ps(_leftInputMask0, pInput0 + 160);
        __m512 _b1_10 = _mm512_shuffle_f32x4(_b0_9, _b0_10, 0x4e);

        __m512 _b0_11 = _mm512_maskz_load_ps(_leftInputMask1, pInput0 + 176);
        __m512 _b1_11 = _mm512_shuffle_f32x4(_b0_10, _b0_11, 0x4e);

        __m512 _b0_12 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 + 192);
        __m512 _b1_12 = _mm512_shuffle_f32x4(_b0_11, _b0_12, 0x4e);

        __m512 _b0_13 = _mm512_maskz_load_ps(_leftInputMask3, pInput0 + 208);
        __m512 _b1_13 = _mm512_shuffle_f32x4(_b0_12, _b0_13, 0x4e);

        __m512 _b0_14 = _mm512_maskz_load_ps(_leftInputMask4, pInput0 + 224);
        __m512 _b1_14 = _mm512_shuffle_f32x4(_b0_13, _b0_14, 0x4e);

        __m512 _b0_15 = _mm512_maskz_load_ps(_leftInputMask5, pInput0 + 240);
        __m512 _b1_15 = _mm512_shuffle_f32x4(_b0_14, _b0_15, 0x4e);

        __m512 _b0_16 = _mm512_maskz_load_ps(_leftInputMask6, pInput0 + 256);
        __m512 _b1_16 = _mm512_shuffle_f32x4(_b0_15, _b0_16, 0x4e);

        __m512 _b0_17 = _mm512_maskz_load_ps(_leftInputMask7, pInput0 + 272);
        __m512 _b1_17 = _mm512_shuffle_f32x4(_b0_16, _b0_17, 0x4e);

        __m512 _b0_18 = _mm512_maskz_load_ps(_leftInputMask6, pInput0 + 288);
        __m512 _b1_18 = _mm512_shuffle_f32x4(_b0_17, _b0_18, 0x4e);

        __m512 _b0_19 = _mm512_maskz_load_ps(_leftInputMask7, pInput0 + 304);
        __m512 _b1_19 = _mm512_shuffle_f32x4(_b0_18, _b0_19, 0x4e);

        __m512 _b0_20 = _mm512_maskz_load_ps(_leftInputMask0, pInput0 + 320);
        __m512 _b1_20 = _mm512_shuffle_f32x4(_b0_9, _b0_20, 0x4e);

        __m512 _b0_21 = _mm512_maskz_load_ps(_leftInputMask1, pInput0 + 336);
        __m512 _b1_21 = _mm512_shuffle_f32x4(_b0_20, _b0_21, 0x4e);

        __m512 _b0_22 = _mm512_maskz_load_ps(_leftInputMask2, pInput0 + 352);
        __m512 _b1_22 = _mm512_shuffle_f32x4(_b0_21, _b0_22, 0x4e);

        __m512 _b0_23 = _mm512_maskz_load_ps(_leftInputMask3, pInput0 + 368);
        __m512 _b1_23 = _mm512_shuffle_f32x4(_b0_22, _b0_23, 0x4e);

        __m512 _b0_24 = _mm512_maskz_load_ps(_leftInputMask4, pInput0 + 384);
        __m512 _b1_24 = _mm512_shuffle_f32x4(_b0_23, _b0_24, 0x4e);

        pInput0 += widthStep;

        for (int64_t i = 0; i < widthIterations; i++) {
                __m512 _acc0 = _mm512_load_ps(pOutput0);
                __m512 _acc1 = _mm512_load_ps(pOutput0 + 16);
                __m512 _acc2 = _mm512_load_ps(pOutput0 + 32);
                __m512 _acc3 = _mm512_load_ps(pOutput0 + 48);
                __m512 _acc4 = _mm512_load_ps(pOutput0 + 64);
                __m512 _acc5 = _mm512_load_ps(pOutput0 + 80);
                __m512 _acc6 = _mm512_load_ps(pOutput0 + 96);
                __m512 _acc7 = _mm512_load_ps(pOutput0 + 112);
                __m512 _acc8 = _mm512_load_ps(pOutput0 + 128);
                __m512 _acc9 = _mm512_load_ps(pOutput0 + 144);
                __m512 _acc10 = _mm512_load_ps(pOutput0 + 160);
                __m512 _acc11 = _mm512_load_ps(pOutput0 + 176);
                __m512 _acc12 = _mm512_load_ps(pOutput0 + 192);
                __m512 _acc13 = _mm512_load_ps(pOutput0 + 208);
                __m512 _acc14 = _mm512_load_ps(pOutput0 + 224);
                __m512 _acc15 = _mm512_load_ps(pOutput0 + 240);
                __m512 _acc16 = _mm512_load_ps(pOutput0 + 256);
                __m512 _acc17 = _mm512_load_ps(pOutput0 + 272);
                __m512 _acc18 = _mm512_load_ps(pOutput0 + 288);
                __m512 _acc19 = _mm512_load_ps(pOutput0 + 304);
                __m512 _acc20 = _mm512_load_ps(pOutput0 + 320);
                __m512 _acc21 = _mm512_load_ps(pOutput0 + 336);
                __m512 _acc22 = _mm512_load_ps(pOutput0 + 352);
                __m512 _acc23 = _mm512_load_ps(pOutput0 + 368);
                __m512 _acc24 = _mm512_load_ps(pOutput0 + 384);

                _acc0 = _mm512_fmadd_ps(_a0, _b0_0, _acc0);
                _acc1 = _mm512_fmadd_ps(_a0, _b0_1, _acc1);
                _acc2 = _mm512_fmadd_ps(_a0, _b0_2, _acc2);
                _acc3 = _mm512_fmadd_ps(_a0, _b0_3, _acc3);
                _acc4 = _mm512_fmadd_ps(_a0, _b0_4, _acc4);
                _acc5 = _mm512_fmadd_ps(_a0, _b0_5, _acc5);
                _acc6 = _mm512_fmadd_ps(_a0, _b0_6, _acc6);
                _acc7 = _mm512_fmadd_ps(_a0, _b0_7, _acc7);
                _acc8 = _mm512_fmadd_ps(_a0, _b0_8, _acc8);
                _acc9 = _mm512_fmadd_ps(_a0, _b0_9, _acc9);
                _acc10 = _mm512_fmadd_ps(_a0, _b0_10, _acc10);
                _acc11 = _mm512_fmadd_ps(_a0, _b0_11, _acc11);
                _acc12 = _mm512_fmadd_ps(_a0, _b0_12, _acc12);
                _acc13 = _mm512_fmadd_ps(_a0, _b0_13, _acc13);
                _acc14 = _mm512_fmadd_ps(_a0, _b0_14, _acc14);
                _acc15 = _mm512_fmadd_ps(_a0, _b0_15, _acc15);
                _acc16 = _mm512_fmadd_ps(_a0, _b0_16, _acc16);
                _acc17 = _mm512_fmadd_ps(_a0, _b0_17, _acc17);
                _acc18 = _mm512_fmadd_ps(_a0, _b0_18, _acc18);
                _acc19 = _mm512_fmadd_ps(_a0, _b0_19, _acc19);
                _acc20 = _mm512_fmadd_ps(_a0, _b0_20, _acc20);
                _acc21 = _mm512_fmadd_ps(_a0, _b0_21, _acc21);
                _acc22 = _mm512_fmadd_ps(_a0, _b0_22, _acc22);
                _acc23 = _mm512_fmadd_ps(_a0, _b0_23, _acc23);
                _acc24 = _mm512_fmadd_ps(_a0, _b0_24, _acc24);

                _b0_0 = _mm512_load_ps(pInput0);

                _acc0 = _mm512_fmadd_ps(_a2, _b0_1, _acc0);
                _acc1 = _mm512_fmadd_ps(_a2, _b0_2, _acc1);
                _acc2 = _mm512_fmadd_ps(_a2, _b0_3, _acc2);
                _acc3 = _mm512_fmadd_ps(_a2, _b0_4, _acc3);
                _acc4 = _mm512_fmadd_ps(_a2, _b0_5, _acc4);
                _acc5 = _mm512_fmadd_ps(_a2, _b0_6, _acc5);
                _acc6 = _mm512_fmadd_ps(_a2, _b0_7, _acc6);
                _acc7 = _mm512_fmadd_ps(_a2, _b0_8, _acc7);
                _acc8 = _mm512_fmadd_ps(_a2, _b0_9, _acc8);
                _acc9 = _mm512_fmadd_ps(_a2, _b0_10, _acc9);
                _acc10 = _mm512_fmadd_ps(_a2, _b0_11, _acc10);
                _acc11 = _mm512_fmadd_ps(_a2, _b0_12, _acc11);
                _acc12 = _mm512_fmadd_ps(_a2, _b0_13, _acc12);
                _acc13 = _mm512_fmadd_ps(_a2, _b0_14, _acc13);
                _acc14 = _mm512_fmadd_ps(_a2, _b0_15, _acc14);
                _acc15 = _mm512_fmadd_ps(_a2, _b0_16, _acc15);
                _acc16 = _mm512_fmadd_ps(_a2, _b0_17, _acc16);
                _acc17 = _mm512_fmadd_ps(_a2, _b0_18, _acc17);
                _acc18 = _mm512_fmadd_ps(_a2, _b0_19, _acc18);
                _acc19 = _mm512_fmadd_ps(_a2, _b0_20, _acc19);
                _acc20 = _mm512_fmadd_ps(_a2, _b0_21, _acc20);
                _acc21 = _mm512_fmadd_ps(_a2, _b0_22, _acc21);
                _acc22 = _mm512_fmadd_ps(_a2, _b0_23, _acc22);
                _acc23 = _mm512_fmadd_ps(_a2, _b0_24, _acc23);
                _acc24 = _mm512_fmadd_ps(_a2, _b0_0, _acc24);

                __m512 _b1_0 = _mm512_shuffle_f32x4(_b0_7, _b0_0, 0x4e);

                _acc0 = _mm512_fmadd_ps(_a1, _b1_1, _acc0);
                _acc1 = _mm512_fmadd_ps(_a1, _b1_2, _acc1);
                _acc2 = _mm512_fmadd_ps(_a1, _b1_3, _acc2);
                _acc3 = _mm512_fmadd_ps(_a1, _b1_4, _acc3);
                _acc4 = _mm512_fmadd_ps(_a1, _b1_5, _acc4);
                _acc5 = _mm512_fmadd_ps(_a1, _b1_6, _acc5);
                _acc6 = _mm512_fmadd_ps(_a1, _b1_7, _acc6);
                _acc7 = _mm512_fmadd_ps(_a1, _b1_8, _acc7);
                _acc8 = _mm512_fmadd_ps(_a1, _b1_9, _acc8);
                _acc9 = _mm512_fmadd_ps(_a1, _b1_10, _acc9);
                _acc10 = _mm512_fmadd_ps(_a1, _b1_11, _acc10);
                _acc11 = _mm512_fmadd_ps(_a1, _b1_12, _acc11);
                _acc12 = _mm512_fmadd_ps(_a1, _b1_13, _acc12);
                _acc13 = _mm512_fmadd_ps(_a1, _b1_14, _acc13);
                _acc14 = _mm512_fmadd_ps(_a1, _b1_15, _acc14);
                _acc15 = _mm512_fmadd_ps(_a1, _b1_16, _acc15);
                _acc16 = _mm512_fmadd_ps(_a1, _b1_17, _acc16);
                _acc17 = _mm512_fmadd_ps(_a1, _b1_18, _acc17);
                _acc18 = _mm512_fmadd_ps(_a1, _b1_19, _acc18);
                _acc19 = _mm512_fmadd_ps(_a1, _b1_20, _acc19);
                _acc20 = _mm512_fmadd_ps(_a1, _b1_21, _acc20);
                _acc21 = _mm512_fmadd_ps(_a1, _b1_22, _acc21);
                _acc22 = _mm512_fmadd_ps(_a1, _b1_23, _acc22);
                _acc23 = _mm512_fmadd_ps(_a1, _b1_24, _acc23);
                _acc24 = _mm512_fmadd_ps(_a1, _b1_0, _acc24);

                _b0_1 = _mm512_load_ps(pInput0 + 16);

                _acc0 = _mm512_fmadd_ps(_a4, _b0_2, _acc0);
                _acc1 = _mm512_fmadd_ps(_a4, _b0_3, _acc1);
                _acc2 = _mm512_fmadd_ps(_a4, _b0_4, _acc2);
                _acc3 = _mm512_fmadd_ps(_a4, _b0_5, _acc3);
                _acc4 = _mm512_fmadd_ps(_a4, _b0_6, _acc4);
                _acc5 = _mm512_fmadd_ps(_a4, _b0_7, _acc5);
                _acc6 = _mm512_fmadd_ps(_a4, _b0_8, _acc6);
                _acc7 = _mm512_fmadd_ps(_a4, _b0_9, _acc7);
                _acc8 = _mm512_fmadd_ps(_a4, _b0_10, _acc8);
                _acc9 = _mm512_fmadd_ps(_a4, _b0_11, _acc9);
                _acc10 = _mm512_fmadd_ps(_a4, _b0_12, _acc10);
                _acc11 = _mm512_fmadd_ps(_a4, _b0_13, _acc11);
                _acc12 = _mm512_fmadd_ps(_a4, _b0_14, _acc12);
                _acc13 = _mm512_fmadd_ps(_a4, _b0_15, _acc13);
                _acc14 = _mm512_fmadd_ps(_a4, _b0_16, _acc14);
                _acc15 = _mm512_fmadd_ps(_a4, _b0_17, _acc15);
                _acc16 = _mm512_fmadd_ps(_a4, _b0_18, _acc16);
                _acc17 = _mm512_fmadd_ps(_a4, _b0_19, _acc17);
                _acc18 = _mm512_fmadd_ps(_a4, _b0_20, _acc18);
                _acc19 = _mm512_fmadd_ps(_a4, _b0_21, _acc19);
                _acc20 = _mm512_fmadd_ps(_a4, _b0_22, _acc20);
                _acc21 = _mm512_fmadd_ps(_a4, _b0_23, _acc21);
                _acc22 = _mm512_fmadd_ps(_a4, _b0_24, _acc22);
                _acc23 = _mm512_fmadd_ps(_a4, _b0_0, _acc23);
                _acc24 = _mm512_fmadd_ps(_a4, _b0_1, _acc24);

                _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

                _acc0 = _mm512_fmadd_ps(_a3, _b1_2, _acc0);
                _acc1 = _mm512_fmadd_ps(_a3, _b1_3, _acc1);
                _acc2 = _mm512_fmadd_ps(_a3, _b1_4, _acc2);
                _acc3 = _mm512_fmadd_ps(_a3, _b1_5, _acc3);
                _acc4 = _mm512_fmadd_ps(_a3, _b1_6, _acc4);
                _acc5 = _mm512_fmadd_ps(_a3, _b1_7, _acc5);
                _acc6 = _mm512_fmadd_ps(_a3, _b1_8, _acc6);
                _acc7 = _mm512_fmadd_ps(_a3, _b1_9, _acc7);
                _acc8 = _mm512_fmadd_ps(_a3, _b1_10, _acc8);
                _acc9 = _mm512_fmadd_ps(_a3, _b1_11, _acc9);
                _acc10 = _mm512_fmadd_ps(_a3, _b1_12, _acc10);
                _acc11 = _mm512_fmadd_ps(_a3, _b1_13, _acc11);
                _acc12 = _mm512_fmadd_ps(_a3, _b1_14, _acc12);
                _acc13 = _mm512_fmadd_ps(_a3, _b1_15, _acc13);
                _acc14 = _mm512_fmadd_ps(_a3, _b1_16, _acc14);
                _acc15 = _mm512_fmadd_ps(_a3, _b1_17, _acc15);
                _acc16 = _mm512_fmadd_ps(_a3, _b1_18, _acc16);
                _acc17 = _mm512_fmadd_ps(_a3, _b1_19, _acc17);
                _acc18 = _mm512_fmadd_ps(_a3, _b1_20, _acc18);
                _acc19 = _mm512_fmadd_ps(_a3, _b1_21, _acc19);
                _acc20 = _mm512_fmadd_ps(_a3, _b1_22, _acc20);
                _acc21 = _mm512_fmadd_ps(_a3, _b1_23, _acc21);
                _acc22 = _mm512_fmadd_ps(_a3, _b1_24, _acc22);
                _acc23 = _mm512_fmadd_ps(_a3, _b1_0, _acc23);
                _acc24 = _mm512_fmadd_ps(_a3, _b1_1, _acc24);

                _b0_2 = _mm512_load_ps(pInput0 + 32);

                _acc0 = _mm512_fmadd_ps(_a6, _b0_3, _acc0);
                _acc1 = _mm512_fmadd_ps(_a6, _b0_4, _acc1);
                _acc2 = _mm512_fmadd_ps(_a6, _b0_5, _acc2);
                _acc3 = _mm512_fmadd_ps(_a6, _b0_6, _acc3);
                _acc4 = _mm512_fmadd_ps(_a6, _b0_7, _acc4);
                _acc5 = _mm512_fmadd_ps(_a6, _b0_0, _acc5);
                _acc6 = _mm512_fmadd_ps(_a6, _b0_1, _acc6);
                _acc7 = _mm512_fmadd_ps(_a6, _b0_2, _acc7);

                _b1_2 = _mm512_shuffle_f32x4(_b0_1, _b0_2, 0x4e);

                _acc0 = _mm512_fmadd_ps(_a5, _b1_3, _acc0);
                _acc1 = _mm512_fmadd_ps(_a5, _b1_4, _acc1);
                _acc2 = _mm512_fmadd_ps(_a5, _b1_5, _acc2);
                _acc3 = _mm512_fmadd_ps(_a5, _b1_6, _acc3);
                _acc4 = _mm512_fmadd_ps(_a5, _b1_7, _acc4);
                _acc5 = _mm512_fmadd_ps(_a5, _b1_0, _acc5);
                _acc6 = _mm512_fmadd_ps(_a5, _b1_1, _acc6);
                _acc7 = _mm512_fmadd_ps(_a5, _b1_2, _acc7);

                _b0_3 = _mm512_load_ps(pInput0 + 48);

                _acc0 = _mm512_fmadd_ps(_a8, _b0_4, _acc0);
                _acc1 = _mm512_fmadd_ps(_a8, _b0_5, _acc1);
                _acc2 = _mm512_fmadd_ps(_a8, _b0_6, _acc2);
                _acc3 = _mm512_fmadd_ps(_a8, _b0_7, _acc3);
                _acc4 = _mm512_fmadd_ps(_a8, _b0_0, _acc4);
                _acc5 = _mm512_fmadd_ps(_a8, _b0_1, _acc5);
                _acc6 = _mm512_fmadd_ps(_a8, _b0_2, _acc6);
                _acc7 = _mm512_fmadd_ps(_a8, _b0_3, _acc7);

                _b1_3 = _mm512_shuffle_f32x4(_b0_2, _b0_3, 0x4e);

                _acc0 = _mm512_fmadd_ps(_a7, _b1_4, _acc0);
                _acc1 = _mm512_fmadd_ps(_a7, _b1_5, _acc1);
                _acc2 = _mm512_fmadd_ps(_a7, _b1_6, _acc2);
                _acc3 = _mm512_fmadd_ps(_a7, _b1_7, _acc3);
                _acc4 = _mm512_fmadd_ps(_a7, _b1_0, _acc4);
                _acc5 = _mm512_fmadd_ps(_a7, _b1_1, _acc5);
                _acc6 = _mm512_fmadd_ps(_a7, _b1_2, _acc6);
                _acc7 = _mm512_fmadd_ps(_a7, _b1_3, _acc7);

                _b0_4 = _mm512_load_ps(pInput0 + 64);

                _acc0 = _mm512_fmadd_ps(_a10, _b0_5, _acc0);
                _acc1 = _mm512_fmadd_ps(_a10, _b0_6, _acc1);
                _acc2 = _mm512_fmadd_ps(_a10, _b0_7, _acc2);
                _acc3 = _mm512_fmadd_ps(_a10, _b0_0, _acc3);
                _acc4 = _mm512_fmadd_ps(_a10, _b0_1, _acc4);
                _acc5 = _mm512_fmadd_ps(_a10, _b0_2, _acc5);
                _acc6 = _mm512_fmadd_ps(_a10, _b0_3, _acc6);
                _acc7 = _mm512_fmadd_ps(_a10, _b0_4, _acc7);

                _b1_4 = _mm512_shuffle_f32x4(_b0_3, _b0_4, 0x4e);

                _acc0 = _mm512_fmadd_ps(_a9, _b1_5, _acc0);
                _acc1 = _mm512_fmadd_ps(_a9, _b1_6, _acc1);
                _acc2 = _mm512_fmadd_ps(_a9, _b1_7, _acc2);
                _acc3 = _mm512_fmadd_ps(_a9, _b1_0, _acc3);
                _acc4 = _mm512_fmadd_ps(_a9, _b1_1, _acc4);
                _acc5 = _mm512_fmadd_ps(_a9, _b1_2, _acc5);
                _acc6 = _mm512_fmadd_ps(_a9, _b1_3, _acc6);
                _acc7 = _mm512_fmadd_ps(_a9, _b1_4, _acc7);

                _b0_5 = _mm512_load_ps(pInput0 + 80);

                _acc0 = _mm512_fmadd_ps(_a12, _b0_6, _acc0);
                _acc1 = _mm512_fmadd_ps(_a12, _b0_7, _acc1);
                _acc2 = _mm512_fmadd_ps(_a12, _b0_0, _acc2);
                _acc3 = _mm512_fmadd_ps(_a12, _b0_1, _acc3);
                _acc4 = _mm512_fmadd_ps(_a12, _b0_2, _acc4);
                _acc5 = _mm512_fmadd_ps(_a12, _b0_3, _acc5);
                _acc6 = _mm512_fmadd_ps(_a12, _b0_4, _acc6);
                _acc7 = _mm512_fmadd_ps(_a12, _b0_5, _acc7);

                _b1_5 = _mm512_shuffle_f32x4(_b0_4, _b0_5, 0x4e);

                _acc0 = _mm512_fmadd_ps(_a11, _b1_6, _acc0);
                _acc1 = _mm512_fmadd_ps(_a11, _b1_7, _acc1);
                _acc2 = _mm512_fmadd_ps(_a11, _b1_0, _acc2);
                _acc3 = _mm512_fmadd_ps(_a11, _b1_1, _acc3);
                _acc4 = _mm512_fmadd_ps(_a11, _b1_2, _acc4);
                _acc5 = _mm512_fmadd_ps(_a11, _b1_3, _acc5);
                _acc6 = _mm512_fmadd_ps(_a11, _b1_4, _acc6);
                _acc7 = _mm512_fmadd_ps(_a11, _b1_5, _acc7);

                _b0_6 = _mm512_load_ps(pInput0 + 96);

                _acc0 = _mm512_fmadd_ps(_a14, _b0_7, _acc0);
                _acc1 = _mm512_fmadd_ps(_a14, _b0_0, _acc1);
                _acc2 = _mm512_fmadd_ps(_a14, _b0_1, _acc2);
                _acc3 = _mm512_fmadd_ps(_a14, _b0_2, _acc3);
                _acc4 = _mm512_fmadd_ps(_a14, _b0_3, _acc4);
                _acc5 = _mm512_fmadd_ps(_a14, _b0_4, _acc5);
                _acc6 = _mm512_fmadd_ps(_a14, _b0_5, _acc6);
                _acc7 = _mm512_fmadd_ps(_a14, _b0_6, _acc7);

                _b1_6 = _mm512_shuffle_f32x4(_b0_5, _b0_6, 0x4e);

                _acc0 = _mm512_fmadd_ps(_a13, _b1_7, _acc0);
                _acc1 = _mm512_fmadd_ps(_a13, _b1_0, _acc1);
                _acc2 = _mm512_fmadd_ps(_a13, _b1_1, _acc2);
                _acc3 = _mm512_fmadd_ps(_a13, _b1_2, _acc3);
                _acc4 = _mm512_fmadd_ps(_a13, _b1_3, _acc4);
                _acc5 = _mm512_fmadd_ps(_a13, _b1_4, _acc5);
                _acc6 = _mm512_fmadd_ps(_a13, _b1_5, _acc6);
                _acc7 = _mm512_fmadd_ps(_a13, _b1_6, _acc7);

                _b0_7 = _mm512_load_ps(pInput0 + 112);

                _acc0 = _mm512_fmadd_ps(_a16, _b0_0, _acc0);
                _acc1 = _mm512_fmadd_ps(_a16, _b0_1, _acc1);
                _acc2 = _mm512_fmadd_ps(_a16, _b0_2, _acc2);
                _acc3 = _mm512_fmadd_ps(_a16, _b0_3, _acc3);
                _acc4 = _mm512_fmadd_ps(_a16, _b0_4, _acc4);
                _acc5 = _mm512_fmadd_ps(_a16, _b0_5, _acc5);
                _acc6 = _mm512_fmadd_ps(_a16, _b0_6, _acc6);
                _acc7 = _mm512_fmadd_ps(_a16, _b0_7, _acc7);

                _b1_7 = _mm512_shuffle_f32x4(_b0_6, _b0_7, 0x4e);

                _acc0 = _mm512_fmadd_ps(_a15, _b1_0, _acc0);
                _acc1 = _mm512_fmadd_ps(_a15, _b1_1, _acc1);
                _acc2 = _mm512_fmadd_ps(_a15, _b1_2, _acc2);
                _acc3 = _mm512_fmadd_ps(_a15, _b1_3, _acc3);
                _acc4 = _mm512_fmadd_ps(_a15, _b1_4, _acc4);
                _acc5 = _mm512_fmadd_ps(_a15, _b1_5, _acc5);
                _acc6 = _mm512_fmadd_ps(_a15, _b1_6, _acc6);
                _acc7 = _mm512_fmadd_ps(_a15, _b1_7, _acc7);

                // Store
                _mm512_store_ps(pOutput0 + 0, _acc0);
                _mm512_store_ps(pOutput0 + 16, _acc1);
                _mm512_store_ps(pOutput0 + 32, _acc2);
                _mm512_store_ps(pOutput0 + 48, _acc3);
                _mm512_store_ps(pOutput0 + 64, _acc4);
                _mm512_store_ps(pOutput0 + 80, _acc5);
                _mm512_store_ps(pOutput0 + 96, _acc6);
                _mm512_store_ps(pOutput0 + 112, _acc7);

                pInput0 += widthStep;
                pOutput0 += widthStep;
        }

        // Right Edge
        {
                __m512 _acc0 = _mm512_mul_ps(_a0, _b0_0);
                __m512 _acc1 = _mm512_mul_ps(_a0, _b0_1);
                __m512 _acc2 = _mm512_mul_ps(_a0, _b0_2);
                __m512 _acc3 = _mm512_mul_ps(_a0, _b0_3);
                __m512 _acc4 = _mm512_mul_ps(_a0, _b0_4);
                __m512 _acc5 = _mm512_mul_ps(_a0, _b0_5);
                __m512 _acc6 = _mm512_mul_ps(_a0, _b0_6);
                __m512 _acc7 = _mm512_mul_ps(_a0, _b0_7);

                _b0_0 = _mm512_maskz_load_ps(_rightInputMask0, pInput0);

                __m512 _aj2 = _mm512_set1_ps(Filter[2]);

                _acc0 = _mm512_fmadd_ps(_aj2, _b0_1, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj2, _b0_2, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj2, _b0_3, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj2, _b0_4, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj2, _b0_5, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj2, _b0_6, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj2, _b0_7, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj2, _b0_0, _acc7);

                __m512 _b1_0 = _mm512_shuffle_f32x4(_b0_7, _b0_0, 0x4e);

                __m512 _aj1 = _mm512_set1_ps(Filter[1]);

                _acc0 = _mm512_fmadd_ps(_aj1, _b1_1, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj1, _b1_2, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj1, _b1_3, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj1, _b1_4, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj1, _b1_5, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj1, _b1_6, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj1, _b1_7, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj1, _b1_0, _acc7);

                _b0_1 = _mm512_maskz_load_ps(_rightInputMask1, pInput0 + 16);

                _aj2 = _mm512_set1_ps(Filter[4]);

                _acc0 = _mm512_fmadd_ps(_aj2, _b0_2, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj2, _b0_3, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj2, _b0_4, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj2, _b0_5, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj2, _b0_6, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj2, _b0_7, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj2, _b0_0, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj2, _b0_1, _acc7);

                _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

                _aj1 = _mm512_set1_ps(Filter[3]);

                _acc0 = _mm512_fmadd_ps(_aj1, _b1_2, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj1, _b1_3, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj1, _b1_4, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj1, _b1_5, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj1, _b1_6, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj1, _b1_7, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj1, _b1_0, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj1, _b1_1, _acc7);

                _b0_2 = _mm512_maskz_load_ps(_rightInputMask2, pInput0 + 32);

                _aj2 = _mm512_set1_ps(Filter[6]);

                _acc0 = _mm512_fmadd_ps(_aj2, _b0_3, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj2, _b0_4, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj2, _b0_5, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj2, _b0_6, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj2, _b0_7, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj2, _b0_0, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj2, _b0_1, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj2, _b0_2, _acc7);

                _b1_2 = _mm512_shuffle_f32x4(_b0_1, _b0_2, 0x4e);

                _aj1 = _mm512_set1_ps(Filter[5]);

                _acc0 = _mm512_fmadd_ps(_aj1, _b1_3, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj1, _b1_4, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj1, _b1_5, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj1, _b1_6, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj1, _b1_7, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj1, _b1_0, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj1, _b1_1, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj1, _b1_2, _acc7);

                _b0_3 = _mm512_maskz_load_ps(_rightInputMask3, pInput0 + 48);

                _aj2 = _mm512_set1_ps(Filter[8]);

                _acc0 = _mm512_fmadd_ps(_aj2, _b0_4, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj2, _b0_5, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj2, _b0_6, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj2, _b0_7, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj2, _b0_0, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj2, _b0_1, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj2, _b0_2, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj2, _b0_3, _acc7);

                _b1_3 = _mm512_shuffle_f32x4(_b0_2, _b0_3, 0x4e);

                _aj1 = _mm512_set1_ps(Filter[7]);

                _acc0 = _mm512_fmadd_ps(_aj1, _b1_4, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj1, _b1_5, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj1, _b1_6, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj1, _b1_7, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj1, _b1_0, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj1, _b1_1, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj1, _b1_2, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj1, _b1_3, _acc7);

                _b0_4 = _mm512_maskz_load_ps(_rightInputMask4, pInput0 + 64);

                _aj2 = _mm512_set1_ps(Filter[10]);

                _acc0 = _mm512_fmadd_ps(_aj2, _b0_5, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj2, _b0_6, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj2, _b0_7, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj2, _b0_0, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj2, _b0_1, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj2, _b0_2, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj2, _b0_3, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj2, _b0_4, _acc7);

                _b1_4 = _mm512_shuffle_f32x4(_b0_3, _b0_4, 0x4e);

                _aj1 = _mm512_set1_ps(Filter[9]);

                _acc0 = _mm512_fmadd_ps(_aj1, _b1_5, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj1, _b1_6, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj1, _b1_7, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj1, _b1_0, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj1, _b1_1, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj1, _b1_2, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj1, _b1_3, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj1, _b1_4, _acc7);

                _b0_5 = _mm512_maskz_load_ps(_rightInputMask5, pInput0 + 80);

                _aj2 = _mm512_set1_ps(Filter[12]);

                _acc0 = _mm512_fmadd_ps(_aj2, _b0_6, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj2, _b0_7, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj2, _b0_0, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj2, _b0_1, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj2, _b0_2, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj2, _b0_3, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj2, _b0_4, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj2, _b0_5, _acc7);

                _b1_5 = _mm512_shuffle_f32x4(_b0_4, _b0_5, 0x4e);

                _aj1 = _mm512_set1_ps(Filter[11]);

                _acc0 = _mm512_fmadd_ps(_aj1, _b1_6, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj1, _b1_7, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj1, _b1_0, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj1, _b1_1, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj1, _b1_2, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj1, _b1_3, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj1, _b1_4, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj1, _b1_5, _acc7);

                _b0_6 = _mm512_maskz_load_ps(_rightInputMask6, pInput0 + 96);

                _aj2 = _mm512_set1_ps(Filter[14]);

                _acc0 = _mm512_fmadd_ps(_aj2, _b0_7, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj2, _b0_0, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj2, _b0_1, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj2, _b0_2, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj2, _b0_3, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj2, _b0_4, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj2, _b0_5, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj2, _b0_6, _acc7);

                _b1_6 = _mm512_shuffle_f32x4(_b0_5, _b0_6, 0x4e);

                _aj1 = _mm512_set1_ps(Filter[13]);

                _acc0 = _mm512_fmadd_ps(_aj1, _b1_7, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj1, _b1_0, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj1, _b1_1, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj1, _b1_2, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj1, _b1_3, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj1, _b1_4, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj1, _b1_5, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj1, _b1_6, _acc7);

                _b0_7 = _mm512_maskz_load_ps(_rightInputMask7, pInput0 + 112);

                _aj2 = _mm512_set1_ps(Filter[16]);

                _acc0 = _mm512_fmadd_ps(_aj2, _b0_0, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj2, _b0_1, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj2, _b0_2, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj2, _b0_3, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj2, _b0_4, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj2, _b0_5, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj2, _b0_6, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj2, _b0_7, _acc7);

                _b1_7 = _mm512_shuffle_f32x4(_b0_6, _b0_7, 0x4e);

                _aj1 = _mm512_set1_ps(Filter[15]);

                _acc0 = _mm512_fmadd_ps(_aj1, _b1_0, _acc0);
                _acc1 = _mm512_fmadd_ps(_aj1, _b1_1, _acc1);
                _acc2 = _mm512_fmadd_ps(_aj1, _b1_2, _acc2);
                _acc3 = _mm512_fmadd_ps(_aj1, _b1_3, _acc3);
                _acc4 = _mm512_fmadd_ps(_aj1, _b1_4, _acc4);
                _acc5 = _mm512_fmadd_ps(_aj1, _b1_5, _acc5);
                _acc6 = _mm512_fmadd_ps(_aj1, _b1_6, _acc6);
                _acc7 = _mm512_fmadd_ps(_aj1, _b1_7, _acc7);

                if (widthOutputRemainder < widthInputRemainder) {
                        _acc0 = _mm512_add_ps(_acc0, _mm512_load_ps(pOutput0));
                        _acc1 = _mm512_add_ps(_acc1, _mm512_load_ps(pOutput0 + 16));
                        _acc2 = _mm512_add_ps(_acc2, _mm512_load_ps(pOutput0 + 32));
                        _acc3 = _mm512_add_ps(_acc3, _mm512_load_ps(pOutput0 + 48));
                        _acc4 = _mm512_add_ps(_acc4, _mm512_load_ps(pOutput0 + 64));
                        _acc5 = _mm512_add_ps(_acc5, _mm512_load_ps(pOutput0 + 80));
                        _acc6 = _mm512_add_ps(_acc6, _mm512_load_ps(pOutput0 + 96));
                        _acc7 = _mm512_add_ps(_acc7, _mm512_load_ps(pOutput0 + 112));

                        _mm512_store_ps(pOutput0 + 0, _acc0);
                        _mm512_store_ps(pOutput0 + 16, _acc1);
                        _mm512_store_ps(pOutput0 + 32, _acc2);
                        _mm512_store_ps(pOutput0 + 48, _acc3);
                        _mm512_store_ps(pOutput0 + 64, _acc4);
                        _mm512_store_ps(pOutput0 + 80, _acc5);
                        _mm512_store_ps(pOutput0 + 96, _acc6);
                        _mm512_store_ps(pOutput0 + 112, _acc7);

                        pOutput0 += widthStep;

                        _b1_0 = _mm512_shuffle_f32x4(_b0_7, _mm512_setzero_ps(), 0x4e);

                        _acc0 = _mm512_mul_ps(_a0, _b0_0);
                        _acc1 = _mm512_mul_ps(_a0, _b0_1);
                        _acc2 = _mm512_mul_ps(_a0, _b0_2);
                        _acc3 = _mm512_mul_ps(_a0, _b0_3);
                        _acc4 = _mm512_mul_ps(_a0, _b0_4);
                        _acc5 = _mm512_mul_ps(_a0, _b0_5);
                        _acc6 = _mm512_mul_ps(_a0, _b0_6);
                        _acc7 = _mm512_mul_ps(_a0, _b0_7);

                        _aj1 = _mm512_set1_ps(Filter[1]);

                        _acc0 = _mm512_fmadd_ps(_aj1, _b1_1, _acc0);
                        _acc1 = _mm512_fmadd_ps(_aj1, _b1_2, _acc1);
                        _acc2 = _mm512_fmadd_ps(_aj1, _b1_3, _acc2);
                        _acc3 = _mm512_fmadd_ps(_aj1, _b1_4, _acc3);
                        _acc4 = _mm512_fmadd_ps(_aj1, _b1_5, _acc4);
                        _acc5 = _mm512_fmadd_ps(_aj1, _b1_6, _acc5);
                        _acc6 = _mm512_fmadd_ps(_aj1, _b1_7, _acc6);
                        _acc7 = _mm512_fmadd_ps(_aj1, _b1_0, _acc7);

                        _aj2 = _mm512_set1_ps(Filter[2]);

                        _acc0 = _mm512_fmadd_ps(_aj2, _b0_1, _acc0);
                        _acc1 = _mm512_fmadd_ps(_aj2, _b0_2, _acc1);
                        _acc2 = _mm512_fmadd_ps(_aj2, _b0_3, _acc2);
                        _acc3 = _mm512_fmadd_ps(_aj2, _b0_4, _acc3);
                        _acc4 = _mm512_fmadd_ps(_aj2, _b0_5, _acc4);
                        _acc5 = _mm512_fmadd_ps(_aj2, _b0_6, _acc5);
                        _acc6 = _mm512_fmadd_ps(_aj2, _b0_7, _acc6);

                        _aj1 = _mm512_set1_ps(Filter[3]);

                        _acc0 = _mm512_fmadd_ps(_aj1, _b1_2, _acc0);
                        _acc1 = _mm512_fmadd_ps(_aj1, _b1_3, _acc1);
                        _acc2 = _mm512_fmadd_ps(_aj1, _b1_4, _acc2);
                        _acc3 = _mm512_fmadd_ps(_aj1, _b1_5, _acc3);
                        _acc4 = _mm512_fmadd_ps(_aj1, _b1_6, _acc4);
                        _acc5 = _mm512_fmadd_ps(_aj1, _b1_7, _acc5);
                        _acc6 = _mm512_fmadd_ps(_aj1, _b1_0, _acc6);

                        _aj2 = _mm512_set1_ps(Filter[4]);

                        _acc0 = _mm512_fmadd_ps(_aj2, _b0_2, _acc0);
                        _acc1 = _mm512_fmadd_ps(_aj2, _b0_3, _acc1);
                        _acc2 = _mm512_fmadd_ps(_aj2, _b0_4, _acc2);
                        _acc3 = _mm512_fmadd_ps(_aj2, _b0_5, _acc3);
                        _acc4 = _mm512_fmadd_ps(_aj2, _b0_6, _acc4);
                        _acc5 = _mm512_fmadd_ps(_aj2, _b0_7, _acc5);

                        _aj1 = _mm512_set1_ps(Filter[5]);

                        _acc0 = _mm512_fmadd_ps(_aj1, _b1_3, _acc0);
                        _acc1 = _mm512_fmadd_ps(_aj1, _b1_4, _acc1);
                        _acc2 = _mm512_fmadd_ps(_aj1, _b1_5, _acc2);
                        _acc3 = _mm512_fmadd_ps(_aj1, _b1_6, _acc3);
                        _acc4 = _mm512_fmadd_ps(_aj1, _b1_7, _acc4);
                        _acc5 = _mm512_fmadd_ps(_aj1, _b1_0, _acc5);

                        _aj2 = _mm512_set1_ps(Filter[6]);

                        _acc0 = _mm512_fmadd_ps(_aj2, _b0_3, _acc0);
                        _acc1 = _mm512_fmadd_ps(_aj2, _b0_4, _acc1);
                        _acc2 = _mm512_fmadd_ps(_aj2, _b0_5, _acc2);
                        _acc3 = _mm512_fmadd_ps(_aj2, _b0_6, _acc3);
                        _acc4 = _mm512_fmadd_ps(_aj2, _b0_7, _acc4);

                        _aj1 = _mm512_set1_ps(Filter[7]);

                        _acc0 = _mm512_fmadd_ps(_aj1, _b1_4, _acc0);
                        _acc1 = _mm512_fmadd_ps(_aj1, _b1_5, _acc1);
                        _acc2 = _mm512_fmadd_ps(_aj1, _b1_6, _acc2);
                        _acc3 = _mm512_fmadd_ps(_aj1, _b1_7, _acc3);
                        _acc4 = _mm512_fmadd_ps(_aj1, _b1_0, _acc4);

                        _aj2 = _mm512_set1_ps(Filter[8]);

                        _acc0 = _mm512_fmadd_ps(_aj2, _b0_4, _acc0);
                        _acc1 = _mm512_fmadd_ps(_aj2, _b0_5, _acc1);
                        _acc2 = _mm512_fmadd_ps(_aj2, _b0_6, _acc2);
                        _acc3 = _mm512_fmadd_ps(_aj2, _b0_7, _acc3);

                        _aj1 = _mm512_set1_ps(Filter[9]);

                        _acc0 = _mm512_fmadd_ps(_aj1, _b1_5, _acc0);
                        _acc1 = _mm512_fmadd_ps(_aj1, _b1_6, _acc1);
                        _acc2 = _mm512_fmadd_ps(_aj1, _b1_7, _acc2);
                        _acc3 = _mm512_fmadd_ps(_aj1, _b1_0, _acc3);

                        _aj2 = _mm512_set1_ps(Filter[10]);

                        _acc0 = _mm512_fmadd_ps(_aj2, _b0_5, _acc0);
                        _acc1 = _mm512_fmadd_ps(_aj2, _b0_6, _acc1);
                        _acc2 = _mm512_fmadd_ps(_aj2, _b0_7, _acc2);

                        _aj1 = _mm512_set1_ps(Filter[11]);

                        _acc0 = _mm512_fmadd_ps(_aj1, _b1_6, _acc0);
                        _acc1 = _mm512_fmadd_ps(_aj1, _b1_7, _acc1);
                        _acc2 = _mm512_fmadd_ps(_aj1, _b1_0, _acc2);

                        _aj2 = _mm512_set1_ps(Filter[12]);

                        _acc0 = _mm512_fmadd_ps(_aj2, _b0_6, _acc0);
                        _acc1 = _mm512_fmadd_ps(_aj2, _b0_7, _acc1);

                        _aj1 = _mm512_set1_ps(Filter[13]);

                        _acc0 = _mm512_fmadd_ps(_aj1, _b1_7, _acc0);
                        _acc1 = _mm512_fmadd_ps(_aj1, _b1_0, _acc1);

                        _aj2 = _mm512_set1_ps(Filter[14]);

                        _acc0 = _mm512_fmadd_ps(_aj2, _b0_7, _acc0);

                        _aj1 = _mm512_set1_ps(Filter[15]);

                        _acc0 = _mm512_fmadd_ps(_aj1, _b1_0, _acc0);
                }

                // Store
                _acc0 = _mm512_add_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask0, pOutput0 + 0));
                _acc1 =
                    _mm512_add_ps(_acc1, _mm512_maskz_load_ps(_rightOutputMask1, pOutput0 + 16));
                _acc2 =
                    _mm512_add_ps(_acc2, _mm512_maskz_load_ps(_rightOutputMask2, pOutput0 + 32));
                _acc3 =
                    _mm512_add_ps(_acc3, _mm512_maskz_load_ps(_rightOutputMask3, pOutput0 + 48));
                _acc4 =
                    _mm512_add_ps(_acc4, _mm512_maskz_load_ps(_rightOutputMask4, pOutput0 + 64));
                _acc5 =
                    _mm512_add_ps(_acc5, _mm512_maskz_load_ps(_rightOutputMask5, pOutput0 + 80));
                _acc6 =
                    _mm512_add_ps(_acc6, _mm512_maskz_load_ps(_rightOutputMask6, pOutput0 + 96));
                _acc7 =
                    _mm512_add_ps(_acc7, _mm512_maskz_load_ps(_rightOutputMask7, pOutput0 + 112));

                _mm512_mask_store_ps(pOutput0 + 0, _rightOutputMask0, _acc0);
                _mm512_mask_store_ps(pOutput0 + 16, _rightOutputMask1, _acc1);
                _mm512_mask_store_ps(pOutput0 + 32, _rightOutputMask2, _acc2);
                _mm512_mask_store_ps(pOutput0 + 48, _rightOutputMask3, _acc3);
                _mm512_mask_store_ps(pOutput0 + 64, _rightOutputMask4, _acc4);
                _mm512_mask_store_ps(pOutput0 + 80, _rightOutputMask5, _acc5);
                _mm512_mask_store_ps(pOutput0 + 96, _rightOutputMask6, _acc6);
                _mm512_mask_store_ps(pOutput0 + 112, _rightOutputMask7, _acc7);
        }
}

void
MlasConv1DSlidingKernelD8K51A(const MLAS_CONV_PARAMETERS* Parameters,
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

        const int64_t DilatedWidth =
            KernelWidth + (KernelWidth - 1) * (Parameters->DilationShape[WidthShapeIndex] - 1);

        assert(KernelWidth == 51);
        assert(OutputWidth == (InputWidth - DilatedWidth + PaddingLeft + PaddingRight + 1));
        MLAS_UNREFERENCED_PARAMETER(PaddingRight);

        constexpr int64_t vectorWidth = 16;

        constexpr size_t widthStep = 400;

        const int64_t paddingLeftSeek = DilatedWidth - PaddingLeft - 1;

        const int64_t paddingLeftShift = widthStep - paddingLeftSeek;

        const int64_t paddingRightShift = std::max<int64_t>(paddingLeftSeek - InputWidth, 0ll);

        __mmask16 _leftInputMask[25];
        for (int64_t k = 0, l = paddingLeftShift; k < 25; k++, l = std::max(l - vectorWidth, 0ll)) {
            _leftInputMask[k] = __mmask16((l < 16ll) ? ~0u << l : 0u);
        }

        const int64_t InputWidthBase = InputWidth - paddingLeftSeek + paddingRightShift;

        const int64_t widthIterations = InputWidthBase / vectorWidth;

        const float* pInput0 = Input - paddingLeftShift;

        __m512 _b0_0 = _mm512_maskz_load_ps(_leftInputMask[0], pInput0 + 0);

        __m512 _acc0 = _mm512_mul_ps(_b0_0, _mm512_set1_ps(Filter[0]));

        __m512 _b0_1 = _mm512_maskz_load_ps(_leftInputMask[1], pInput0 + 1 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc0);
        __m512 _acc1 = _mm512_mul_ps(_b0_1, _mm512_set1_ps(Filter[0]));

        __m512 _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc0);

        _b0_0 = _mm512_maskz_load_ps(_leftInputMask[2], pInput0 + 2 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[4]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[2]), _acc1);
        __m512 _acc2 = _mm512_mul_ps(_b0_0, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_1, _b0_0, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc1);

        _b0_1 = _mm512_maskz_load_ps(_leftInputMask[3], pInput0 + 3 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[6]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[4]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc2);
        __m512 _acc3 = _mm512_mul_ps(_b0_1, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc2);

        _b0_0 = _mm512_maskz_load_ps(_leftInputMask[4], pInput0 + 4 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[8]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[6]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[4]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[2]), _acc3);
        __m512 _acc4 = _mm512_mul_ps(_b0_0, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_1, _b0_0, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc3);

        _b0_1 = _mm512_maskz_load_ps(_leftInputMask[5], pInput0 + 5 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[10]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[8]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[6]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[4]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc4);
        __m512 _acc5 = _mm512_mul_ps(_b0_1, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc4);

        _b0_0 = _mm512_maskz_load_ps(_leftInputMask[6], pInput0 + 6 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[12]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[10]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[8]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[6]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[4]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[2]), _acc5);
        __m512 _acc6 = _mm512_mul_ps(_b0_0, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_1, _b0_0, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc5);

        _b0_1 = _mm512_maskz_load_ps(_leftInputMask[7], pInput0 + 7 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[14]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[12]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[10]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[8]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[6]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[4]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc6);
        __m512 _acc7 = _mm512_mul_ps(_b0_1, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc6);

        _b0_0 = _mm512_maskz_load_ps(_leftInputMask[8], pInput0 + 8 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[16]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[14]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[12]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[10]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[8]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[6]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[4]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[2]), _acc7);
        __m512 _acc8 = _mm512_mul_ps(_b0_0, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_1, _b0_0, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc7);

        _b0_1 = _mm512_maskz_load_ps(_leftInputMask[9], pInput0 + 9 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[18]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[16]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[14]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[12]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[10]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[8]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[6]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[4]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc8);
        __m512 _acc9 = _mm512_mul_ps(_b0_1, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc8);

        _b0_0 = _mm512_maskz_load_ps(_leftInputMask[10], pInput0 + 10 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[20]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[18]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[16]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[14]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[12]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[10]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[8]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[6]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[4]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[2]), _acc9);
        __m512 _acc10 = _mm512_mul_ps(_b0_0, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_1, _b0_0, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc9);

        _b0_1 = _mm512_maskz_load_ps(_leftInputMask[11], pInput0 + 11 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[22]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[20]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[18]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[16]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[14]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[12]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[10]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[8]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[6]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[4]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc10);
        __m512 _acc11 = _mm512_mul_ps(_b0_1, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc10);

        _b0_0 = _mm512_maskz_load_ps(_leftInputMask[12], pInput0 + 12 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[24]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[22]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[20]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[18]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[16]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[14]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[12]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[10]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[8]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[6]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[4]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[2]), _acc11);
        __m512 _acc12 = _mm512_mul_ps(_b0_0, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_1, _b0_0, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc11);

        _b0_1 = _mm512_maskz_load_ps(_leftInputMask[13], pInput0 + 13 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[26]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[24]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[22]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[20]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[18]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[16]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[14]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[12]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[10]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[8]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[6]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[4]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc12);
        __m512 _acc13 = _mm512_mul_ps(_b0_1, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc12);

        _b0_0 = _mm512_maskz_load_ps(_leftInputMask[14], pInput0 + 14 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[28]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[26]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[24]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[22]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[20]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[18]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[16]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[14]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[12]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[10]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[8]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[6]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[4]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[2]), _acc13);
        __m512 _acc14 = _mm512_mul_ps(_b0_0, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_1, _b0_0, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[27]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc13);

        _b0_1 = _mm512_maskz_load_ps(_leftInputMask[15], pInput0 + 15 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[30]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[28]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[26]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[24]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[22]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[20]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[18]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[16]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[14]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[12]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[10]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[8]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[6]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[4]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc14);
        __m512 _acc15 = _mm512_mul_ps(_b0_1, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[29]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[27]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc14);

        _b0_0 = _mm512_maskz_load_ps(_leftInputMask[16], pInput0 + 16 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[32]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[30]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[28]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[26]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[24]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[22]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[20]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[18]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[16]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[14]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[12]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[10]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[8]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[6]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[4]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[2]), _acc15);
        __m512 _acc16 = _mm512_mul_ps(_b0_0, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_1, _b0_0, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[31]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[29]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[27]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc15);

        _b0_1 = _mm512_maskz_load_ps(_leftInputMask[17], pInput0 + 17 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[34]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[32]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[30]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[28]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[26]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[24]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[22]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[20]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[18]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[16]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[14]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[12]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[10]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[8]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[6]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[4]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc16);
        __m512 _acc17 = _mm512_mul_ps(_b0_1, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[33]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[31]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[29]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[27]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc16);

        _b0_0 = _mm512_maskz_load_ps(_leftInputMask[18], pInput0 + 18 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[36]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[34]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[32]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[30]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[28]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[26]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[24]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[22]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[20]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[18]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[16]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[14]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[12]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[10]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[8]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[6]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[4]), _acc16);
        _acc17 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[2]), _acc17);
        __m512 _acc18 = _mm512_mul_ps(_b0_0, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_1, _b0_0, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[35]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[33]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[31]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[29]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[27]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc16);
        _acc17 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc17);

        _b0_1 = _mm512_maskz_load_ps(_leftInputMask[19], pInput0 + 19 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[38]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[36]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[34]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[32]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[30]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[28]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[26]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[24]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[22]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[20]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[18]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[16]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[14]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[12]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[10]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[8]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[6]), _acc16);
        _acc17 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[4]), _acc17);
        _acc18 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc18);
        __m512 _acc19 = _mm512_mul_ps(_b0_1, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[37]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[35]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[33]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[31]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[29]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[27]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc16);
        _acc17 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc17);
        _acc18 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc18);

        _b0_0 = _mm512_maskz_load_ps(_leftInputMask[20], pInput0 + 20 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[40]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[38]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[36]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[34]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[32]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[30]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[28]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[26]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[24]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[22]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[20]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[18]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[16]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[14]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[12]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[10]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[8]), _acc16);
        _acc17 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[6]), _acc17);
        _acc18 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[4]), _acc18);
        _acc19 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[2]), _acc19);
        __m512 _acc20 = _mm512_mul_ps(_b0_0, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_1, _b0_0, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[39]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[37]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[35]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[33]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[31]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[29]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[27]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc16);
        _acc17 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc17);
        _acc18 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc18);
        _acc19 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc19);

        _b0_1 = _mm512_maskz_load_ps(_leftInputMask[21], pInput0 + 21 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[42]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[40]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[38]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[36]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[34]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[32]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[30]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[28]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[26]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[24]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[22]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[20]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[18]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[16]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[14]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[12]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[10]), _acc16);
        _acc17 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[8]), _acc17);
        _acc18 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[6]), _acc18);
        _acc19 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[4]), _acc19);
        _acc20 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc20);
        __m512 _acc21 = _mm512_mul_ps(_b0_1, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[41]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[39]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[37]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[35]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[33]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[31]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[29]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[27]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc16);
        _acc17 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc17);
        _acc18 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc18);
        _acc19 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc19);
        _acc20 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc20);

        _b0_0 = _mm512_maskz_load_ps(_leftInputMask[22], pInput0 + 22 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[44]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[42]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[40]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[38]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[36]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[34]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[32]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[30]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[28]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[26]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[24]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[22]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[20]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[18]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[16]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[14]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[12]), _acc16);
        _acc17 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[10]), _acc17);
        _acc18 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[8]), _acc18);
        _acc19 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[6]), _acc19);
        _acc20 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[4]), _acc20);
        _acc21 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[2]), _acc21);
        __m512 _acc22 = _mm512_mul_ps(_b0_0, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_1, _b0_0, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[43]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[41]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[39]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[37]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[35]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[33]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[31]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[29]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[27]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc16);
        _acc17 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc17);
        _acc18 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc18);
        _acc19 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc19);
        _acc20 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc20);
        _acc21 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc21);

        _b0_1 = _mm512_maskz_load_ps(_leftInputMask[23], pInput0 + 23 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[46]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[44]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[42]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[40]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[38]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[36]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[34]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[32]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[30]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[28]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[26]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[24]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[22]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[20]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[18]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[16]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[14]), _acc16);
        _acc17 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[12]), _acc17);
        _acc18 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[10]), _acc18);
        _acc19 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[8]), _acc19);
        _acc20 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[6]), _acc20);
        _acc21 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[4]), _acc21);
        _acc22 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc22);
        __m512 _acc23 = _mm512_mul_ps(_b0_1, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[45]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[43]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[41]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[39]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[37]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[35]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[33]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[31]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[29]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[27]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc16);
        _acc17 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc17);
        _acc18 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc18);
        _acc19 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc19);
        _acc20 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc20);
        _acc21 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc21);
        _acc22 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc22);

        _b0_0 = _mm512_maskz_load_ps(_leftInputMask[24], pInput0 + 24 * vectorWidth);

        _acc0 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[48]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[46]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[44]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[42]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[40]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[38]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[36]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[34]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[32]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[30]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[28]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[26]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[24]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[22]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[20]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[18]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[16]), _acc16);
        _acc17 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[14]), _acc17);
        _acc18 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[12]), _acc18);
        _acc19 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[10]), _acc19);
        _acc20 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[8]), _acc20);
        _acc21 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[6]), _acc21);
        _acc22 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[4]), _acc22);
        _acc23 = _mm512_fmadd_ps(_b0_0, _mm512_set1_ps(Filter[2]), _acc23);
        __m512 _acc24 = _mm512_mul_ps(_b0_0, _mm512_set1_ps(Filter[0]));

        _b1_1 = _mm512_shuffle_f32x4(_b0_1, _b0_0, 0x4e);

        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[47]), _acc0);
        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[45]), _acc1);
        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[43]), _acc2);
        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[41]), _acc3);
        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[39]), _acc4);
        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[37]), _acc5);
        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[35]), _acc6);
        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[33]), _acc7);
        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[31]), _acc8);
        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[29]), _acc9);
        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[27]), _acc10);
        _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc11);
        _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc12);
        _acc13 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc13);
        _acc14 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc14);
        _acc15 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc15);
        _acc16 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc16);
        _acc17 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc17);
        _acc18 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc18);
        _acc19 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc19);
        _acc20 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc20);
        _acc21 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc21);
        _acc22 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc22);
        _acc23 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc23);

        pInput0 += widthStep;

        float* pOutput0 = Output;

        for (int64_t i = 0; i < widthIterations; i++) {
                _b0_1 = _mm512_load_ps(pInput0);

                _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[50]), _acc0);
                _acc1 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[48]), _acc1);
                _acc2 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[46]), _acc2);
                _acc3 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[44]), _acc3);
                _acc4 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[42]), _acc4);
                _acc5 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[40]), _acc5);
                _acc6 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[38]), _acc6);
                _acc7 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[36]), _acc7);
                _acc8 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[34]), _acc8);
                _acc9 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[32]), _acc9);
                _acc10 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[30]), _acc10);
                _acc11 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[28]), _acc11);
                _acc12 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[26]), _acc12);
                _acc13 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[24]), _acc13);
                _acc14 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[22]), _acc14);
                _acc15 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[20]), _acc15);
                _acc16 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[18]), _acc16);
                _acc17 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[16]), _acc17);
                _acc18 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[14]), _acc18);
                _acc19 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[12]), _acc19);
                _acc20 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[10]), _acc20);
                _acc21 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[8]), _acc21);
                _acc22 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[6]), _acc22);
                _acc23 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[4]), _acc23);
                _acc24 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc24);

                _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

                _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[49]), _acc0);
                _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[47]), _acc1);
                _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[45]), _acc2);
                _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[43]), _acc3);
                _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[41]), _acc4);
                _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[39]), _acc5);
                _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[37]), _acc6);
                _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[35]), _acc7);
                _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[33]), _acc8);
                _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[31]), _acc9);
                _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[29]), _acc10);
                _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[27]), _acc11);
                _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc12);
                _acc13 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc13);
                _acc14 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc14);
                _acc15 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc15);
                _acc16 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc16);
                _acc17 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc17);
                _acc18 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc18);
                _acc19 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc19);
                _acc20 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc20);
                _acc21 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc21);
                _acc22 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc22);
                _acc23 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc23);
                _acc24 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc24);

                __m512 _accf = _mm512_add_ps(_acc0, _mm512_load_ps(pOutput0));

                _acc0 = _acc1;
                _acc1 = _acc2;
                _acc2 = _acc3;
                _acc3 = _acc4;
                _acc4 = _acc5;
                _acc5 = _acc6;
                _acc6 = _acc7;
                _acc7 = _acc8;
                _acc8 = _acc9;
                _acc9 = _acc10;
                _acc10 = _acc11;
                _acc11 = _acc12;
                _acc12 = _acc13;
                _acc13 = _acc14;
                _acc14 = _acc15;
                _acc15 = _acc16;
                _acc16 = _acc17;
                _acc17 = _acc18;
                _acc18 = _acc19;
                _acc19 = _acc20;
                _acc20 = _acc21;
                _acc21 = _acc22;
                _acc22 = _acc23;
                _acc23 = _acc24;
                _acc24 = _mm512_mul_ps(_b0_1, _mm512_set1_ps(Filter[0]));

                _mm512_store_ps(pOutput0, _accf);

                _b0_0 = _b0_1;

                pInput0 += vectorWidth;

                pOutput0 += vectorWidth;
        }

        // Right Edge
        {
                const int64_t widthInputRemainder = InputWidthBase - widthIterations * vectorWidth;
                const __mmask16 _rightInputMask = __mmask16(~(~0u << widthInputRemainder));

                _b0_1 = _mm512_maskz_load_ps(_rightInputMask, pInput0);

                _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[50]), _acc0);
                _acc1 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[48]), _acc1);
                _acc2 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[46]), _acc2);
                _acc3 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[44]), _acc3);
                _acc4 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[42]), _acc4);
                _acc5 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[40]), _acc5);
                _acc6 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[38]), _acc6);
                _acc7 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[36]), _acc7);
                _acc8 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[34]), _acc8);
                _acc9 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[32]), _acc9);
                _acc10 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[30]), _acc10);
                _acc11 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[28]), _acc11);
                _acc12 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[26]), _acc12);
                _acc13 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[24]), _acc13);
                _acc14 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[22]), _acc14);
                _acc15 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[20]), _acc15);
                _acc16 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[18]), _acc16);
                _acc17 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[16]), _acc17);
                _acc18 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[14]), _acc18);
                _acc19 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[12]), _acc19);
                _acc20 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[10]), _acc20);
                _acc21 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[8]), _acc21);
                _acc22 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[6]), _acc22);
                _acc23 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[4]), _acc23);
                _acc24 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc24);

                _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

                _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[49]), _acc0);
                _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[47]), _acc1);
                _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[45]), _acc2);
                _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[43]), _acc3);
                _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[41]), _acc4);
                _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[39]), _acc5);
                _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[37]), _acc6);
                _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[35]), _acc7);
                _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[33]), _acc8);
                _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[31]), _acc9);
                _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[29]), _acc10);
                _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[27]), _acc11);
                _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc12);
                _acc13 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc13);
                _acc14 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc14);
                _acc15 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc15);
                _acc16 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc16);
                _acc17 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc17);
                _acc18 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc18);
                _acc19 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc19);
                _acc20 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc20);
                _acc21 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc21);
                _acc22 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc22);
                _acc23 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc23);
                _acc24 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc24);

                int64_t widthOutputRemainder = OutputWidth - widthIterations * vectorWidth;

                for (; widthOutputRemainder > vectorWidth; widthOutputRemainder -= vectorWidth) {
                        __m512 _accf = _mm512_add_ps(_acc0, _mm512_load_ps(pOutput0));

                        _acc0 = _acc1;
                        _acc1 = _acc2;
                        _acc2 = _acc3;
                        _acc3 = _acc4;
                        _acc4 = _acc5;
                        _acc5 = _acc6;
                        _acc6 = _acc7;
                        _acc7 = _acc8;
                        _acc8 = _acc9;
                        _acc9 = _acc10;
                        _acc10 = _acc11;
                        _acc11 = _acc12;
                        _acc12 = _acc13;
                        _acc13 = _acc14;
                        _acc14 = _acc15;
                        _acc15 = _acc16;
                        _acc16 = _acc17;
                        _acc17 = _acc18;
                        _acc18 = _acc19;
                        _acc19 = _acc20;
                        _acc20 = _acc21;
                        _acc21 = _acc22;
                        _acc22 = _acc23;
                        _acc23 = _acc24;
                        _acc24 = _mm512_mul_ps(_b0_1, _mm512_set1_ps(Filter[0]));

                        _mm512_store_ps(pOutput0, _accf);

                        _b0_0 = _b0_1;

                        pOutput0 += vectorWidth;

                        _b0_1 = _mm512_setzero_ps();

                        _acc0 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[50]), _acc0);
                        _acc1 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[48]), _acc1);
                        _acc2 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[46]), _acc2);
                        _acc3 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[44]), _acc3);
                        _acc4 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[42]), _acc4);
                        _acc5 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[40]), _acc5);
                        _acc6 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[38]), _acc6);
                        _acc7 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[36]), _acc7);
                        _acc8 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[34]), _acc8);
                        _acc9 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[32]), _acc9);
                        _acc10 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[30]), _acc10);
                        _acc11 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[28]), _acc11);
                        _acc12 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[26]), _acc12);
                        _acc13 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[24]), _acc13);
                        _acc14 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[22]), _acc14);
                        _acc15 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[20]), _acc15);
                        _acc16 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[18]), _acc16);
                        _acc17 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[16]), _acc17);
                        _acc18 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[14]), _acc18);
                        _acc19 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[12]), _acc19);
                        _acc20 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[10]), _acc20);
                        _acc21 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[8]), _acc21);
                        _acc22 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[6]), _acc22);
                        _acc23 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[4]), _acc23);
                        _acc24 = _mm512_fmadd_ps(_b0_1, _mm512_set1_ps(Filter[2]), _acc24);

                        _b1_1 = _mm512_shuffle_f32x4(_b0_0, _b0_1, 0x4e);

                        _acc0 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[49]), _acc0);
                        _acc1 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[47]), _acc1);
                        _acc2 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[45]), _acc2);
                        _acc3 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[43]), _acc3);
                        _acc4 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[41]), _acc4);
                        _acc5 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[39]), _acc5);
                        _acc6 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[37]), _acc6);
                        _acc7 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[35]), _acc7);
                        _acc8 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[33]), _acc8);
                        _acc9 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[31]), _acc9);
                        _acc10 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[29]), _acc10);
                        _acc11 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[27]), _acc11);
                        _acc12 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[25]), _acc12);
                        _acc13 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[23]), _acc13);
                        _acc14 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[21]), _acc14);
                        _acc15 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[19]), _acc15);
                        _acc16 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[17]), _acc16);
                        _acc17 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[15]), _acc17);
                        _acc18 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[13]), _acc18);
                        _acc19 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[11]), _acc19);
                        _acc20 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[9]), _acc20);
                        _acc21 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[7]), _acc21);
                        _acc22 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[5]), _acc22);
                        _acc23 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[3]), _acc23);
                        _acc24 = _mm512_fmadd_ps(_b1_1, _mm512_set1_ps(Filter[1]), _acc24);
                }

                const __mmask16 _rightOutputMask = __mmask16(~(~0u << widthOutputRemainder));

                __m512 _accf =
                    _mm512_add_ps(_acc0, _mm512_maskz_load_ps(_rightOutputMask, pOutput0));

                _mm512_mask_store_ps(pOutput0, _rightOutputMask, _accf);
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
        const int64_t DilationlWidth = Parameters->DilationShape[WidthShapeIndex];

		const int64_t DilatedKernelWidth =
                KernelWidth + (KernelWidth - 1) * (DilationlWidth - 1);

		if (DilatedKernelWidth <= 16) {
			return MlasConv1DSlidingKernelK17;
		} else if (DilatedKernelWidth <= 32) {
			return MlasConv1DSlidingKernelK32;
		} else if (DilatedKernelWidth <= 48) {
			return MlasConv1DSlidingKernelK48;
		} else if (DilatedKernelWidth <= 64) {
			return MlasConv1DSlidingKernelK64;
		} else if ((KernelWidth) == 51 && (DilationlWidth == 8)) {
            return MlasConv1DSlidingKernelD8K51;
		}
		else {
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
		}
		else if ((KernelHeight == 5) && (KernelWidth == 5)) {
			return MlasConv2DSlidingKernelK5x5;
		}
		else if (KernelWidth <= 16) {
			return MlasConv2DSlidingKernelK17;
		}
		else if (KernelWidth <= 32) {
			return MlasConv2DSlidingKernelK32;
		}
		else if (KernelWidth <= 48) {
			return MlasConv2DSlidingKernelK48;
		}
		else if (KernelWidth <= 64) {
			return MlasConv2DSlidingKernelK64;
		}
		else {
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
