/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    snchwc_strassen_kernel_avx512f.cpp

Abstract:

    AVX-512 pointwise convolution kernel variant for the fused one-level
    Strassen algorithm on the NCHWc layout.

    Differences from the standard pointwise kernel:

      - The input operand may be a folded two-term presum: the broadcast
        value is computed as InputA[i] + SignA * InputB[i], riding the spare
        load and ALU ports ahead of the FMA chain.

      - The register tile is written to one or two output destinations with
        independent signs and accumulate flags, so the Strassen C-quadrant
        combines happen from registers instead of through scratch memory.

    Bias addition and optional ReLU are applied per destination when
    requested (on the final product updating a given C quadrant).

--*/

#include "mlasi.h"

#if defined(MLAS_TARGET_AMD64)

#include <immintrin.h>

//
// One 16-channel input block step for a FilterCount x 6 register tile.
// Filter rows are loaded per channel; broadcasts come from one or two
// input streams.
//

namespace {

template <size_t FilterCount, size_t OutputCount>
struct MLAS_STRASSEN_TILE {
    __m512 Acc[FilterCount][OutputCount];

    MLAS_FORCEINLINE void Zero() {
#pragma GCC unroll 4
        for (size_t f = 0; f < FilterCount; f++) {
#pragma GCC unroll 6
            for (size_t o = 0; o < OutputCount; o++) {
                Acc[f][o] = _mm512_setzero_ps();
            }
        }
    }

    template <bool DualInput, bool SubtractInput>
    MLAS_FORCEINLINE void ChannelStep(const float* a1, const float* a2,
                                      const float* filter, size_t FilterStride) {
        __m512 f[FilterCount];
#pragma GCC unroll 4
        for (size_t fi = 0; fi < FilterCount; fi++) {
            f[fi] = _mm512_loadu_ps(filter + fi * FilterStride);
        }
#pragma GCC unroll 6
        for (size_t o = 0; o < OutputCount; o++) {
            float value = a1[o * 16];
            if (DualInput) {
                value = SubtractInput ? value - a2[o * 16] : value + a2[o * 16];
            }
            const __m512 b = _mm512_set1_ps(value);
#pragma GCC unroll 4
            for (size_t fi = 0; fi < FilterCount; fi++) {
                Acc[fi][o] = _mm512_fmadd_ps(b, f[fi], Acc[fi][o]);
            }
        }
    }
};

//
// Epilogue: write the tile into a destination with sign/accumulate/bias/relu.
//

template <size_t FilterCount, size_t OutputCount>
MLAS_FORCEINLINE
void
StrassenStoreDestination(
    const MLAS_STRASSEN_TILE<FilterCount, OutputCount>& Tile,
    float* Output,
    size_t OutputStrideElements,
    bool Subtract,
    bool Accumulate,
    const float* Bias,
    bool Relu
    )
{
    const __m512 Zero = _mm512_setzero_ps();

#pragma GCC unroll 4
    for (size_t fi = 0; fi < FilterCount; fi++) {

        float* Plane = Output + fi * OutputStrideElements;

        __m512 BiasVector = Zero;
        if (Bias != nullptr) {
            BiasVector = _mm512_loadu_ps(Bias + fi * 16);
        }

#pragma GCC unroll 6
        for (size_t o = 0; o < OutputCount; o++) {

            __m512 Value = Tile.Acc[fi][o];

            if (Subtract) {
                Value = _mm512_sub_ps(Zero, Value);
            }

            if (Accumulate) {
                Value = _mm512_add_ps(_mm512_loadu_ps(Plane + o * 16), Value);
            }

            if (Bias != nullptr) {
                Value = _mm512_add_ps(Value, BiasVector);
            }

            if (Relu) {
                Value = _mm512_max_ps(Value, Zero);
            }

            _mm512_storeu_ps(Plane + o * 16, Value);
        }
    }
}

//
// Processes one spatial segment of OutputCount positions.
//

template <size_t FilterCount, size_t OutputCount, bool DualInput, bool SubtractInput>
MLAS_FORCEINLINE
void
StrassenProcessSegment(
    const MLAS_STRASSEN_POINTWISE_PARAMS* Params,
    size_t OutputIndex
    )
{
    MLAS_STRASSEN_TILE<FilterCount, OutputCount> Tile;
    Tile.Zero();

    const size_t InputStride = Params->InputStrideElements;

    const float* a1 = Params->InputA + OutputIndex * 16;
    const float* a2 = DualInput ? Params->InputB + OutputIndex * 16 : nullptr;
    const float* filter = Params->Filter;

    const size_t FilterStride = Params->FilterStrideElements;

    for (size_t cb = 0; cb < Params->InputChannelBlocks; cb++) {

        const float* a1c = a1;
        const float* a2c = a2;

        for (size_t lane = 0; lane < 16; lane++) {
            Tile.template ChannelStep<DualInput, SubtractInput>(
                a1c, a2c, filter + lane * 16, FilterStride);
            a1c += 1;
            if (DualInput) {
                a2c += 1;
            }
        }

        a1 += InputStride;
        if (DualInput) {
            a2 += InputStride;
        }
        filter += 16 * 16;
    }

    StrassenStoreDestination<FilterCount, OutputCount>(
        Tile,
        Params->Output1 + OutputIndex * 16,
        Params->OutputStrideElements,
        Params->Subtract1,
        Params->Accumulate1,
        Params->Bias1,
        Params->Relu1);

    if (Params->Output2 != nullptr) {
        StrassenStoreDestination<FilterCount, OutputCount>(
            Tile,
            Params->Output2 + OutputIndex * 16,
            Params->OutputStrideElements,
            Params->Subtract2,
            Params->Accumulate2,
            Params->Bias2,
            Params->Relu2);
    }
}

template <size_t FilterCount, bool DualInput, bool SubtractInput>
void
StrassenPointwiseKernelImpl(
    const MLAS_STRASSEN_POINTWISE_PARAMS* Params
    )
{
    size_t OutputIndex = 0;

    while (Params->OutputCount - OutputIndex >= 6) {
        StrassenProcessSegment<FilterCount, 6, DualInput, SubtractInput>(Params, OutputIndex);
        OutputIndex += 6;
    }

    switch (Params->OutputCount - OutputIndex) {
        case 5:
            StrassenProcessSegment<FilterCount, 5, DualInput, SubtractInput>(Params, OutputIndex);
            break;
        case 4:
            StrassenProcessSegment<FilterCount, 4, DualInput, SubtractInput>(Params, OutputIndex);
            break;
        case 3:
            StrassenProcessSegment<FilterCount, 3, DualInput, SubtractInput>(Params, OutputIndex);
            break;
        case 2:
            StrassenProcessSegment<FilterCount, 2, DualInput, SubtractInput>(Params, OutputIndex);
            break;
        case 1:
            StrassenProcessSegment<FilterCount, 1, DualInput, SubtractInput>(Params, OutputIndex);
            break;
        default:
            break;
    }
}

template <size_t FilterCount>
void
StrassenPointwiseKernelDispatchInput(
    const MLAS_STRASSEN_POINTWISE_PARAMS* Params
    )
{
    if (Params->InputB == nullptr) {
        StrassenPointwiseKernelImpl<FilterCount, false, false>(Params);
    } else if (Params->SubtractInput) {
        StrassenPointwiseKernelImpl<FilterCount, true, true>(Params);
    } else {
        StrassenPointwiseKernelImpl<FilterCount, true, false>(Params);
    }
}

} // namespace

void
MlasStrassenPointwiseKernelAvx512F(
    const MLAS_STRASSEN_POINTWISE_PARAMS* Params
    )
{
    switch (Params->FilterCount) {
        case 4:
            StrassenPointwiseKernelDispatchInput<4>(Params);
            break;
        case 3:
            StrassenPointwiseKernelDispatchInput<3>(Params);
            break;
        case 2:
            StrassenPointwiseKernelDispatchInput<2>(Params);
            break;
        default:
            StrassenPointwiseKernelDispatchInput<1>(Params);
            break;
    }
}

#endif
