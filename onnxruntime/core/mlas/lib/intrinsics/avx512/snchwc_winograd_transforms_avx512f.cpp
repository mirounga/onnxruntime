/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    snchwc_winograd_transforms_avx512f.cpp

Abstract:

    This module implements the AVX-512 input and output tile transforms for
    the NCHWc Winograd F(4x4,3x3) convolution algorithm. The 16-channel
    NCHWc block dimension maps directly onto the 16 float lanes of a ZMM
    register, so each transform value is a single vector.

--*/

#include "mlasi.h"

#if defined(MLAS_TARGET_AMD64)

#include <immintrin.h>

void
MlasWinogradInputTransformAvx512F(
    const MLAS_WINOGRAD_WORK_BLOCK* WorkBlock,
    size_t Tile,
    size_t InputBlock
    )
/*++

Routine Description:

    Transforms one 6x6 input patch for one 16-channel input block as B'dB
    and scatters the 36 coefficient vectors into the V matrices.

--*/
{
    constexpr size_t BlockSize = 16;

    const size_t LocalTile = Tile - WorkBlock->TileBegin;

    const size_t th = Tile / WorkBlock->TilesW;
    const size_t tw = Tile % WorkBlock->TilesW;

    const ptrdiff_t ih0 = ptrdiff_t(th * 4) - WorkBlock->PaddingTop;
    const ptrdiff_t iw0 = ptrdiff_t(tw * 4) - WorkBlock->PaddingLeft;

    const float* InputPlane = WorkBlock->Input +
        InputBlock * WorkBlock->InputHeight * WorkBlock->InputWidth * BlockSize;

    const __m512 Two = _mm512_set1_ps(2.0f);
    const __m512 Four = _mm512_set1_ps(4.0f);
    const __m512 Five = _mm512_set1_ps(5.0f);

    //
    // Gather the 6x6 patch with zero fill outside the input bounds, applying
    // the column pass w = B'd one column at a time to bound register usage.
    //

    __m512 w[6][6];

    const bool Interior =
        ih0 >= 0 && ih0 + 6 <= ptrdiff_t(WorkBlock->InputHeight) &&
        iw0 >= 0 && iw0 + 6 <= ptrdiff_t(WorkBlock->InputWidth);

    for (size_t c = 0; c < 6; c++) {

        __m512 d0, d1, d2, d3, d4, d5;

        if (Interior) {

            const float* Column = &InputPlane[(size_t(ih0) * WorkBlock->InputWidth + size_t(iw0) + c) * BlockSize];
            const size_t RowStride = WorkBlock->InputWidth * BlockSize;

            d0 = _mm512_loadu_ps(Column);
            d1 = _mm512_loadu_ps(Column + RowStride);
            d2 = _mm512_loadu_ps(Column + 2 * RowStride);
            d3 = _mm512_loadu_ps(Column + 3 * RowStride);
            d4 = _mm512_loadu_ps(Column + 4 * RowStride);
            d5 = _mm512_loadu_ps(Column + 5 * RowStride);

        } else {

            const ptrdiff_t iw = iw0 + ptrdiff_t(c);
            const bool ColumnValid = iw >= 0 && iw < ptrdiff_t(WorkBlock->InputWidth);

            __m512 d[6];

            for (size_t r = 0; r < 6; r++) {
                const ptrdiff_t ih = ih0 + ptrdiff_t(r);
                if (ColumnValid && ih >= 0 && ih < ptrdiff_t(WorkBlock->InputHeight)) {
                    d[r] = _mm512_loadu_ps(&InputPlane[(size_t(ih) * WorkBlock->InputWidth + size_t(iw)) * BlockSize]);
                } else {
                    d[r] = _mm512_setzero_ps();
                }
            }

            d0 = d[0]; d1 = d[1]; d2 = d[2]; d3 = d[3]; d4 = d[4]; d5 = d[5];
        }

        //
        // Column pass: w[r][c] = (B'd)[r] for this column.
        //
        //    w0 = 4*d0 - 5*d2 + d4
        //    w1 = -4*(d1 + d2) + d3 + d4
        //    w2 = 4*(d1 - d2) - d3 + d4
        //    w3 = 2*(d3 - d1) + d4 - d2
        //    w4 = 2*(d1 - d3) + d4 - d2
        //    w5 = 4*d1 - 5*d3 + d5
        //

        const __m512 s12 = _mm512_add_ps(d1, d2);
        const __m512 d12 = _mm512_sub_ps(d1, d2);
        const __m512 s34 = _mm512_add_ps(d3, d4);
        const __m512 d42 = _mm512_sub_ps(d4, d2);
        const __m512 d31 = _mm512_sub_ps(d3, d1);

        w[0][c] = _mm512_fmadd_ps(Four, d0, _mm512_fnmadd_ps(Five, d2, d4));
        w[1][c] = _mm512_fnmadd_ps(Four, s12, s34);
        w[2][c] = _mm512_fmadd_ps(Four, d12, _mm512_sub_ps(d4, d3));
        w[3][c] = _mm512_fmadd_ps(Two, d31, d42);
        w[4][c] = _mm512_fnmadd_ps(Two, d31, d42);
        w[5][c] = _mm512_fmadd_ps(Four, d1, _mm512_fnmadd_ps(Five, d3, d5));
    }

    //
    // Row pass: u = wB' applied along each row, storing the 36 coefficient
    // vectors to the V matrices.
    //

    float* V = WorkBlock->V + LocalTile * WorkBlock->InputChannels + InputBlock * BlockSize;
    const size_t VCoeffStride = WorkBlock->VCoeffStride;

    for (size_t r = 0; r < 6; r++) {

        const __m512 w0 = w[r][0], w1 = w[r][1], w2 = w[r][2];
        const __m512 w3 = w[r][3], w4 = w[r][4], w5 = w[r][5];

        const __m512 s12 = _mm512_add_ps(w1, w2);
        const __m512 d12 = _mm512_sub_ps(w1, w2);
        const __m512 s34 = _mm512_add_ps(w3, w4);
        const __m512 d42 = _mm512_sub_ps(w4, w2);
        const __m512 d31 = _mm512_sub_ps(w3, w1);

        float* VRow = V + r * 6 * VCoeffStride;

        _mm512_storeu_ps(VRow, _mm512_fmadd_ps(Four, w0, _mm512_fnmadd_ps(Five, w2, w4)));
        _mm512_storeu_ps(VRow + VCoeffStride, _mm512_fnmadd_ps(Four, s12, s34));
        _mm512_storeu_ps(VRow + 2 * VCoeffStride, _mm512_fmadd_ps(Four, d12, _mm512_sub_ps(w4, w3)));
        _mm512_storeu_ps(VRow + 3 * VCoeffStride, _mm512_fmadd_ps(Two, d31, d42));
        _mm512_storeu_ps(VRow + 4 * VCoeffStride, _mm512_fnmadd_ps(Two, d31, d42));
        _mm512_storeu_ps(VRow + 5 * VCoeffStride, _mm512_fmadd_ps(Four, w1, _mm512_fnmadd_ps(Five, w3, w5)));
    }
}

void
MlasWinogradOutputTransformAvx512F(
    const MLAS_WINOGRAD_WORK_BLOCK* WorkBlock,
    size_t Tile,
    size_t OutputBlock
    )
/*++

Routine Description:

    Combines the 36 coefficient vectors of one tile for one 16-channel
    output block as A'mA into a 4x4 output tile with bias addition and
    optional ReLU folded in.

--*/
{
    constexpr size_t BlockSize = 16;

    const size_t LocalTile = Tile - WorkBlock->TileBegin;

    const size_t th = Tile / WorkBlock->TilesW;
    const size_t tw = Tile % WorkBlock->TilesW;

    const size_t oh0 = th * 4;
    const size_t ow0 = tw * 4;

    const size_t ValidRows = (WorkBlock->OutputHeight - oh0 < 4) ? WorkBlock->OutputHeight - oh0 : 4;
    const size_t ValidCols = (WorkBlock->OutputWidth - ow0 < 4) ? WorkBlock->OutputWidth - ow0 : 4;

    const float* M = WorkBlock->M + LocalTile * WorkBlock->OutputChannels + OutputBlock * BlockSize;
    const size_t MCoeffStride = WorkBlock->MCoeffStride;

    const __m512 Two = _mm512_set1_ps(2.0f);
    const __m512 Four = _mm512_set1_ps(4.0f);
    const __m512 Eight = _mm512_set1_ps(8.0f);

    //
    // Column pass: p = A'm, reducing the 6 rows to 4.
    //
    //    p0 = m0 + (m1 + m2) + (m3 + m4)
    //    p1 = (m1 - m2) + 2*(m3 - m4)
    //    p2 = (m1 + m2) + 4*(m3 + m4)
    //    p3 = (m1 - m2) + 8*(m3 - m4) + m5
    //

    __m512 p[4][6];

    for (size_t c = 0; c < 6; c++) {

        const float* MColumn = M + c * MCoeffStride;

        const __m512 m0 = _mm512_loadu_ps(MColumn);
        const __m512 m1 = _mm512_loadu_ps(MColumn + 6 * MCoeffStride);
        const __m512 m2 = _mm512_loadu_ps(MColumn + 12 * MCoeffStride);
        const __m512 m3 = _mm512_loadu_ps(MColumn + 18 * MCoeffStride);
        const __m512 m4 = _mm512_loadu_ps(MColumn + 24 * MCoeffStride);
        const __m512 m5 = _mm512_loadu_ps(MColumn + 30 * MCoeffStride);

        const __m512 s12 = _mm512_add_ps(m1, m2);
        const __m512 d12 = _mm512_sub_ps(m1, m2);
        const __m512 s34 = _mm512_add_ps(m3, m4);
        const __m512 d34 = _mm512_sub_ps(m3, m4);

        p[0][c] = _mm512_add_ps(_mm512_add_ps(m0, s12), s34);
        p[1][c] = _mm512_fmadd_ps(Two, d34, d12);
        p[2][c] = _mm512_fmadd_ps(Four, s34, s12);
        p[3][c] = _mm512_add_ps(_mm512_fmadd_ps(Eight, d34, d12), m5);
    }

    //
    // Row pass with bias and optional ReLU, storing the valid portion of the
    // 4x4 output tile.
    //

    const __m512 BiasVector = (WorkBlock->Bias != nullptr)
        ? _mm512_loadu_ps(&WorkBlock->Bias[OutputBlock * BlockSize])
        : _mm512_setzero_ps();

    const __m512 ZeroVector = _mm512_setzero_ps();
    const bool Relu = WorkBlock->Relu;

    float* OutputPlane = WorkBlock->Output +
        OutputBlock * WorkBlock->OutputHeight * WorkBlock->OutputWidth * BlockSize;

    for (size_t r = 0; r < ValidRows; r++) {

        const __m512 p0 = p[r][0], p1 = p[r][1], p2 = p[r][2];
        const __m512 p3 = p[r][3], p4 = p[r][4], p5 = p[r][5];

        const __m512 s12 = _mm512_add_ps(p1, p2);
        const __m512 d12 = _mm512_sub_ps(p1, p2);
        const __m512 s34 = _mm512_add_ps(p3, p4);
        const __m512 d34 = _mm512_sub_ps(p3, p4);

        __m512 y[4];

        y[0] = _mm512_add_ps(_mm512_add_ps(p0, s12), s34);
        y[1] = _mm512_fmadd_ps(Two, d34, d12);
        y[2] = _mm512_fmadd_ps(Four, s34, s12);
        y[3] = _mm512_add_ps(_mm512_fmadd_ps(Eight, d34, d12), p5);

        float* OutputRow = &OutputPlane[((oh0 + r) * WorkBlock->OutputWidth + ow0) * BlockSize];

        for (size_t c = 0; c < ValidCols; c++) {
            __m512 Value = _mm512_add_ps(y[c], BiasVector);
            if (Relu) {
                Value = _mm512_max_ps(Value, ZeroVector);
            }
            _mm512_storeu_ps(OutputRow + c * BlockSize, Value);
        }
    }
}

#endif
