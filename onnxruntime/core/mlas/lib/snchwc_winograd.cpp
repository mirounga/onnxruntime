/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    snchwc_winograd.cpp

Abstract:

    This module implements the single precision Winograd F(4x4,3x3)
    convolution algorithm for the NCHWc blocked layout.

    The convolution is decomposed into three phases:

        1. Input transform: each 6x6 input patch (stride 4, so adjacent
           patches overlap by 2) is transformed as B'dB and scattered into
           36 per-coefficient matrices V[k] of shape [Tiles][InputChannels].

        2. Batched GEMM: for each coefficient k, M[k] = V[k] * U[k], where
           U[k] is the transformed filter of shape [InputChannels][OutputChannels],
           prepacked once with MlasGemmPackB.

        3. Output transform: for each (tile, output channel block), the 36
           coefficient values are combined as A'mA into a 4x4 output tile,
           with bias addition and optional ReLU folded in.

    The 16-channel NCHWc block dimension maps naturally onto vector lanes,
    so all transforms operate on contiguous 16-float groups.

    This algorithm is not bitwise identical to the direct convolution
    algorithm and is only selected when explicitly enabled.

--*/

#include "mlasi.h"

#if defined(MLAS_TARGET_AMD64)

//
// Number of Winograd coefficients (6x6) and output tile dimension.
//

#define MLAS_WINOGRAD_COEFF_COUNT       36
#define MLAS_WINOGRAD_INPUT_TILE        6
#define MLAS_WINOGRAD_OUTPUT_TILE       4

//
// Heuristic thresholds for selecting Winograd over the direct algorithm.
//

static constexpr size_t MlasWinogradMinChannels = 32;
static constexpr size_t MlasWinogradMinSpatial = 8;
static constexpr size_t MlasWinogradMinTiles = 16;

//
// Cap on the per-inference transform workspace. Tile ranges are chunked so
// the V and M buffers stay within this budget.
//

static constexpr size_t MlasWinogradWorkspaceCap = size_t(8) * 1024 * 1024;

bool
MLASCALL
MlasNchwcConvWinogradSupported(
    void
    )
{
    return MlasNchwcGetBlockSize() == 16;
}

size_t
MLASCALL
MlasNchwcConvWinogradFilterTransformSize(
    size_t OutputChannels,
    size_t InputChannels,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    const size_t PackedSize = UpAlignSize(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans,
        OutputChannels, InputChannels, BackendKernelSelectorConfig));

    return MLAS_WINOGRAD_COEFF_COUNT * PackedSize;
}

void
MLASCALL
MlasNchwcConvWinogradFilterTransform(
    size_t OutputChannels,
    size_t InputChannels,
    const float* Filter,
    void* TransformedFilter,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
/*++

Routine Description:

    Transforms a 3x3 filter in OIHWBiBo format to 36 per-coefficient
    matrices U[k] of shape [InputChannels][OutputChannels], each prepacked
    with MlasGemmPackB for use as the B operand of the coefficient GEMMs.

--*/
{
    const size_t BlockSize = MlasNchwcGetBlockSize();

    const size_t InputBlocks = InputChannels / BlockSize;
    const size_t OutputBlocks = OutputChannels / BlockSize;

    //
    // Winograd F(4x4,3x3) filter transform matrix G (6x3).
    //

    static constexpr double G[6][3] = {
        {  1.0 / 4.0,          0.0,        0.0 },
        { -1.0 / 6.0,  -1.0 / 6.0, -1.0 / 6.0 },
        { -1.0 / 6.0,   1.0 / 6.0, -1.0 / 6.0 },
        { 1.0 / 24.0,  1.0 / 12.0,  1.0 / 6.0 },
        { 1.0 / 24.0, -1.0 / 12.0,  1.0 / 6.0 },
        {        0.0,         0.0,        1.0 },
    };

    //
    // Compute U[k][ci][co] = (G g G')[k] into a temporary buffer, then pack
    // each of the 36 [InputChannels][OutputChannels] matrices.
    //

    const size_t UMatrixSize = InputChannels * OutputChannels;

    MlasThreadedBufAlloc(MLAS_WINOGRAD_COEFF_COUNT * UMatrixSize * sizeof(float));

    float* UAll = reinterpret_cast<float*>(ThreadedBufHolder.get());

    for (size_t ob = 0; ob < OutputBlocks; ob++) {

        for (size_t ib = 0; ib < InputBlocks; ib++) {

            const float* FilterBlock = Filter +
                (ob * InputBlocks + ib) * 9 * BlockSize * BlockSize;

            for (size_t il = 0; il < BlockSize; il++) {

                for (size_t ol = 0; ol < BlockSize; ol++) {

                    //
                    // Gather the 3x3 filter for this (input, output) channel
                    // pair from the OIHWBiBo layout.
                    //

                    double g[3][3];

                    for (size_t kh = 0; kh < 3; kh++) {
                        for (size_t kw = 0; kw < 3; kw++) {
                            g[kh][kw] = double(FilterBlock[((kh * 3 + kw) * BlockSize + il) * BlockSize + ol]);
                        }
                    }

                    //
                    // Compute Gg (6x3), then (Gg)G' (6x6). Double precision
                    // intermediates reduce rounding in this one-time step.
                    //

                    double Gg[6][3];

                    for (size_t r = 0; r < 6; r++) {
                        for (size_t c = 0; c < 3; c++) {
                            Gg[r][c] = G[r][0] * g[0][c] + G[r][1] * g[1][c] + G[r][2] * g[2][c];
                        }
                    }

                    const size_t ci = ib * BlockSize + il;
                    const size_t co = ob * BlockSize + ol;

                    for (size_t r = 0; r < 6; r++) {
                        for (size_t c = 0; c < 6; c++) {
                            const double u = Gg[r][0] * G[c][0] + Gg[r][1] * G[c][1] + Gg[r][2] * G[c][2];
                            UAll[(r * 6 + c) * UMatrixSize + ci * OutputChannels + co] = float(u);
                        }
                    }
                }
            }
        }
    }

    //
    // Pack each coefficient matrix as the GEMM B operand.
    //

    const size_t PackedSize = UpAlignSize(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans,
        OutputChannels, InputChannels, BackendKernelSelectorConfig));

    uint8_t* PackedBase = reinterpret_cast<uint8_t*>(TransformedFilter);

    for (size_t k = 0; k < MLAS_WINOGRAD_COEFF_COUNT; k++) {
        MlasGemmPackB(CblasNoTrans, CblasNoTrans, OutputChannels, InputChannels,
            UAll + k * UMatrixSize, OutputChannels,
            PackedBase + k * PackedSize, BackendKernelSelectorConfig);
    }
}

bool
MLASCALL
MlasNchwcConvWinogradEligible(
    const int64_t* InputShape,
    const int64_t* OutputShape,
    const int64_t* Padding
    )
{
    if (!MlasNchwcConvWinogradSupported()) {
        return false;
    }

    const size_t BlockSize = MlasNchwcGetBlockSize();

    const size_t InputChannels = size_t(InputShape[1]);
    const size_t InputHeight = size_t(InputShape[2]);
    const size_t InputWidth = size_t(InputShape[3]);

    const size_t OutputChannels = size_t(OutputShape[1]);
    const size_t OutputHeight = size_t(OutputShape[2]);
    const size_t OutputWidth = size_t(OutputShape[3]);

    if ((InputChannels % BlockSize) != 0 || (OutputChannels % BlockSize) != 0) {
        return false;
    }

    for (size_t i = 0; i < 4; i++) {
        if (Padding[i] < 0 || Padding[i] > 1) {
            return false;
        }
    }

    //
    // Require output dimensions consistent with a 3x3 stride-1 dilation-1
    // convolution of the padded input.
    //

    if (OutputHeight != InputHeight + size_t(Padding[0]) + size_t(Padding[2]) - 2 ||
        OutputWidth != InputWidth + size_t(Padding[1]) + size_t(Padding[3]) - 2) {
        return false;
    }

    //
    // Heuristic: small channel counts or small spatial extents do not
    // amortize the transform overhead, and heavily padded tile grids waste
    // the algorithmic gain.
    //

    if (InputChannels < MlasWinogradMinChannels || OutputChannels < MlasWinogradMinChannels) {
        return false;
    }

    if (OutputHeight < MlasWinogradMinSpatial || OutputWidth < MlasWinogradMinSpatial) {
        return false;
    }

    const size_t TilesH = (OutputHeight + MLAS_WINOGRAD_OUTPUT_TILE - 1) / MLAS_WINOGRAD_OUTPUT_TILE;
    const size_t TilesW = (OutputWidth + MLAS_WINOGRAD_OUTPUT_TILE - 1) / MLAS_WINOGRAD_OUTPUT_TILE;
    const size_t Tiles = TilesH * TilesW;

    if (Tiles < MlasWinogradMinTiles) {
        return false;
    }

    if (16 * Tiles > 2 * OutputHeight * OutputWidth) {
        return false;
    }

    return true;
}

void
MLASCALL
MlasNchwcConvWinograd(
    const int64_t* InputShape,
    const int64_t* Padding,
    const int64_t* OutputShape,
    const float* Input,
    const void* TransformedFilter,
    const float* Bias,
    float* Output,
    const MLAS_ACTIVATION* Activation,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    constexpr size_t BlockSize = 16;

    const size_t BatchCount = size_t(InputShape[0]);
    const size_t InputChannels = size_t(InputShape[1]);
    const size_t InputHeight = size_t(InputShape[2]);
    const size_t InputWidth = size_t(InputShape[3]);

    const size_t OutputChannels = size_t(OutputShape[1]);
    const size_t OutputHeight = size_t(OutputShape[2]);
    const size_t OutputWidth = size_t(OutputShape[3]);

    const size_t InputBlocks = InputChannels / BlockSize;
    const size_t OutputBlocks = OutputChannels / BlockSize;

    const size_t TilesH = (OutputHeight + MLAS_WINOGRAD_OUTPUT_TILE - 1) / MLAS_WINOGRAD_OUTPUT_TILE;
    const size_t TilesW = (OutputWidth + MLAS_WINOGRAD_OUTPUT_TILE - 1) / MLAS_WINOGRAD_OUTPUT_TILE;
    const size_t Tiles = TilesH * TilesW;

    //
    // Chunk the tile range so the transform workspace stays bounded.
    //

    const size_t BytesPerTile = MLAS_WINOGRAD_COEFF_COUNT * (InputChannels + OutputChannels) * sizeof(float);

    size_t MaximumTilesPerChunk = MlasWinogradWorkspaceCap / BytesPerTile;
    MaximumTilesPerChunk = (std::max)(MaximumTilesPerChunk, size_t(64));
    MaximumTilesPerChunk = (std::min)(MaximumTilesPerChunk, Tiles);

    const size_t VCoeffStride = MaximumTilesPerChunk * InputChannels;
    const size_t MCoeffStride = MaximumTilesPerChunk * OutputChannels;

    const size_t VBufferSize = UpAlignSize(MLAS_WINOGRAD_COEFF_COUNT * VCoeffStride * sizeof(float));
    const size_t MBufferSize = UpAlignSize(MLAS_WINOGRAD_COEFF_COUNT * MCoeffStride * sizeof(float));

    MlasThreadedBufAlloc(VBufferSize + MBufferSize);

    float* V = reinterpret_cast<float*>(ThreadedBufHolder.get());
    float* M = reinterpret_cast<float*>(ThreadedBufHolder.get() + VBufferSize);

    const ptrdiff_t tids = MlasGetMaximumThreadCount(ThreadPool);

    const size_t PackedSize = UpAlignSize(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans,
        OutputChannels, InputChannels, BackendKernelSelectorConfig));
    const uint8_t* PackedBase = reinterpret_cast<const uint8_t*>(TransformedFilter);

    const bool Relu = Activation->ActivationKind == MlasReluActivation;

    MLAS_WINOGRAD_WORK_BLOCK WorkBlock;

    WorkBlock.Bias = Bias;
    WorkBlock.V = V;
    WorkBlock.M = M;
    WorkBlock.VCoeffStride = VCoeffStride;
    WorkBlock.MCoeffStride = MCoeffStride;
    WorkBlock.InputChannels = InputChannels;
    WorkBlock.OutputChannels = OutputChannels;
    WorkBlock.InputHeight = InputHeight;
    WorkBlock.InputWidth = InputWidth;
    WorkBlock.OutputHeight = OutputHeight;
    WorkBlock.OutputWidth = OutputWidth;
    WorkBlock.PaddingTop = ptrdiff_t(Padding[0]);
    WorkBlock.PaddingLeft = ptrdiff_t(Padding[1]);
    WorkBlock.TilesW = TilesW;
    WorkBlock.Relu = Relu;

    for (size_t batch = 0; batch < BatchCount; batch++) {

        WorkBlock.Input = Input + batch * InputChannels * InputHeight * InputWidth;
        WorkBlock.Output = Output + batch * OutputChannels * OutputHeight * OutputWidth;

        for (size_t TileBegin = 0; TileBegin < Tiles; TileBegin += MaximumTilesPerChunk) {

            const size_t TileCount = (std::min)(MaximumTilesPerChunk, Tiles - TileBegin);

            WorkBlock.TileBegin = TileBegin;
            WorkBlock.TileCount = TileCount;

            //
            // Phase 1: input transform over (tile, input block) pairs.
            //

            const size_t InputUnits = TileCount * InputBlocks;
            const ptrdiff_t InputWorkers = (std::min)(ptrdiff_t(InputUnits), tids);

            MlasTrySimpleParallel(ThreadPool, InputWorkers,
                [&](ptrdiff_t Index) {
                    size_t WorkIndex;
                    size_t WorkRemaining;
                    MlasPartitionWork(Index, InputWorkers, InputUnits, &WorkIndex, &WorkRemaining);
                    while (WorkRemaining-- > 0) {
                        const size_t Tile = TileBegin + (WorkIndex / InputBlocks);
                        const size_t InputBlock = WorkIndex % InputBlocks;
                        MlasWinogradInputTransformAvx512F(&WorkBlock, Tile, InputBlock);
                        WorkIndex++;
                    }
                });

            //
            // Phase 2: batched per-coefficient GEMMs.
            //

            MLAS_SGEMM_DATA_PARAMS GemmParams[MLAS_WINOGRAD_COEFF_COUNT];

            for (size_t k = 0; k < MLAS_WINOGRAD_COEFF_COUNT; k++) {
                GemmParams[k].A = V + k * VCoeffStride;
                GemmParams[k].lda = InputChannels;
                GemmParams[k].B = reinterpret_cast<const float*>(PackedBase + k * PackedSize);
                GemmParams[k].ldb = 0;
                GemmParams[k].BIsPacked = true;
                GemmParams[k].C = M + k * MCoeffStride;
                GemmParams[k].ldc = OutputChannels;
                GemmParams[k].alpha = 1.0f;
                GemmParams[k].beta = 0.0f;
            }

            MlasGemmBatch(CblasNoTrans, CblasTrans, TileCount, OutputChannels, InputChannels,
                GemmParams, MLAS_WINOGRAD_COEFF_COUNT, ThreadPool, BackendKernelSelectorConfig);

            //
            // Phase 3: output transform over (tile, output block) pairs.
            //

            const size_t OutputUnits = TileCount * OutputBlocks;
            const ptrdiff_t OutputWorkers = (std::min)(ptrdiff_t(OutputUnits), tids);

            MlasTrySimpleParallel(ThreadPool, OutputWorkers,
                [&](ptrdiff_t Index) {
                    size_t WorkIndex;
                    size_t WorkRemaining;
                    MlasPartitionWork(Index, OutputWorkers, OutputUnits, &WorkIndex, &WorkRemaining);
                    while (WorkRemaining-- > 0) {
                        const size_t Tile = TileBegin + (WorkIndex / OutputBlocks);
                        const size_t OutputBlock = WorkIndex % OutputBlocks;
                        MlasWinogradOutputTransformAvx512F(&WorkBlock, Tile, OutputBlock);
                        WorkIndex++;
                    }
                });
        }

        //
        // Apply non-ReLU activations as an elementwise post-pass.
        //

        if (Activation->ActivationKind != MlasIdentityActivation &&
            Activation->ActivationKind != MlasReluActivation) {
            const size_t OutputSize = OutputChannels * OutputHeight * OutputWidth;
            MlasActivation(Activation, WorkBlock.Output, nullptr, 1, OutputSize, OutputSize);
        }
    }
}

#else

//
// Stub implementations for targets without NCHWc Winograd support.
//

bool
MLASCALL
MlasNchwcConvWinogradSupported(
    void
    )
{
    return false;
}

size_t
MLASCALL
MlasNchwcConvWinogradFilterTransformSize(
    size_t OutputChannels,
    size_t InputChannels,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    MLAS_UNREFERENCED_PARAMETER(OutputChannels);
    MLAS_UNREFERENCED_PARAMETER(InputChannels);
    MLAS_UNREFERENCED_PARAMETER(BackendKernelSelectorConfig);
    return 0;
}

void
MLASCALL
MlasNchwcConvWinogradFilterTransform(
    size_t OutputChannels,
    size_t InputChannels,
    const float* Filter,
    void* TransformedFilter,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    MLAS_UNREFERENCED_PARAMETER(OutputChannels);
    MLAS_UNREFERENCED_PARAMETER(InputChannels);
    MLAS_UNREFERENCED_PARAMETER(Filter);
    MLAS_UNREFERENCED_PARAMETER(TransformedFilter);
    MLAS_UNREFERENCED_PARAMETER(BackendKernelSelectorConfig);
}

bool
MLASCALL
MlasNchwcConvWinogradEligible(
    const int64_t* InputShape,
    const int64_t* OutputShape,
    const int64_t* Padding
    )
{
    MLAS_UNREFERENCED_PARAMETER(InputShape);
    MLAS_UNREFERENCED_PARAMETER(OutputShape);
    MLAS_UNREFERENCED_PARAMETER(Padding);
    return false;
}

void
MLASCALL
MlasNchwcConvWinograd(
    const int64_t* InputShape,
    const int64_t* Padding,
    const int64_t* OutputShape,
    const float* Input,
    const void* TransformedFilter,
    const float* Bias,
    float* Output,
    const MLAS_ACTIVATION* Activation,
    MLAS_THREADPOOL* ThreadPool,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* BackendKernelSelectorConfig
    )
{
    MLAS_UNREFERENCED_PARAMETER(InputShape);
    MLAS_UNREFERENCED_PARAMETER(Padding);
    MLAS_UNREFERENCED_PARAMETER(OutputShape);
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(TransformedFilter);
    MLAS_UNREFERENCED_PARAMETER(Bias);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(Activation);
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);
    MLAS_UNREFERENCED_PARAMETER(BackendKernelSelectorConfig);
}

#endif
