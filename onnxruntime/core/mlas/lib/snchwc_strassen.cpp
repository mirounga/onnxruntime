/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    snchwc_strassen.cpp

Abstract:

    This module implements a fused one-level Strassen algorithm for NCHWc
    pointwise (1x1, stride 1, no padding) convolutions.

    The GEMM view of a pointwise convolution, C[HW x Cout] = A[HW x Cin] x
    B[Cin x Cout], is split into 2x2 quadrants (spatial halves x channel
    halves) and computed with the classic seven-product Strassen schedule:

        P1 = (A11 + A22) (B11 + B22)      -> C11 (+), C22 (+)
        P2 = (A21 + A22)  B11             -> C21 (+), C22 (-)
        P3 =  A11        (B12 - B22)      -> C12 (+), C22 (+)
        P4 =  A22        (B21 - B11)      -> C11 (+), C21 (+)
        P5 = (A11 + A12)  B22             -> C12 (+), C11 (-)
        P6 = (A21 - A11) (B11 + B12)      -> C22 (+)
        P7 = (A12 - A22) (B21 + B22)      -> C11 (+)

    Two structural decisions keep the memory traffic close to the direct
    algorithm's:

    1. Weight combinations are formed transiently. The original OIHWBiBo
       filter remains the only long lived representation; each combination is
       packed into thread local scratch for one panel of output blocks and is
       reused across that panel's entire spatial range. Forming a combination
       costs O(K*N) additions, which measurement puts at a fraction of a
       percent of the panel's multiply-accumulate work, so paying it per
       inference is far cheaper than carrying a permanently expanded weight
       set through the cache hierarchy.

    2. Products are ordered innermost. For each small spatial chunk the seven
       products run back to back, so the four activation quadrant tiles and
       the four output sub-tiles stay resident in the first level cache while
       the twelve output updates of the schedule are applied to them. Ordering
       the products outermost instead would push those updates out to memory.

    Activation side two term sums fold into the kernel's broadcast loads and
    the output combines fold into its epilogue, so no standalone addition
    passes over memory occur at any point.

    Results are numerically close to, but not bitwise identical with, the
    direct algorithm. The algorithm is only selected when explicitly enabled.

--*/

#include "mlasi.h"

#if defined(MLAS_TARGET_AMD64)

//
// Eligibility floors. Quadrant halves must stay block aligned, and the
// spatial extent must be large enough to amortize packing the weight
// combinations.
//

static constexpr size_t MlasStrassenMinChannels = 64;
static constexpr size_t MlasStrassenMinSpatial = 128;

//
// Spatial positions processed between kernel invocations. Sized so that the
// activation quadrant tiles and output sub-tiles a chunk touches remain in
// the first level cache across all seven products, while still amortizing
// the per invocation overhead. A multiple of the kernel's six wide unrolling.
//

static constexpr size_t MlasStrassenSpatialChunk = 12;

//
// Budget for the seven weight combination panels held in thread local
// scratch. Kept below the per core second level cache so the panels stay
// resident while a panel's spatial range is processed.
//

static constexpr size_t MlasStrassenPanelBudgetBytes = 512 * 1024;

bool
MLASCALL
MlasNchwcConvStrassenSupported(
    void
    )
{
    return MlasNchwcGetBlockSize() == 16;
}

bool
MLASCALL
MlasNchwcConvStrassenEligible(
    const int64_t* InputShape,
    const int64_t* OutputShape
    )
{
    if (!MlasNchwcConvStrassenSupported()) {
        return false;
    }

    const size_t InputChannels = size_t(InputShape[1]);
    const size_t OutputChannels = size_t(OutputShape[1]);

    if (InputShape[2] != OutputShape[2] || InputShape[3] != OutputShape[3]) {
        return false;
    }

    const size_t SpatialSize = size_t(InputShape[2]) * size_t(InputShape[3]);

    if ((InputChannels % 32) != 0 || (OutputChannels % 32) != 0) {
        return false;
    }

    if ((SpatialSize % 2) != 0) {
        return false;
    }

    if (InputChannels < MlasStrassenMinChannels ||
        OutputChannels < MlasStrassenMinChannels ||
        SpatialSize < MlasStrassenMinSpatial) {
        return false;
    }

    return true;
}

//
// Weight side operands of the classic schedule. Quadrant indices are
// (input half, output half); B11 is the first input half by the first output
// half. A zero second scale marks a single quadrant operand.
//
//    P1: B11 + B22    P2: B11          P3: B12 - B22    P4: B21 - B11
//    P5: B22          P6: B11 + B12    P7: B21 + B22
//

struct MLAS_STRASSEN_COMBO {
    size_t InHalfA;
    size_t OutHalfA;
    float Scale2;
    size_t InHalfB;
    size_t OutHalfB;
};

static const MLAS_STRASSEN_COMBO MlasStrassenCombos[7] = {
    {0, 0, +1.0f, 1, 1},   // P1
    {0, 0,  0.0f, 0, 0},   // P2
    {0, 1, -1.0f, 1, 1},   // P3
    {1, 0, -1.0f, 0, 0},   // P4
    {1, 1,  0.0f, 0, 0},   // P5
    {0, 0, +1.0f, 0, 1},   // P6
    {1, 0, +1.0f, 1, 1},   // P7
};

//
// Per product runtime descriptors for the classic schedule.
//
//  A side operand: quadrants (spatial half, input half); optional second with
//  sign. C side: one or two destinations (spatial half, output half) with
//  signs; Last1/Last2 mark the final update of a C quadrant, which is where
//  bias and any fused ReLU are applied.
//

struct MLAS_STRASSEN_PRODUCT {
    size_t Am1, Ai1;
    int HasA2;
    int SubA;
    size_t Am2, Ai2;
    size_t Cm1, Co1;
    int Sub1;
    int Acc1;
    int Last1;
    int HasC2;
    size_t Cm2, Co2;
    int Sub2;
    int Acc2;
    int Last2;
};

static const MLAS_STRASSEN_PRODUCT MlasStrassenSchedule[7] = {
    // P1 = (A11+A22)(B11+B22) -> C11 store, C22 store
    {0, 0, 1, 0, 1, 1,  0, 0, 0, 0, 0,  1, 1, 1, 0, 0, 0},
    // P2 = (A21+A22) B11      -> C21 store, C22 subtract accumulate
    {1, 0, 1, 0, 1, 1,  1, 0, 0, 0, 0,  1, 1, 1, 1, 1, 0},
    // P3 = A11 (B12-B22)      -> C12 store, C22 accumulate
    {0, 0, 0, 0, 0, 0,  0, 1, 0, 0, 0,  1, 1, 1, 0, 1, 0},
    // P4 = A22 (B21-B11)      -> C11 accumulate, C21 accumulate (last)
    {1, 1, 0, 0, 0, 0,  0, 0, 0, 1, 0,  1, 1, 0, 0, 1, 1},
    // P5 = (A11+A12) B22      -> C12 accumulate (last), C11 subtract accumulate
    {0, 0, 1, 0, 0, 1,  0, 1, 0, 1, 1,  1, 0, 0, 1, 1, 0},
    // P6 = (A21-A11)(B11+B12) -> C22 accumulate (last)
    {1, 0, 1, 1, 0, 0,  1, 1, 0, 1, 1,  0, 0, 0, 0, 0, 0},
    // P7 = (A12-A22)(B21+B22) -> C11 accumulate (last)
    {0, 1, 1, 1, 1, 1,  0, 0, 0, 1, 1,  0, 0, 0, 0, 0, 0},
};

struct MLAS_STRASSEN_WORK_BLOCK {
    const float* Input;
    const float* Filter;            // original OIHWBiBo weights
    const float* Bias;
    float* Output;
    size_t SpatialSize;
    size_t HalfSpatial;
    size_t InputChannels;
    size_t OutputChannels;
    size_t InputBlocks;             // full Cin / BlockSize, for source indexing
    size_t HalfInBlocks;
    size_t HalfOutBlocks;
    size_t PanelFilterCount;        // output blocks per panel, at most 4
    size_t PanelCount;
    size_t PanelFloats;             // floats in one product's panel
    size_t FilterStrideElements;
    bool Relu;
};

//
// Packs one weight combination for a panel of output blocks into the layout
// the kernel reads: [FilterIndex][InputBlock][InputLane][16 output lanes],
// with FilterStrideElements between filter indices.
//

static
void
MlasStrassenPackPanel(
    const MLAS_STRASSEN_WORK_BLOCK* wb,
    size_t PanelBase,
    size_t FilterCount,
    size_t Product,
    float* Panel
    )
{
    constexpr size_t BlockSize = 16;
    constexpr size_t BlockElements = BlockSize * BlockSize;

    const MLAS_STRASSEN_COMBO& combo = MlasStrassenCombos[Product];

    for (size_t fi = 0; fi < FilterCount; fi++) {

        const size_t obA = combo.OutHalfA * wb->HalfOutBlocks + PanelBase + fi;
        const size_t obB = combo.OutHalfB * wb->HalfOutBlocks + PanelBase + fi;

        float* dst = Panel + fi * wb->FilterStrideElements;

        for (size_t cb = 0; cb < wb->HalfInBlocks; cb++) {

            const size_t ibA = combo.InHalfA * wb->HalfInBlocks + cb;
            const float* srcA = wb->Filter + (obA * wb->InputBlocks + ibA) * BlockElements;

            if (combo.Scale2 == 0.0f) {
                memcpy(dst, srcA, BlockElements * sizeof(float));
            } else {
                const size_t ibB = combo.InHalfB * wb->HalfInBlocks + cb;
                const float* srcB = wb->Filter + (obB * wb->InputBlocks + ibB) * BlockElements;
                if (combo.Scale2 > 0.0f) {
                    for (size_t i = 0; i < BlockElements; i++) {
                        dst[i] = srcA[i] + srcB[i];
                    }
                } else {
                    for (size_t i = 0; i < BlockElements; i++) {
                        dst[i] = srcA[i] - srcB[i];
                    }
                }
            }

            dst += BlockElements;
        }
    }
}

//
// Runs the seven products over one spatial chunk for one panel of output
// blocks. The activation quadrant tiles and output sub-tiles this touches are
// small enough to stay in the first level cache for the duration.
//

static
void
MlasStrassenProcessChunk(
    const MLAS_STRASSEN_WORK_BLOCK* wb,
    const float* Scratch,
    size_t PanelBase,
    size_t FilterCount,
    size_t Pos,
    size_t Count
    )
{
    constexpr size_t BlockSize = 16;

    const size_t PlaneStride = wb->SpatialSize * BlockSize;

    for (size_t p = 0; p < 7; p++) {

        const MLAS_STRASSEN_PRODUCT& s = MlasStrassenSchedule[p];

        MLAS_STRASSEN_POINTWISE_PARAMS params;

        auto InputQuadrant = [&](size_t mh, size_t ih) -> const float* {
            return wb->Input + (ih * wb->HalfInBlocks) * PlaneStride +
                   (mh * wb->HalfSpatial + Pos) * BlockSize;
        };
        auto OutputQuadrant = [&](size_t mh, size_t oh) -> float* {
            return wb->Output + (oh * wb->HalfOutBlocks + PanelBase) * PlaneStride +
                   (mh * wb->HalfSpatial + Pos) * BlockSize;
        };
        auto BiasSlice = [&](size_t oh) -> const float* {
            if (wb->Bias == nullptr) {
                return nullptr;
            }
            return wb->Bias + (oh * wb->HalfOutBlocks + PanelBase) * BlockSize;
        };

        params.InputA = InputQuadrant(s.Am1, s.Ai1);
        params.InputB = s.HasA2 ? InputQuadrant(s.Am2, s.Ai2) : nullptr;
        params.SubtractInput = s.SubA != 0;
        params.Filter = Scratch + p * wb->PanelFloats;

        params.Output1 = OutputQuadrant(s.Cm1, s.Co1);
        params.Subtract1 = s.Sub1 != 0;
        params.Accumulate1 = s.Acc1 != 0;
        params.Bias1 = s.Last1 ? BiasSlice(s.Co1) : nullptr;
        params.Relu1 = s.Last1 && wb->Relu;

        if (s.HasC2) {
            params.Output2 = OutputQuadrant(s.Cm2, s.Co2);
            params.Subtract2 = s.Sub2 != 0;
            params.Accumulate2 = s.Acc2 != 0;
            params.Bias2 = s.Last2 ? BiasSlice(s.Co2) : nullptr;
            params.Relu2 = s.Last2 && wb->Relu;
        } else {
            params.Output2 = nullptr;
            params.Subtract2 = false;
            params.Accumulate2 = false;
            params.Bias2 = nullptr;
            params.Relu2 = false;
        }

        params.InputChannelBlocks = wb->HalfInBlocks;
        params.FilterCount = FilterCount;
        params.InputStrideElements = PlaneStride;
        params.OutputStrideElements = PlaneStride;
        params.FilterStrideElements = wb->FilterStrideElements;
        params.OutputCount = Count;

        MlasStrassenPointwiseKernelAvx512F(&params);
    }
}

//
// Processes a range of panels over a range of spatial positions: pack the
// seven combinations for a panel once, then sweep the spatial range with the
// products innermost.
//

static
void
MlasStrassenExecuteRange(
    const MLAS_STRASSEN_WORK_BLOCK* wb,
    float* Scratch,
    size_t PanelBegin,
    size_t PanelEnd,
    size_t PosBegin,
    size_t PosEnd
    )
{
    for (size_t panel = PanelBegin; panel < PanelEnd; panel++) {

        const size_t PanelBase = panel * wb->PanelFilterCount;
        const size_t FilterCount =
            (std::min)(wb->PanelFilterCount, wb->HalfOutBlocks - PanelBase);

        for (size_t p = 0; p < 7; p++) {
            MlasStrassenPackPanel(wb, PanelBase, FilterCount, p,
                                  Scratch + p * wb->PanelFloats);
        }

        for (size_t pos = PosBegin; pos < PosEnd; pos += MlasStrassenSpatialChunk) {
            const size_t Count = (std::min)(MlasStrassenSpatialChunk, PosEnd - pos);
            MlasStrassenProcessChunk(wb, Scratch, PanelBase, FilterCount, pos, Count);
        }
    }
}

void
MLASCALL
MlasNchwcConvStrassen(
    const int64_t* InputShape,
    const int64_t* OutputShape,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* Output,
    const MLAS_ACTIVATION* Activation,
    MLAS_THREADPOOL* ThreadPool
    )
{
    constexpr size_t BlockSize = 16;

    const size_t BatchCount = size_t(InputShape[0]);
    const size_t InputChannels = size_t(InputShape[1]);
    const size_t OutputChannels = size_t(OutputShape[1]);
    const size_t SpatialSize = size_t(InputShape[2]) * size_t(InputShape[3]);

    MLAS_STRASSEN_WORK_BLOCK wb;

    wb.Filter = Filter;
    wb.Bias = Bias;
    wb.SpatialSize = SpatialSize;
    wb.HalfSpatial = SpatialSize / 2;
    wb.InputChannels = InputChannels;
    wb.OutputChannels = OutputChannels;
    wb.InputBlocks = InputChannels / BlockSize;
    wb.HalfInBlocks = InputChannels / 2 / BlockSize;
    wb.HalfOutBlocks = OutputChannels / 2 / BlockSize;
    wb.FilterStrideElements = wb.HalfInBlocks * BlockSize * BlockSize;
    wb.Relu = Activation->ActivationKind == MlasReluActivation;

    const ptrdiff_t tids = MlasGetMaximumThreadCount(ThreadPool);

    //
    // Choose the panel width and the partitioning axis together, by relative
    // cost. Three effects trade off:
    //
    //   Kernel width. A panel of four output blocks fills the kernel's
    //   register tile; narrower panels retire fewer accumulators per operand
    //   load, and a single block falls back to the embedded broadcast form,
    //   which is far weaker.
    //
    //   Load balance. A panel is the unit of work when panels are the
    //   partitioning axis, so a panel count that does not divide across the
    //   threads leaves some of them idle for a whole round.
    //
    //   Packing. Forming the seven combinations for a panel is O(K*N) of
    //   read-modify-write traffic amortized over the spatial extent the panel
    //   is reused across, which per multiply-accumulate is (threads sharing
    //   the panel) / HalfSpatial. Partitioning panels packs each exactly once;
    //   partitioning spatially makes every thread pack every panel.
    //
    // The packing weight below is the measured cost of a packed float
    // relative to a multiply-accumulate; the kernel width penalties are
    // likewise measured relative to a full width panel.
    //

    static constexpr double PackWeight = 10.0;
    static const double KernelPenalty[5] = {0.0, 1.60, 1.20, 1.05, 1.00};

    const size_t PerFilterFloats = 7 * wb.FilterStrideElements;

    size_t WidestPanel = MlasStrassenPanelBudgetBytes / (PerFilterFloats * sizeof(float));
    WidestPanel = (std::max)(WidestPanel, size_t(1));
    WidestPanel = (std::min)(WidestPanel, size_t(4));
    WidestPanel = (std::min)(WidestPanel, wb.HalfOutBlocks);

    //
    // Spatial split, always available: the widest panel, perfectly balanced,
    // but every thread packs every panel.
    //

    bool SpatialParallel = true;
    size_t PanelFilterCount = WidestPanel;
    size_t PanelCount = (wb.HalfOutBlocks + WidestPanel - 1) / WidestPanel;

    double BestCost = KernelPenalty[WidestPanel] *
        (1.0 + PackWeight * double(tids) / double(wb.HalfSpatial));

    for (size_t fc = WidestPanel; fc >= 1; fc--) {

        const size_t Panels = (wb.HalfOutBlocks + fc - 1) / fc;

        if (Panels < size_t(tids)) {
            continue;
        }

        const size_t Rounds = (Panels + size_t(tids) - 1) / size_t(tids);
        const double Imbalance = double(Rounds * size_t(tids)) / double(Panels);

        const double Cost = Imbalance * KernelPenalty[fc] *
            (1.0 + PackWeight / double(wb.HalfSpatial));

        if (Cost < BestCost - 1.0e-9) {
            BestCost = Cost;
            SpatialParallel = false;
            PanelFilterCount = fc;
            PanelCount = Panels;
        }
    }

    wb.PanelFilterCount = PanelFilterCount;
    wb.PanelCount = PanelCount;
    wb.PanelFloats = PanelFilterCount * wb.FilterStrideElements;

    //
    // Scratch for the seven panels. Eligibility bounds every factor well
    // below the point of overflow, but the arithmetic is checked because this
    // is reachable from a public entry point.
    //

    const size_t ScratchFloats = 7 * wb.PanelFloats;

    if (wb.PanelFloats == 0 || ScratchFloats / 7 != wb.PanelFloats ||
        ScratchFloats > (~size_t(0)) / sizeof(float)) {
        return;
    }


    for (size_t batch = 0; batch < BatchCount; batch++) {

        wb.Input = Input + batch * InputChannels * SpatialSize;
        wb.Output = Output + batch * OutputChannels * SpatialSize;

        MlasTrySimpleParallel(ThreadPool, tids,
            [&](ptrdiff_t Index) {

                MlasThreadedBufAlloc(ScratchFloats * sizeof(float));
                float* Scratch = reinterpret_cast<float*>(ThreadedBufHolder.get());

                size_t Begin;
                size_t Count;

                if (SpatialParallel) {

                    MlasPartitionWork(Index, tids, wb.HalfSpatial, &Begin, &Count);

                    if (Count == 0) {
                        return;
                    }

                    MlasStrassenExecuteRange(&wb, Scratch, 0, wb.PanelCount,
                                             Begin, Begin + Count);

                } else {

                    MlasPartitionWork(Index, tids, wb.PanelCount, &Begin, &Count);

                    if (Count == 0) {
                        return;
                    }

                    MlasStrassenExecuteRange(&wb, Scratch, Begin, Begin + Count,
                                             0, wb.HalfSpatial);
                }
            });

        if (Activation->ActivationKind != MlasIdentityActivation &&
            Activation->ActivationKind != MlasReluActivation) {
            const size_t OutputSize = OutputChannels * SpatialSize;
            MlasActivation(Activation, wb.Output, nullptr, 1, OutputSize, OutputSize);
        }
    }
}

#else

//
// Stubs for targets without Strassen support.
//

bool
MLASCALL
MlasNchwcConvStrassenSupported(
    void
    )
{
    return false;
}

bool
MLASCALL
MlasNchwcConvStrassenEligible(
    const int64_t* InputShape,
    const int64_t* OutputShape
    )
{
    MLAS_UNREFERENCED_PARAMETER(InputShape);
    MLAS_UNREFERENCED_PARAMETER(OutputShape);
    return false;
}

void
MLASCALL
MlasNchwcConvStrassen(
    const int64_t* InputShape,
    const int64_t* OutputShape,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* Output,
    const MLAS_ACTIVATION* Activation,
    MLAS_THREADPOOL* ThreadPool
    )
{
    MLAS_UNREFERENCED_PARAMETER(InputShape);
    MLAS_UNREFERENCED_PARAMETER(OutputShape);
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Filter);
    MLAS_UNREFERENCED_PARAMETER(Bias);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(Activation);
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);
}

#endif
