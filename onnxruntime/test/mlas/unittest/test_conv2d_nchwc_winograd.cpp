// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// Tests for the NCHWc Winograd F(4x4,3x3) convolution algorithm. Winograd
// arithmetic is not bitwise identical to direct convolution, so results are
// compared against the im2col+GEMM reference with a relative tolerance
// (CloseEnough) instead of the exact-match harness used by the direct tests.
//

#include "test_conv2d.h"

#include <memory>
#include <vector>

template <bool Threaded>
class MlasNchwcConvWinogradTest : public MlasConv2DTest<Threaded> {
 protected:
  MLAS_ACTIVATION_KIND ActivationKind_ = MlasIdentityActivation;

  void MlasConv2D(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t PaddingRightHeight,
      size_t PaddingRightWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth,
      size_t OutputHeight,
      size_t OutputWidth,
      const float* Input,
      const float* Filter,
      const float* Bias,
      float* Output) override {
    MLAS_UNREFERENCED_PARAMETER(GroupCount);
    MLAS_UNREFERENCED_PARAMETER(KernelHeight);
    MLAS_UNREFERENCED_PARAMETER(KernelWidth);
    MLAS_UNREFERENCED_PARAMETER(DilationHeight);
    MLAS_UNREFERENCED_PARAMETER(DilationWidth);
    MLAS_UNREFERENCED_PARAMETER(StrideHeight);
    MLAS_UNREFERENCED_PARAMETER(StrideWidth);

    int64_t InputShape[] = {int64_t(BatchCount), int64_t(InputChannels), int64_t(InputHeight), int64_t(InputWidth)};
    int64_t FilterShape[] = {int64_t(FilterCount), int64_t(InputChannels), 3, 3};
    int64_t OutputShape[] = {int64_t(BatchCount), int64_t(FilterCount), int64_t(OutputHeight), int64_t(OutputWidth)};
    int64_t Padding[] = {int64_t(PaddingLeftHeight), int64_t(PaddingLeftWidth), int64_t(PaddingRightHeight), int64_t(PaddingRightWidth)};

    //
    // Reorder the input and filter into the blocked formats.
    //

    size_t NchwcInputElements = BatchCount * InputChannels * InputHeight * InputWidth;
    float* NchwcInput = BufferNchwcInput.GetBuffer(NchwcInputElements);
    ReorderInputNchw(InputShape, Input, NchwcInput);

    size_t FilterElements = FilterCount * InputChannels * 9;
    float* ReorderedFilter = BufferNchwcFilter.GetBuffer(FilterElements);
    MlasReorderFilterOIHWBiBo(FilterShape, Filter, ReorderedFilter);

    //
    // Transform and pack the filter.
    //

    const size_t TransformedFilterSize =
        MlasNchwcConvWinogradFilterTransformSize(FilterCount, InputChannels, nullptr);

    TransformedFilterBuffer_.resize(TransformedFilterSize + 64);
    void* TransformedFilter = reinterpret_cast<void*>(
        (reinterpret_cast<uintptr_t>(TransformedFilterBuffer_.data()) + 63) & ~uintptr_t(63));

    MlasNchwcConvWinogradFilterTransform(FilterCount, InputChannels, ReorderedFilter,
                                         TransformedFilter, nullptr);

    //
    // Run the Winograd convolution and reorder the output back to NCHW.
    //

    size_t NchwcOutputElements = BatchCount * FilterCount * OutputHeight * OutputWidth;
    float* NchwcOutput = BufferNchwcOutput.GetBuffer(NchwcOutputElements);

    MLAS_ACTIVATION Activation;
    Activation.ActivationKind = ActivationKind_;
    if (ActivationKind_ == MlasHardSigmoidActivation) {
      Activation.Parameters.HardSigmoid.alpha = 0.2f;
      Activation.Parameters.HardSigmoid.beta = 0.5f;
    }

    MlasNchwcConvWinograd(InputShape,
                          Padding,
                          OutputShape,
                          NchwcInput,
                          TransformedFilter,
                          Bias,
                          NchwcOutput,
                          &Activation,
                          MlasConv2DTest<Threaded>::threadpool_,
                          nullptr);

    MlasReorderOutputNchw(OutputShape, NchwcOutput, Output, MlasConv2DTest<Threaded>::threadpool_);
  }

  void TestWinograd(
      size_t BatchCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t PaddingHeight,
      size_t PaddingWidth,
      bool WithBias,
      MLAS_ACTIVATION_KIND ActivationKind) {
    const size_t OutputHeight = InputHeight + 2 * PaddingHeight - 2;
    const size_t OutputWidth = InputWidth + 2 * PaddingWidth - 2;
    const size_t OutputElements = BatchCount * FilterCount * OutputHeight * OutputWidth;

    const float* Input = MlasConv2DTest<Threaded>::BufferInput.GetBuffer(BatchCount * InputChannels * InputHeight * InputWidth);
    const float* Filter = MlasConv2DTest<Threaded>::BufferFilter.GetBuffer(FilterCount * InputChannels * 9);
    const float* Bias = WithBias ? MlasConv2DTest<Threaded>::BufferBias.GetBuffer(FilterCount) : nullptr;
    float* Output = MlasConv2DTest<Threaded>::BufferOutput.GetBuffer(OutputElements);
    float* OutputRef = MlasConv2DTest<Threaded>::BufferOutputReference.GetBuffer(OutputElements);

    ActivationKind_ = ActivationKind;

    this->MlasConv2D(BatchCount, 1, InputChannels, InputHeight, InputWidth, FilterCount,
                     3, 3, PaddingHeight, PaddingWidth, PaddingHeight, PaddingWidth,
                     1, 1, 1, 1, OutputHeight, OutputWidth, Input, Filter, Bias, Output);

    MLAS_ACTIVATION Activation;
    Activation.ActivationKind = ActivationKind;
    if (ActivationKind == MlasHardSigmoidActivation) {
      Activation.Parameters.HardSigmoid.alpha = 0.2f;
      Activation.Parameters.HardSigmoid.beta = 0.5f;
    }

    //
    // The reference implementation dereferences the bias unconditionally, so
    // substitute zeros when testing the no-bias path.
    //

    const float* ReferenceBias = Bias;
    if (ReferenceBias == nullptr) {
      ZeroBias_.assign(FilterCount, 0.0f);
      ReferenceBias = ZeroBias_.data();
    }

    //
    // The reference path only implements identity and ReLU; apply other
    // elementwise activations to the reference output manually.
    //

    MLAS_ACTIVATION ReferenceActivation = Activation;
    if (ActivationKind != MlasIdentityActivation && ActivationKind != MlasReluActivation) {
      ReferenceActivation.ActivationKind = MlasIdentityActivation;
    }

    MlasConv2DTest<Threaded>::ReferenceConv2DWithOptions(
        BatchCount, 1, InputChannels, InputHeight, InputWidth, FilterCount,
        3, 3, PaddingHeight, PaddingWidth, 1, 1, 1, 1, OutputHeight, OutputWidth,
        Input, Filter, ReferenceBias, ReferenceActivation, 0.0f, nullptr, OutputRef);

    if (ActivationKind == MlasHardSigmoidActivation) {
      const float alpha = Activation.Parameters.HardSigmoid.alpha;
      const float beta = Activation.Parameters.HardSigmoid.beta;
      for (size_t i = 0; i < OutputElements; i++) {
        OutputRef[i] = std::min(std::max(alpha * OutputRef[i] + beta, 0.0f), 1.0f);
      }
    }

    for (size_t i = 0; i < OutputElements; i++) {
      ASSERT_TRUE(CloseEnough(Output[i], OutputRef[i]))
          << " @" << i << " got " << Output[i] << " expected " << OutputRef[i]
          << " shape " << InputChannels << "->" << FilterCount
          << " " << InputHeight << "x" << InputWidth
          << " pad " << PaddingHeight << "," << PaddingWidth
          << " bias " << WithBias << " act " << int(ActivationKind);
    }
  }

  MatrixGuardBuffer<float> BufferNchwcInput;
  MatrixGuardBuffer<float> BufferNchwcFilter;
  MatrixGuardBuffer<float> BufferNchwcOutput;
  std::vector<uint8_t> TransformedFilterBuffer_;
  std::vector<float> ZeroBias_;

 public:
  static const char* GetTestSuiteName(void) {
    static const std::string suite_name(Threaded ? "Conv2dNchwcWinograd_Threaded" : "Conv2dNchwcWinograd_SingleThread");
    return suite_name.c_str();
  }

  MlasNchwcConvWinogradTest() : MlasConv2DTest<Threaded>() {}

  void ExecuteShort(void) {
    //
    // Eligibility predicate checks: rejected shapes.
    //

    {
      // Spatial too small.
      int64_t InputShape[] = {1, 96, 5, 5};
      int64_t OutputShape[] = {1, 96, 5, 5};
      int64_t Padding[] = {1, 1, 1, 1};
      ASSERT_FALSE(MlasNchwcConvWinogradEligible(InputShape, OutputShape, Padding));
    }
    {
      // Channels below heuristic floor.
      int64_t InputShape[] = {1, 16, 52, 52};
      int64_t OutputShape[] = {1, 32, 52, 52};
      int64_t Padding[] = {1, 1, 1, 1};
      ASSERT_FALSE(MlasNchwcConvWinogradEligible(InputShape, OutputShape, Padding));
    }
    {
      // Output shape inconsistent with 3x3 stride-1 (models stride 2).
      int64_t InputShape[] = {1, 96, 52, 52};
      int64_t OutputShape[] = {1, 96, 26, 26};
      int64_t Padding[] = {1, 1, 1, 1};
      ASSERT_FALSE(MlasNchwcConvWinogradEligible(InputShape, OutputShape, Padding));
    }
    {
      // Eligible reference case (yolox hot shape).
      int64_t InputShape[] = {1, 96, 52, 52};
      int64_t OutputShape[] = {1, 96, 52, 52};
      int64_t Padding[] = {1, 1, 1, 1};
      ASSERT_TRUE(MlasNchwcConvWinogradEligible(InputShape, OutputShape, Padding));
    }

    //
    // Correctness matrix. Includes the yolox hot shapes, non-multiple-of-4
    // spatial extents (partial edge tiles), pad 0 and 1, batching, bias and
    // activation variants.
    //

    static const struct {
      size_t Cin;
      size_t Cout;
      size_t H;
      size_t W;
    } Shapes[] = {
        {16, 16, 12, 12},    // minimum channels, exact tile grid
        {32, 32, 104, 104},  // yolox
        {48, 48, 52, 52},    // yolox
        {96, 96, 52, 52},    // yolox hottest
        {96, 96, 26, 26},    // yolox
        {96, 96, 13, 13},    // yolox, partial tiles both dims
        {192, 192, 13, 13},  // yolox
        {96, 192, 26, 26},   // Cin != Cout
        {192, 96, 26, 26},
        {32, 64, 27, 31},    // odd spatial, partial tiles
        {64, 32, 9, 9},      // small spatial
    };

    for (const auto& s : Shapes) {
      for (size_t pad = 0; pad <= 1; pad++) {
        TestWinograd(1, s.Cin, s.H, s.W, s.Cout, pad, pad, true, MlasIdentityActivation);
      }
      TestWinograd(1, s.Cin, s.H, s.W, s.Cout, 1, 1, false, MlasIdentityActivation);
      TestWinograd(1, s.Cin, s.H, s.W, s.Cout, 1, 1, true, MlasReluActivation);
    }

    // Activation post-pass path and batching.
    TestWinograd(1, 96, 52, 52, 96, 1, 1, true, MlasHardSigmoidActivation);
    TestWinograd(2, 48, 26, 26, 48, 1, 1, true, MlasIdentityActivation);
    TestWinograd(2, 32, 33, 17, 64, 1, 1, true, MlasReluActivation);

    // Asymmetric padding (top/left 1, bottom/right 0 handled via explicit call).
    {
      ActivationKind_ = MlasIdentityActivation;
      const size_t Cin = 32, Cout = 32, H = 24, W = 24;
      const size_t OH = H + 1 - 2, OW = W + 1 - 2;
      const size_t OutputElements = Cout * OH * OW;
      const float* Input = MlasConv2DTest<Threaded>::BufferInput.GetBuffer(Cin * H * W);
      const float* Filter = MlasConv2DTest<Threaded>::BufferFilter.GetBuffer(Cout * Cin * 9);
      const float* Bias = MlasConv2DTest<Threaded>::BufferBias.GetBuffer(Cout);
      float* Output = MlasConv2DTest<Threaded>::BufferOutput.GetBuffer(OutputElements);
      float* OutputRef = MlasConv2DTest<Threaded>::BufferOutputReference.GetBuffer(OutputElements);

      this->MlasConv2D(1, 1, Cin, H, W, Cout, 3, 3, 1, 1, 0, 0, 1, 1, 1, 1, OH, OW,
                       Input, Filter, Bias, Output);

      MLAS_ACTIVATION Activation;
      Activation.ActivationKind = MlasIdentityActivation;
      MlasConv2DTest<Threaded>::ReferenceConv2DWithOptions(
          1, 1, Cin, H, W, Cout, 3, 3, 1, 1, 1, 1, 1, 1, OH, OW,
          Input, Filter, Bias, Activation, 0.0f, nullptr, OutputRef);

      for (size_t i = 0; i < OutputElements; i++) {
        ASSERT_TRUE(CloseEnough(Output[i], OutputRef[i]))
            << " @" << i << " got " << Output[i] << " expected " << OutputRef[i]
            << " asymmetric padding case";
      }
    }
  }

  void ExecuteLong(void) override {
    ExecuteShort();
  }
};

template <typename TMlasTester>
class WinogradShortExecuteTest : public MlasTestFixture<TMlasTester> {
 public:
  void TestBody() override {
    MlasTestFixture<TMlasTester>::mlas_tester->ExecuteShort();
  }

  static size_t RegisterShortExecute() {
    testing::RegisterTest(
        TMlasTester::GetTestSuiteName(),
        "ShortExecute",
        nullptr,
        "ShortExecute",
        __FILE__,
        __LINE__,
        [=]() -> MlasTestFixture<TMlasTester>* {
          return new WinogradShortExecuteTest<TMlasTester>();
        });
    return 1;
  }
};

static size_t Conv2dNchwcWinogradRegistShortExecute() {
  size_t count = 0;

  if (MlasNchwcConvWinogradSupported()) {
    count += WinogradShortExecuteTest<MlasNchwcConvWinogradTest<false>>::RegisterShortExecute();
    if (GetMlasThreadPool() != nullptr) {
      count += WinogradShortExecuteTest<MlasNchwcConvWinogradTest<true>>::RegisterShortExecute();
    }
  }

  return count;
}

static size_t Conv2dNchwcWinogradRegistLongExecute() {
  size_t count = 0;

  if (MlasNchwcConvWinogradSupported()) {
    count += MlasLongExecuteTests<MlasNchwcConvWinogradTest<false>>::RegisterLongExecute();
    if (GetMlasThreadPool() != nullptr) {
      count += MlasLongExecuteTests<MlasNchwcConvWinogradTest<true>>::RegisterLongExecute();
    }
  }

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? Conv2dNchwcWinogradRegistShortExecute() : Conv2dNchwcWinogradRegistLongExecute();
});
