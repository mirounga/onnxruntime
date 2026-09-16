// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// Tests for the NCHWc fused Strassen pointwise convolution algorithm.
// Strassen arithmetic is not bitwise identical to direct convolution, so
// results are compared against the im2col+GEMM reference with a relative
// tolerance (CloseEnough).
//

#include "test_conv2d.h"

#include <cmath>
#include <random>
#include <vector>

template <bool Threaded>
class MlasNchwcConvStrassenTest : public MlasConv2DTest<Threaded> {
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
    MLAS_UNREFERENCED_PARAMETER(PaddingLeftHeight);
    MLAS_UNREFERENCED_PARAMETER(PaddingLeftWidth);
    MLAS_UNREFERENCED_PARAMETER(PaddingRightHeight);
    MLAS_UNREFERENCED_PARAMETER(PaddingRightWidth);
    MLAS_UNREFERENCED_PARAMETER(DilationHeight);
    MLAS_UNREFERENCED_PARAMETER(DilationWidth);
    MLAS_UNREFERENCED_PARAMETER(StrideHeight);
    MLAS_UNREFERENCED_PARAMETER(StrideWidth);

    int64_t InputShape[] = {int64_t(BatchCount), int64_t(InputChannels), int64_t(InputHeight), int64_t(InputWidth)};
    int64_t FilterShape[] = {int64_t(FilterCount), int64_t(InputChannels), 1, 1};
    int64_t OutputShape[] = {int64_t(BatchCount), int64_t(FilterCount), int64_t(OutputHeight), int64_t(OutputWidth)};

    size_t NchwcInputElements = BatchCount * InputChannels * InputHeight * InputWidth;
    float* NchwcInput = BufferNchwcInput.GetBuffer(NchwcInputElements);
    ReorderInputNchw(InputShape, Input, NchwcInput);

    float* ReorderedFilter = BufferNchwcFilter.GetBuffer(FilterCount * InputChannels);
    MlasReorderFilterOIHWBiBo(FilterShape, Filter, ReorderedFilter);

    size_t NchwcOutputElements = BatchCount * FilterCount * OutputHeight * OutputWidth;
    float* NchwcOutput = BufferNchwcOutput.GetBuffer(NchwcOutputElements);

    MLAS_ACTIVATION Activation;
    Activation.ActivationKind = ActivationKind_;
    if (ActivationKind_ == MlasHardSigmoidActivation) {
      Activation.Parameters.HardSigmoid.alpha = 0.2f;
      Activation.Parameters.HardSigmoid.beta = 0.5f;
    }

    MlasNchwcConvStrassen(InputShape, OutputShape, NchwcInput, ReorderedFilter,
                          Bias, NchwcOutput, &Activation,
                          MlasConv2DTest<Threaded>::threadpool_);

    MlasReorderOutputNchw(OutputShape, NchwcOutput, Output, MlasConv2DTest<Threaded>::threadpool_);
  }

  void TestStrassen(
      size_t BatchCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      bool WithBias,
      MLAS_ACTIVATION_KIND ActivationKind) {
    const size_t OutputElements = BatchCount * FilterCount * InputHeight * InputWidth;

    const float* Input = MlasConv2DTest<Threaded>::BufferInput.GetBuffer(BatchCount * InputChannels * InputHeight * InputWidth);
    const float* Filter = MlasConv2DTest<Threaded>::BufferFilter.GetBuffer(FilterCount * InputChannels);
    const float* Bias = WithBias ? MlasConv2DTest<Threaded>::BufferBias.GetBuffer(FilterCount) : nullptr;
    float* Output = MlasConv2DTest<Threaded>::BufferOutput.GetBuffer(OutputElements);
    float* OutputRef = MlasConv2DTest<Threaded>::BufferOutputReference.GetBuffer(OutputElements);

    ActivationKind_ = ActivationKind;

    this->MlasConv2D(BatchCount, 1, InputChannels, InputHeight, InputWidth, FilterCount,
                     1, 1, 0, 0, 0, 0, 1, 1, 1, 1, InputHeight, InputWidth,
                     Input, Filter, Bias, Output);

    MLAS_ACTIVATION Activation;
    Activation.ActivationKind = ActivationKind;
    if (ActivationKind == MlasHardSigmoidActivation) {
      Activation.Parameters.HardSigmoid.alpha = 0.2f;
      Activation.Parameters.HardSigmoid.beta = 0.5f;
    }

    const float* ReferenceBias = Bias;
    if (ReferenceBias == nullptr) {
      ZeroBias_.assign(FilterCount, 0.0f);
      ReferenceBias = ZeroBias_.data();
    }

    MLAS_ACTIVATION ReferenceActivation = Activation;
    if (ActivationKind != MlasIdentityActivation && ActivationKind != MlasReluActivation) {
      ReferenceActivation.ActivationKind = MlasIdentityActivation;
    }

    MlasConv2DTest<Threaded>::ReferenceConv2DWithOptions(
        BatchCount, 1, InputChannels, InputHeight, InputWidth, FilterCount,
        1, 1, 0, 0, 1, 1, 1, 1, InputHeight, InputWidth,
        Input, Filter, ReferenceBias, ReferenceActivation, 0.0f, nullptr, OutputRef);

    if (ActivationKind == MlasHardSigmoidActivation) {
      const float alpha = Activation.Parameters.HardSigmoid.alpha;
      const float beta = Activation.Parameters.HardSigmoid.beta;
      for (size_t i = 0; i < OutputElements; i++) {
        OutputRef[i] = std::min(std::max(alpha * OutputRef[i] + beta, 0.0f), 1.0f);
      }
    }

    float MaxAbs = 0.0f;
    float MaxRel = 0.0f;

    for (size_t i = 0; i < OutputElements; i++) {
      ASSERT_TRUE(CloseEnough(Output[i], OutputRef[i]))
          << " @" << i << " got " << Output[i] << " expected " << OutputRef[i]
          << " shape " << InputChannels << "->" << FilterCount
          << " " << InputHeight << "x" << InputWidth
          << " bias " << WithBias << " act " << int(ActivationKind);

      const float Abs = std::abs(Output[i] - OutputRef[i]);
      const float Scale = std::max(std::abs(Output[i]), std::abs(OutputRef[i]));
      MaxAbs = std::max(MaxAbs, Abs);
      if (Scale > 1e-4f) {
        MaxRel = std::max(MaxRel, Abs / Scale);
      }
    }

    WorstAbs_ = std::max(WorstAbs_, MaxAbs);
    WorstRel_ = std::max(WorstRel_, MaxRel);
  }

  //
  // Adversarial numerics: Strassen reorders summation, so exercise inputs
  // built to provoke cancellation rather than only well conditioned noise.
  //
  //   0 mixed magnitudes across channels   1 mostly zero, sparse large values
  //   2 alternating signs (cancellation)   3 values straddling the ReLU hinge
  //

  void TestAdversarial(size_t InputChannels, size_t InputHeight, size_t InputWidth,
                       size_t FilterCount, int Pattern, MLAS_ACTIVATION_KIND ActivationKind) {
    const size_t InputElements = InputChannels * InputHeight * InputWidth;
    const size_t FilterElements = FilterCount * InputChannels;
    const size_t OutputElements = FilterCount * InputHeight * InputWidth;

    float* Input = MlasConv2DTest<Threaded>::BufferInput.GetBuffer(InputElements);
    float* Filter = MlasConv2DTest<Threaded>::BufferFilter.GetBuffer(FilterElements);
    float* Bias = MlasConv2DTest<Threaded>::BufferBias.GetBuffer(FilterCount);
    float* Output = MlasConv2DTest<Threaded>::BufferOutput.GetBuffer(OutputElements);
    float* OutputRef = MlasConv2DTest<Threaded>::BufferOutputReference.GetBuffer(OutputElements);

    std::mt19937 rng(20260809u + unsigned(Pattern));
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);

    for (size_t i = 0; i < InputElements; i++) {
      switch (Pattern) {
        case 0: {
          const int mag = int((i / (InputHeight * InputWidth)) % 5) - 2;
          Input[i] = unit(rng) * std::pow(10.0f, float(mag) * 3.0f);
          break;
        }
        case 1:
          Input[i] = ((i % 37) == 0) ? unit(rng) * 1.0e3f : 0.0f;
          break;
        case 2:
          Input[i] = ((i & 1) ? 1.0f : -1.0f) * (1.0f + unit(rng) * 1.0e-3f);
          break;
        default:
          Input[i] = unit(rng) * 1.0e-3f;
          break;
      }
    }

    for (size_t i = 0; i < FilterElements; i++) {
      Filter[i] = (Pattern == 2) ? ((i & 1) ? 1.0f : -1.0f) : unit(rng);
    }

    for (size_t i = 0; i < FilterCount; i++) {
      Bias[i] = (Pattern == 3) ? 0.0f : unit(rng) * ((Pattern == 0) ? 1.0e4f : 1.0f);
    }

    ActivationKind_ = ActivationKind;

    this->MlasConv2D(1, 1, InputChannels, InputHeight, InputWidth, FilterCount,
                     1, 1, 0, 0, 0, 0, 1, 1, 1, 1, InputHeight, InputWidth,
                     Input, Filter, Bias, Output);

    MLAS_ACTIVATION Activation;
    Activation.ActivationKind = ActivationKind;

    MlasConv2DTest<Threaded>::ReferenceConv2DWithOptions(
        1, 1, InputChannels, InputHeight, InputWidth, FilterCount,
        1, 1, 0, 0, 1, 1, 1, 1, InputHeight, InputWidth,
        Input, Filter, Bias, Activation, 0.0f, nullptr, OutputRef);

    //
    // Cancellation-heavy inputs make relative error meaningless where the
    // result is near zero, so compare against the magnitude of the terms that
    // produced it: the accumulated scale of one output's dot product.
    //

    float RefScale = 0.0f;
    for (size_t i = 0; i < OutputElements; i++) {
      RefScale = std::max(RefScale, std::abs(OutputRef[i]));
    }
    const float Floor = std::max(RefScale, 1.0f) * 1.0e-4f;

    float MaxAbs = 0.0f;
    for (size_t i = 0; i < OutputElements; i++) {
      const float Abs = std::abs(Output[i] - OutputRef[i]);
      MaxAbs = std::max(MaxAbs, Abs);
      ASSERT_LE(Abs, Floor)
          << " pattern " << Pattern << " @" << i << " got " << Output[i]
          << " expected " << OutputRef[i] << " scale " << RefScale;
    }
    WorstAbs_ = std::max(WorstAbs_, MaxAbs);
  }

  MatrixGuardBuffer<float> BufferNchwcInput;
  MatrixGuardBuffer<float> BufferNchwcFilter;
  MatrixGuardBuffer<float> BufferNchwcOutput;
  std::vector<float> ZeroBias_;
  float WorstAbs_ = 0.0f;
  float WorstRel_ = 0.0f;

 public:
  static const char* GetTestSuiteName(void) {
    static const std::string suite_name(Threaded ? "Conv2dNchwcStrassen_Threaded" : "Conv2dNchwcStrassen_SingleThread");
    return suite_name.c_str();
  }

  MlasNchwcConvStrassenTest() : MlasConv2DTest<Threaded>() {}

  void ExecuteShort(void) {
    //
    // Eligibility predicate checks.
    //

    {
      // Channels not a multiple of 32.
      int64_t InputShape[] = {1, 48, 16, 16};
      int64_t OutputShape[] = {1, 64, 16, 16};
      ASSERT_FALSE(MlasNchwcConvStrassenEligible(InputShape, OutputShape));
    }
    {
      // Spatial too small.
      int64_t InputShape[] = {1, 256, 7, 7};
      int64_t OutputShape[] = {1, 256, 7, 7};
      ASSERT_FALSE(MlasNchwcConvStrassenEligible(InputShape, OutputShape));
    }
    {
      // Eligible reference case (mobileclip fc1).
      int64_t InputShape[] = {1, 256, 16, 16};
      int64_t OutputShape[] = {1, 768, 16, 16};
      ASSERT_TRUE(MlasNchwcConvStrassenEligible(InputShape, OutputShape));
    }

    //
    // Correctness matrix: the mobileclip fc shape families plus ragged
    // half-Cout filter sets, odd spatial splits, batching, bias and
    // activation variants.
    //

    static const struct {
      size_t Cin;
      size_t Cout;
      size_t H;
      size_t W;
    } Shapes[] = {
        {64, 192, 64, 64},    // S0 fc1 (half-out 6 blocks: ragged 4+2)
        {192, 64, 64, 64},    // S0 fc2 (half-out 2 blocks: FC=2)
        {128, 384, 32, 32},   // S2 fc1
        {384, 128, 32, 32},   // S2 fc2
        {256, 768, 16, 16},   // S4 fc1
        {768, 256, 16, 16},   // S4 fc2
        {512, 1536, 8, 8},    // S7 fc1
        {1536, 512, 8, 8},    // S7 fc2
        {64, 64, 8, 8},       // minimum eligible size
        {96, 96, 12, 12},     // half-out 3 blocks (FC=3), odd-ish spatial
        {64, 128, 10, 13},    // non-square, HW=130 (odd half-split of 65)
    };

    for (const auto& s : Shapes) {
      TestStrassen(1, s.Cin, s.H, s.W, s.Cout, true, MlasIdentityActivation);
      TestStrassen(1, s.Cin, s.H, s.W, s.Cout, false, MlasIdentityActivation);
      TestStrassen(1, s.Cin, s.H, s.W, s.Cout, true, MlasReluActivation);
    }

    // Activation post-pass path and batching.
    TestStrassen(1, 256, 16, 16, 768, true, MlasHardSigmoidActivation);
    TestStrassen(2, 128, 32, 32, 384, true, MlasIdentityActivation);
    TestStrassen(3, 64, 16, 16, 64, true, MlasReluActivation);

    // Adversarial numerics across the hot shape families.
    for (int pattern = 0; pattern < 4; pattern++) {
      const MLAS_ACTIVATION_KIND act =
          (pattern == 3) ? MlasReluActivation : MlasIdentityActivation;
      TestAdversarial(128, 32, 32, 384, pattern, act);
      TestAdversarial(256, 16, 16, 768, pattern, act);
      TestAdversarial(64, 64, 64, 192, pattern, act);
    }

    std::cout << "[ strassen ] worst abs error " << WorstAbs_
              << ", worst rel error " << WorstRel_ << std::endl;
  }

  void ExecuteLong(void) override {
    ExecuteShort();
  }
};

template <typename TMlasTester>
class StrassenShortExecuteTest : public MlasTestFixture<TMlasTester> {
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
          return new StrassenShortExecuteTest<TMlasTester>();
        });
    return 1;
  }
};

static size_t Conv2dNchwcStrassenRegistShortExecute() {
  size_t count = 0;

  if (MlasNchwcConvStrassenSupported()) {
    count += StrassenShortExecuteTest<MlasNchwcConvStrassenTest<false>>::RegisterShortExecute();
    if (GetMlasThreadPool() != nullptr) {
      count += StrassenShortExecuteTest<MlasNchwcConvStrassenTest<true>>::RegisterShortExecute();
    }
  }

  return count;
}

static size_t Conv2dNchwcStrassenRegistLongExecute() {
  size_t count = 0;

  if (MlasNchwcConvStrassenSupported()) {
    count += MlasLongExecuteTests<MlasNchwcConvStrassenTest<false>>::RegisterLongExecute();
    if (GetMlasThreadPool() != nullptr) {
      count += MlasLongExecuteTests<MlasNchwcConvStrassenTest<true>>::RegisterLongExecute();
    }
  }

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? Conv2dNchwcStrassenRegistShortExecute() : Conv2dNchwcStrassenRegistLongExecute();
});
