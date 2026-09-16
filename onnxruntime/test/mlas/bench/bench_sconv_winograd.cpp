// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// Benchmarks comparing the NCHWc Winograd F(4x4,3x3) convolution against the
// direct NCHWc algorithm on 3x3 stride-1 pad-1 shapes.
//

#include "mlas.h"
#include "bench_util.h"
#include "core/util/thread_utils.h"

#include <memory>
#include <stdexcept>
#include <vector>

static const std::vector<std::string> conv_winograd_bench_arg_names = {"IC", "OC", "HW", "Threads"};

static std::unique_ptr<onnxruntime::concurrency::ThreadPool> CreateBenchThreadPool(int threads) {
  if (threads <= 1) {
    return nullptr;
  }
  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = threads;
  tpo.auto_set_affinity = true;
  return onnxruntime::concurrency::CreateThreadPool(
      &onnxruntime::Env::Default(), tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP);
}

static void SCONV_NCHWC_WINOGRAD(benchmark::State& state, bool use_winograd) {
  const size_t Cin = static_cast<size_t>(state.range(0));
  const size_t Cout = static_cast<size_t>(state.range(1));
  const size_t HW = static_cast<size_t>(state.range(2));
  const int Threads = static_cast<int>(state.range(3));

  if (!MlasNchwcConvWinogradSupported()) {
    state.SkipWithError("Winograd not supported on this platform");
    return;
  }

  int64_t InputShape[] = {1, int64_t(Cin), int64_t(HW), int64_t(HW)};
  int64_t OutputShape[] = {1, int64_t(Cout), int64_t(HW), int64_t(HW)};
  int64_t KernelShape[] = {3, 3};
  int64_t DilationShape[] = {1, 1};
  int64_t Padding[] = {1, 1, 1, 1};
  int64_t StrideShape[] = {1, 1};

  auto Input = RandomVectorUniform(Cin * HW * HW, -1.0f, 1.0f);
  auto Filter = RandomVectorUniform(Cout * Cin * 9, -1.0f, 1.0f);
  auto Bias = RandomVectorUniform(Cout, -1.0f, 1.0f);
  std::vector<float> Output(Cout * HW * HW);

  int64_t FilterShape[] = {int64_t(Cout), int64_t(Cin), 3, 3};
  std::vector<float> ReorderedFilter(Cout * Cin * 9);
  MlasReorderFilterOIHWBiBo(FilterShape, Filter.data(), ReorderedFilter.data());

  MLAS_ACTIVATION Activation;
  Activation.ActivationKind = MlasIdentityActivation;

  auto tp = CreateBenchThreadPool(Threads);

  if (use_winograd) {
    const size_t TransformedSize = MlasNchwcConvWinogradFilterTransformSize(Cout, Cin, nullptr);
    std::vector<uint8_t> TransformedFilter(TransformedSize + 64);
    void* AlignedFilter = reinterpret_cast<void*>(
        (reinterpret_cast<uintptr_t>(TransformedFilter.data()) + 63) & ~uintptr_t(63));
    MlasNchwcConvWinogradFilterTransform(Cout, Cin, ReorderedFilter.data(), AlignedFilter, nullptr);

    for (auto _ : state) {
      MlasNchwcConvWinograd(InputShape, Padding, OutputShape, Input.data(), AlignedFilter,
                            Bias.data(), Output.data(), &Activation, tp.get(), nullptr);
    }
  } else {
    for (auto _ : state) {
      MlasNchwcConv(InputShape, KernelShape, DilationShape, Padding, StrideShape, OutputShape,
                    1, Input.data(), ReorderedFilter.data(), Bias.data(), Output.data(),
                    &Activation, true, tp.get(), nullptr, false);
    }
  }
}

static void WINOGRAD(benchmark::State& state) { SCONV_NCHWC_WINOGRAD(state, true); }
static void DIRECT(benchmark::State& state) { SCONV_NCHWC_WINOGRAD(state, false); }

static void ConvWinogradCases(benchmark::internal::Benchmark* b) {
  b->ArgNames(conv_winograd_bench_arg_names);
  for (int threads : {1, 4}) {
    b->Args({96, 96, 52, threads});   // yolox hottest
    b->Args({96, 96, 26, threads});   // yolox
    b->Args({48, 48, 52, threads});   // yolox
    b->Args({32, 32, 104, threads});  // yolox
    b->Args({192, 192, 13, threads}); // yolox smallest eligible
    b->Args({32, 32, 208, threads});  // stress: large spatial
  }
}

BENCHMARK(WINOGRAD)->Apply(ConvWinogradCases)->UseRealTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(DIRECT)->Apply(ConvWinogradCases)->UseRealTime()->Unit(benchmark::kMicrosecond);
