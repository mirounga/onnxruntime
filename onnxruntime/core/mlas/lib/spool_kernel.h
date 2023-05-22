/*++

Copyright (c) Microsoft Corporation.  All rights reserved.

Licensed under the MIT License.

Module Name:

    spool_kernel.h

Abstract:

    This module contains the private data structures and procedure prototypes
    for the pooling primitives.

--*/

//
// Define the parameters to execute segments of a pooling operation on worker
// threads.
//

struct MLAS_POOL_WORK_BLOCK
{
    MLAS_POOLING_KIND PoolingKind;
    size_t InputShape[3];
    size_t InputSize;
    size_t OutputShape[3];
    int64_t KernelShape[3];
    int64_t Padding[6];
    int64_t StrideShape[3];
};

//
// Define the prototype of the pooling kernel routine.
//

typedef
void
(MLAS_POOL_KERNEL_ROUTINE)(
    const MLAS_POOL_WORK_BLOCK* WorkBlock,
    size_t ChannelCount,
    const float* Input,
    float* Output
    );

void
MlasPool1DSlidingKernelMax(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                           size_t ChannelCount,
                           const float* Input,
                           float* Output);

void
MlasPool1DSlidingKernelAvgWithPad(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                  size_t ChannelCount,
                                  const float* Input,
                                  float* Output);

void
MlasPool1DSlidingKernelAvgNoPad(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                size_t ChannelCount,
                                const float* Input,
                                float* Output);

void
MlasPool2DSlidingKernelMax(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                           size_t ChannelCount,
                           const float* Input,
                           float* Output);

void
MlasPool2DSlidingKernelAvgWithPad(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                  size_t ChannelCount,
                                  const float* Input,
                                  float* Output);

void
MlasPool2DSlidingKernelAvgNoPad(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                size_t ChannelCount,
                                const float* Input,
                                float* Output);

void
MlasPool3DSlidingKernelMax(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                           size_t ChannelCount,
                           const float* Input,
                           float* Output);

void
MlasPool3DSlidingKernelAvgWithPad(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                  size_t ChannelCount,
                                  const float* Input,
                                  float* Output);

void
MlasPool3DSlidingKernelAvgNoPad(const MLAS_POOL_WORK_BLOCK* WorkBlock,
                                size_t ChannelCount,
                                const float* Input,
                                float* Output);
