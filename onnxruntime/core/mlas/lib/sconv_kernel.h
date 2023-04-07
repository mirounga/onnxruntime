/*++

Copyright (c) Microsoft Corporation.  All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_kernel.h

Abstract:

    This module contains the private data structures and procedure prototypes
    for the convolution primitives.

--*/

typedef
void
(MLAS_CONV_KERNEL_ROUTINE)(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output
    );

MLAS_CONV_KERNEL_ROUTINE*
MlasGetSlidingConvolutionKernel(const MLAS_CONV_PARAMETERS* Parameters);
