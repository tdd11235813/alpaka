/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/math/sincos/Traits.hpp>

#include <cuda_runtime.h>
#include <type_traits>


namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! sincos.
        class SinCosCudaBuiltIn
        {
        public:
            using SinCosBase = SinCosCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################

            //! sincos trait specialization.
            template<>
            struct SinCos<
                SinCosCudaBuiltIn,
                double>
            {
                __device__ static auto sincos(
                    SinCosCudaBuiltIn const & sin_cos, // must not be 'sincos' (e.g. gcc4.9+nvcc8 cannot distinguish variable & function name)
                    double const & arg,
                    double & result_sin,
                    double & result_cos)
                -> void
                {
                    alpaka::ignore_unused(sin_cos);
                    ::sincos(arg, &result_sin, &result_cos);
                }
            };

            //! sincos trait specialization.
            template<>
            struct SinCos<
                SinCosCudaBuiltIn,
                float>
            {
                __device__ static auto sincos(
                    SinCosCudaBuiltIn const & sin_cos,
                    float const & arg,
                    float & result_sin,
                    float & result_cos)
                -> void
                {
                    alpaka::ignore_unused(sin_cos);
                    ::sincosf(arg, &result_sin, &result_cos);
                }
            };

        }
    }
}

#endif
