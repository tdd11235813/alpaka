/**
 * \file
 * Copyright 2014-2015 Benjamin Worpitz, Rene Widera
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/Common.hpp>               // ALPAKA_FN_*, __HIPCC__

#include <alpaka/block/shared/dyn/Traits.hpp>   // AllocVar

#include <type_traits>                          // std::is_trivially_default_constructible, std::is_trivially_destructible

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace dyn
            {
                //#############################################################################
                //! The GPU HIP block shared memory allocator.
                //#############################################################################
                class BlockSharedMemDynHipBuiltIn
                {
                public:
                    using BlockSharedMemDynBase = BlockSharedMemDynHipBuiltIn;

                    //-----------------------------------------------------------------------------
                    //! Default constructor.
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynHipBuiltIn() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_HIP_ONLY BlockSharedMemDynHipBuiltIn(BlockSharedMemDynHipBuiltIn const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_HIP_ONLY BlockSharedMemDynHipBuiltIn(BlockSharedMemDynHipBuiltIn &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_HIP_ONLY auto operator=(BlockSharedMemDynHipBuiltIn const &) -> BlockSharedMemDynHipBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_HIP_ONLY auto operator=(BlockSharedMemDynHipBuiltIn &&) -> BlockSharedMemDynHipBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemDynHipBuiltIn() = default;
                };

                namespace traits
                {
                    //#############################################################################
                    //!
                    //#############################################################################
                    template<
                        typename T>
                    struct GetMem<
                        T,
                        BlockSharedMemDynHipBuiltIn>
                    {
                        //-----------------------------------------------------------------------------
                        //
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_ACC_HIP_ONLY static auto getMem(
                            block::shared::dyn::BlockSharedMemDynHipBuiltIn const &)
                        -> T *
                        {
                            // Because unaligned access to variables is not allowed in device code,
                            // we have to use the widest possible type to have all types aligned correctly.
                            // See: http://docs.nvidia.com/hip/hip-c-programming-guide/index.html#shared
                            // http://docs.nvidia.com/hip/hip-c-programming-guide/index.html#vector-types
                            extern __shared__ float4 shMem[];
                            return reinterpret_cast<T *>(shMem);
                        }
                    };
                }
            }
        }
    }
}

#endif
