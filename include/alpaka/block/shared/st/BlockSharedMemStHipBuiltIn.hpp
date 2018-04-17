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

#include <alpaka/core/Common.hpp>           // ALPAKA_FN_*, __HIPCC__

#include <alpaka/block/shared/st/Traits.hpp>// AllocVar

#include <type_traits>                      // std::is_trivially_default_constructible, std::is_trivially_destructible
#include <cstdint>                          // uint8_t

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace st
            {
                //#############################################################################
                //! The GPU HIP block shared memory allocator.
                //#############################################################################
                class BlockSharedMemStHipBuiltIn
                {
                public:
                    using BlockSharedMemStBase = BlockSharedMemStHipBuiltIn;

                    //-----------------------------------------------------------------------------
                    //! Default constructor.
                    //-----------------------------------------------------------------------------
                    BlockSharedMemStHipBuiltIn() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_HIP_ONLY BlockSharedMemStHipBuiltIn(BlockSharedMemStHipBuiltIn const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_HIP_ONLY BlockSharedMemStHipBuiltIn(BlockSharedMemStHipBuiltIn &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_HIP_ONLY auto operator=(BlockSharedMemStHipBuiltIn const &) -> BlockSharedMemStHipBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_HIP_ONLY auto operator=(BlockSharedMemStHipBuiltIn &&) -> BlockSharedMemStHipBuiltIn & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemStHipBuiltIn() = default;
                };

                namespace traits
                {
                    //#############################################################################
                    //!
                    //#############################################################################
                    template<
                        typename T,
                        std::size_t TuniqueId>
                    struct AllocVar<
                        T,
                        TuniqueId,
                        BlockSharedMemStHipBuiltIn>
                    {
                        //-----------------------------------------------------------------------------
                        //
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_ACC_HIP_ONLY static auto allocVar(
                            block::shared::st::BlockSharedMemStHipBuiltIn const &)
                        -> T &
                        {
                            __shared__ uint8_t shMem alignas(alignof(T)) [sizeof(T)];
                            return *(
                                reinterpret_cast<T*>( shMem ));
                        }
                    };
                    //#############################################################################
                    //!
                    //#############################################################################
                    template<>
                    struct FreeMem<
                        BlockSharedMemStHipBuiltIn>
                    {
                        //-----------------------------------------------------------------------------
                        //
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_ACC_HIP_ONLY static auto freeMem(
                            block::shared::st::BlockSharedMemStHipBuiltIn const &)
                        -> void
                        {
                            // Nothing to do. HIP block shared memory is automatically freed when all threads left the block.
                        }
                    };
                }
            }
        }
    }
}

#endif
