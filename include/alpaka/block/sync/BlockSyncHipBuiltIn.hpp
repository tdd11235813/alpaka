/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_*, __HIPCC__


#include <alpaka/block/sync/Traits.hpp> // SyncBlockThreads

namespace alpaka
{
    namespace block
    {
        namespace sync
        {
            //#############################################################################
            //! The GPU HIP block synchronization.
            //#############################################################################
            class BlockSyncHipBuiltIn
            {
            public:
                using BlockSyncBase = BlockSyncHipBuiltIn;

                //-----------------------------------------------------------------------------
                //! Default constructor.
                //-----------------------------------------------------------------------------
                BlockSyncHipBuiltIn() = default;
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY BlockSyncHipBuiltIn(BlockSyncHipBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY BlockSyncHipBuiltIn(BlockSyncHipBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY auto operator=(BlockSyncHipBuiltIn const &) -> BlockSyncHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY auto operator=(BlockSyncHipBuiltIn &&) -> BlockSyncHipBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                /*virtual*/ ~BlockSyncHipBuiltIn() = default;
            };

            namespace traits
            {
                //#############################################################################
                //!
                //#############################################################################
                template<>
                struct SyncBlockThreads<
                    BlockSyncHipBuiltIn>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_HIP_ONLY static auto syncBlockThreads(
                        block::sync::BlockSyncHipBuiltIn const & /*blockSync*/)
                    -> void
                    {
                        __syncthreads();
                    }
                };

                //#############################################################################
                //!
                //#############################################################################
                template<>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::Count,
                    BlockSyncHipBuiltIn>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_HIP_ONLY static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncHipBuiltIn const & /*blockSync*/,
                        int predicate)
                    -> int
                    {
                        return __syncthreads_count(predicate);
                    }
                };

                //#############################################################################
                //!
                //#############################################################################
                template<>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::LogicalAnd,
                    BlockSyncHipBuiltIn>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_HIP_ONLY static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncHipBuiltIn const & /*blockSync*/,
                        int predicate)
                    -> int
                    {
                        return __syncthreads_and(predicate);
                    }
                };

                //#############################################################################
                //!
                //#############################################################################
                template<>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::LogicalOr,
                    BlockSyncHipBuiltIn>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_HIP_ONLY static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncHipBuiltIn const & /*blockSync*/,
                        int predicate)
                    -> int
                    {
                        return __syncthreads_or(predicate);
                    }
                };
            }
        }
    }
}

#endif
