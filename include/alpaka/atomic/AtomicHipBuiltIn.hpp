/**
 * \file
 * Copyright 2014-2016 Benjamin Worpitz, Rene Widera
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

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>

namespace alpaka
{
    namespace atomic
    {
        //#############################################################################
        //! The GPU HIP accelerator atomic ops.
        //
        //  Atomics can used in the hierarchy level grids, blocks and threads.
        //  Atomics are not guaranteed to be save between devices
        class AtomicHipBuiltIn
        {
        public:

            //-----------------------------------------------------------------------------
            AtomicHipBuiltIn() = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY AtomicHipBuiltIn(AtomicHipBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY AtomicHipBuiltIn(AtomicHipBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY auto operator=(AtomicHipBuiltIn const &) -> AtomicHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_ACC_HIP_ONLY auto operator=(AtomicHipBuiltIn &&) -> AtomicHipBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AtomicHipBuiltIn() = default;
        };

        namespace traits
        {
            //#############################################################################
            //! The specializations to execute the requested atomic ops of the HIP accelerator.
            //-----------------------------------------------------------------------------
            // Add.
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicHipBuiltIn,
                float,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    float * const addr,
                    float const & value)
                -> float
                {
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicHipBuiltIn,
                double,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    double * const addr,
                    double const & value)
                -> double
                {
                    return atomicAdd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            // Sub.
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Sub,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicSub(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Sub,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicSub(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            // Min.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Min,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicMin(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Min,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicMin(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            /*template<
               typename THierarchy>
            struct AtomicOp<
                op::Min,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicMin(addr, value);
                }
            };*/
            //-----------------------------------------------------------------------------
            // Max.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Max,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicMax(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Max,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicMax(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            /*template<
               typename THierarchy>
            struct AtomicOp<
                op::Max,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicMax(addr, value);
                }
            };*/
            //-----------------------------------------------------------------------------
            // Exch.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicExch(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicExch(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicExch(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicHipBuiltIn,
                float,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    float * const addr,
                    float const & value)
                -> float
                {
                    return atomicExch(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            // Inc.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Inc,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicInc(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            // Dec.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Dec,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicDec(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            // And.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicAnd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicAnd(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            /*template<
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicAnd(addr, value);
                }
            };*/
            //-----------------------------------------------------------------------------
            // Or.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicOr(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                //
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicOr(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            /*template<
               typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicOr(addr, value);
                }
            };*/
            //-----------------------------------------------------------------------------
            // Xor.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & value)
                -> int
                {
                    return atomicXor(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicXor(addr, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            /*template<
               typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicXor(addr, value);
                }
            };*/
            //-----------------------------------------------------------------------------
            // Cas.
            //-----------------------------------------------------------------------------
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Cas,
                atomic::AtomicHipBuiltIn,
                int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    int * const addr,
                    int const & compare,
                    int const & value)
                -> int
                {
                    return atomicCAS(addr, compare, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Cas,
                atomic::AtomicHipBuiltIn,
                unsigned int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned int * const addr,
                    unsigned int const & compare,
                    unsigned int const & value)
                -> unsigned int
                {
                    return atomicCAS(addr, compare, value);
                }
            };
            //-----------------------------------------------------------------------------
            //! The GPU HIP accelerator atomic operation.
            //-----------------------------------------------------------------------------
            template<
                typename THierarchy>
            struct AtomicOp<
                op::Cas,
                atomic::AtomicHipBuiltIn,
                unsigned long long int,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_HIP_ONLY static auto atomicOp(
                    atomic::AtomicHipBuiltIn const &,
                    unsigned long long int * const addr,
                    unsigned long long int const & compare,
                    unsigned long long int const & value)
                -> unsigned long long int
                {
                    return atomicCAS(addr, compare, value);
                }
            };
        }
    }
}

#endif
